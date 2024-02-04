"""
Minimal Proof of Concept Implementation of
https://arxiv.org/abs/2312.07360
https://arxiv.org/pdf/2312.07360.pdf
https://github.com/CompVis/fm-boosting
This example uses CelebA as an example dataset since we need both 1024x1024 and 256x256 resolution 
images, and CelebA is available as 1024x1024, allowing us to simply downsample to 256x256 during training.
"""
from pathlib import Path
import os
import copy

from absl import app, flags
from PIL import Image
from einops import rearrange
from diffusers import StableDiffusionPipeline

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader as PytorchDataLoader
from torchvision import transforms as T, utils
from torchdyn.core import NeuralODE
from tqdm import trange
from torchvision.utils import make_grid, save_image

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
from torchcfm.models.unet.unet import UNetModel, UNetModelWrapper

#### Setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to(device)

# only need the KL Autoencoder
vae = pipe.vae


#### Helper Classes
class ImageDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png']
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = []
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        print(f'{len(self.paths)} training samples found at {folder}')
        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize(image_size),
            #T.RandomHorizontalFlip(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)


# helperfunction so we can downsample loaded images to 256x256 resolution
def downsample_2d(X, sz):
    """
    Downsamples a stack of square images.
    Args:
        X: a stack of images (batch, channels, ny, ny).
        sz: the desired size of images.
    Returns:
        The downsampled images, a tensor of shape (batch, channel, sz, sz)
    """
    kernel = torch.tensor([[.25, .5, .25], 
                           [.5, 1, .5], 
                           [.25, .5, .25]], device=X.device).reshape(1, 1, 3, 3)
    kernel = kernel.repeat((X.shape[1], 1, 1, 1))
    while sz < X.shape[-1] / 2:
        # Downsample by a factor 2 with smoothing
        mask = torch.ones(1, *X.shape[1:])
        mask = F.conv2d(mask, kernel, groups=X.shape[1], stride=2, padding=1)
        X = F.conv2d(X, kernel, groups=X.shape[1], stride=2, padding=1)
        # Normalize the edges and corners.
        X = X = X / mask
    return F.interpolate(X, size=sz, mode='bilinear')

# helper function that allows us to save exponential moving average of model parameters
def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay + source_dict[key].data * (1 - decay)
        )


def generate_samples(model, parallel, savedir, step, net_="normal"):
    """Save 64 generated images (8 x 8) for sanity check along training.
    Parameters
    ----------
    model:
        represents the neural network that we want to generate samples from
    parallel: bool
        represents the parallel training flag. Torchdyn only runs on 1 GPU, we need to send the models from several GPUs to 1 GPU.
    savedir: str
        represents the path where we want to save the generated images
    step: int
        represents the current step of training
    """
    model.eval()
    model_ = copy.deepcopy(model)
    if parallel:
        # Send the models from GPU to CPU for inference with NeuralODE from Torchdyn
        model_ = model_.module.to(device)
    node_ = NeuralODE(model_, solver="euler", sensitivity="adjoint")
    with torch.no_grad():
        traj = node_.trajectory(
            torch.randn(64, 3, 32, 32, device=device),
            t_span=torch.linspace(0, 1, 100, device=device),
        )
        traj = traj[-1, :].view([-1, 3, 32, 32]).clip(-1, 1)
        traj = traj / 2 + 0.5
    save_image(traj, savedir + f"{net_}_generated_FM_images_step_{step}.png", nrow=8)
    model.train()


def infiniteloop(dataloader):
    while True:
        for x in iter(dataloader):
            yield x


#### Flow Matching
FLAGS = flags.FLAGS

flags.DEFINE_string("data_path", "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/data/CelebA/celeba/img_align_celeba", help="path to CelebA directory")
flags.DEFINE_string("model", "otcfm", help="flow matching model type")
flags.DEFINE_string("output_dir", "./ot_cfm_results/", help="output_directory")
# UNet
flags.DEFINE_integer("num_channel", 128, help="base channel of UNet")

# Training
flags.DEFINE_float("lr", 2e-4, help="target learning rate")  # TRY 2e-4
flags.DEFINE_float("grad_clip", 1.0, help="gradient norm clipping")
flags.DEFINE_integer(
    "total_steps", 400001, help="total training steps"
)  # Lipman et al uses 400k but double batch size
flags.DEFINE_integer("warmup", 5000, help="learning rate warmup")
flags.DEFINE_integer("batch_size", 8, help="batch size")  # Lipman et al uses 128
flags.DEFINE_integer("num_workers", 1, help="workers of Dataloader")
flags.DEFINE_float("ema_decay", 0.9999, help="ema decay rate")
flags.DEFINE_bool("parallel", False, help="multi gpu training")

# Evaluation
flags.DEFINE_integer(
    "save_step",
    5000,
    help="frequency of saving checkpoints, 0 to disable during training",
)

# remove for actual script
# import sys
# FLAGS(sys.argv)

def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup


def train(argv):
    print(
        "lr, total_steps, ema decay, save_step:",
        FLAGS.lr,
        FLAGS.total_steps,
        FLAGS.ema_decay,
        FLAGS.save_step,
    )

    # DATASETS/DATALOADER
    celeba_path = Path(FLAGS.data_path)
    assert celeba_path.exists()
    dataset = ImageDataset(folder=celeba_path, image_size=1024)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=FLAGS.num_workers,
        drop_last=True,
    )

    datalooper = infiniteloop(dataloader)
    

    # MODELS
    net_model = UNetModelWrapper(
        dim=(4, 128, 128), # must match channel, h, w of hi res latent
        num_res_blocks=3,               # assuming this is "Depth" in table 8 https://arxiv.org/pdf/2312.07360.pdf#table.caption.28
        num_channels=FLAGS.num_channel, # 128 like in table 8
        channel_mult=[1, 2, 3, 4],      # channel mult from table 8 FacesHQ
        num_heads=4,                # Number of Heads table 8
        num_head_channels=64,       # Head channels table 8
        attention_resolutions="16", # Attention resolutions also table 8
        dropout=0.1,                # added dropout for a bit of regularization
    ).to(
        device
    )  # new dropout + bs of 128

    ema_model = copy.deepcopy(net_model)
    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    if FLAGS.parallel:
        print(
            "Warning: parallel training is performing slightly worse than single GPU training due to statistics computation in dataparallel. We recommend to train over a single GPU, which requires around 8 Gb of GPU memory."
        )
        net_model = torch.nn.DataParallel(net_model)
        ema_model = torch.nn.DataParallel(ema_model)

    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print("Model params: %.2f M" % (model_size / 1024 / 1024))

    #################################
    #            OT-CFM
    #################################

    sigma = 0.0
    if FLAGS.model == "otcfm":
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "icfm":
        FM = ConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "fm":
        FM = TargetConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "si":
        FM = VariancePreservingConditionalFlowMatcher(sigma=sigma)
    else:
        raise NotImplementedError(
            f"Unknown model {FLAGS.model}, must be one of ['otcfm', 'icfm', 'fm', 'si']"
        )

    savedir = FLAGS.output_dir + FLAGS.model + "/"
    os.makedirs(savedir, exist_ok=True)

    with trange(FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            optim.zero_grad()
            hi_res_images = next(datalooper).to(device)
            lo_res_images = downsample_2d(hi_res_images.cpu(), sz=256).to(device)
            with torch.no_grad():
                hi_res_latents = vae.encode(hi_res_images).latent_dist.sample()
                lo_res_latents = vae.encode(lo_res_images).latent_dist.sample()
            # hi_res_latent_shape: torch.Size([1, 4, 128, 128])
            # lo_res_latent_shape: torch.Size([1, 4, 32, 32])
            # we naively upsample the low res latent to 4x128x128
            # then add noise to this
            # then perform flow matching from noised latent to original latent
            # then decode the flow matched latent with the vae decoder
            # naively upsample the low resolution latents
            with torch.no_grad():
                # output size must match last 2 spatial dimensions of hi res latents
                bilinear_latents = F.interpolate(lo_res_latents, size=hi_res_latents.shape[-2:])
            # concatenate noise to bilinear latents
            # TODO: Do this according to cosine schedule 3.2.3 Noise Augmentation in paper
            # TODO: Here I just add noise to the tensor since it would lead to a shape mismatch 
            # during sampling location and conditional flow.
            # TODO: Need reference implementation for this since this shape mismatch is also 
            # present in figure 3 but is not explained.
            # I will just add some noise with the minimum sigma as mentioned in section 8
            # 1e-4
            means = torch.zeros_like(bilinear_latents)
            noise = torch.normal(means, std=1e-4)
            # noise = torch.randn_like(bilinear_latents, )
            #### Now we can do the actual flow matching
            # now we obtain our x1 by just concatenating noise like they did in the paper
            # target
            x1 = hi_res_latents
            # source
            x0 = bilinear_latents + noise
            

            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            # xt = samples drawn from probability path pxt at time t
            # ut = conditional vector field 
            vt = net_model(t, xt)
            # Training objective in the paper: 
            # L2 loss of (net(t, xt) - x1-x0)
            # The conditioning signal x1 (hi_res_latents) remains unchanged
            # the starting point x0 correspoinds to an encoded representation of the image
            # standard flow matching loss
            loss = torch.mean((vt - ut) ** 2)
            # coupling flow matching loss
            # loss = torch.mean((vt - (x1 - x0)) ** 2) (??)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)  # new
            optim.step()
            sched.step()
            ema(net_model, ema_model, FLAGS.ema_decay)  # new
            if step % 1 == 0:
                print(loss.data.cpu().numpy().tolist())
            # sample and Saving the weights
            if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
                # TODO: this does not work yet since we only obtain latents
                # Need an additional decoding step after generating samples.
                # Needs to be adjusted in generate samples function aswell.
                # generate_samples(net_model, FLAGS.parallel, savedir, step, net_="normal")
                # generate_samples(ema_model, FLAGS.parallel, savedir, step, net_="ema")
                torch.save(
                    {
                        "net_model": net_model.state_dict(),
                        "ema_model": ema_model.state_dict(),
                        "sched": sched.state_dict(),
                        "optim": optim.state_dict(),
                        "step": step,
                    },
                    savedir + f"{FLAGS.model}_celeba_weights_step_{step}.pt",
                )


if __name__ == "__main__":
    app.run(train)
