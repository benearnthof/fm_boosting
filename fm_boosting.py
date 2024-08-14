"""
Minimal Proof of Concept Implementation of
https://arxiv.org/abs/2312.07360
https://arxiv.org/pdf/2312.07360.pdf
https://github.com/CompVis/fm-boosting
This example uses CelebA as an example dataset since we need both 1024x1024 and 256x256 resolution 
images, and CelebA is available as 1024x1024, allowing us to simply downsample to 256x256 during training.
# UPDATE: Use simpler conditioning routine like in Flow Matching in Latent Space: https://arxiv.org/pdf/2307.08698
"""
import argparse
import os
import shutil

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from datasets_prep import get_superres_dataset
from models import get_flow_model
from omegaconf import OmegaConf
from torch.multiprocessing import Process
from torchdiffeq import odeint_adjoint as odeint


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def get_weight(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


class WrapperCondFlow(nn.Module):
    def __init__(self, model, cond) -> None:
        super().__init__()
        self.model = model
        self.cond = cond
    def forward(self, t, x):
        x = torch.cat([x, self.cond], 1)
        return self.model(t, x)


def broadcast_params(params):
    for param in params:
        dist.broadcast(param.data, src=0)


def sample_from_model(model, z_0):
    # how to pass the cond
    t = torch.tensor([1.0, 0.0], device="cuda")
    fake_image = odeint(model, z_0, t, atol=1e-8, rtol=1e-8)
    return fake_image


# %%
def train(rank, gpu, args):
    from diffusers.models import AutoencoderKL
    from EMA import EMA

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    device = torch.device("cuda:{}".format(gpu))

    batch_size = args.batch_size

    dataset = get_superres_dataset(args)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=1, rank=rank)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )

    args.layout = False
    model = get_flow_model(args).to(device)
    first_stage_model = AutoencoderKL.from_pretrained(args.pretrained_autoencoder_ckpt, cache_dir="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/data/cache_dir").to(device)
    # TODO: specify cache_dir as argument

    first_stage_model = first_stage_model.eval()
    first_stage_model.train = False
    for param in first_stage_model.parameters():
        param.requires_grad = False

    print("AutoKL size: {:.3f}MB".format(get_weight(first_stage_model)))
    print("FM size: {:.3f}MB".format(get_weight(model)))

    # TODO: check if this works with distributed training and add it
    # broadcast_params(model.parameters())

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    if args.use_ema:
        optimizer = EMA(optimizer, ema_decay=args.ema_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch, eta_min=1e-5)

    # TODO: ddp
    # model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=False)

    exp = args.exp
    parent_dir = "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/data/latent_flow_superres{}".format(args.dataset)

    exp_path = os.path.join(parent_dir, exp)
    if rank == 0:
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
            config_dict = vars(args)
            OmegaConf.save(config_dict, os.path.join(exp_path, "config.yaml"))

    if args.resume:
        checkpoint_file = os.path.join(exp_path, "content.pth")
        checkpoint = torch.load(checkpoint_file, map_location=device)
        init_epoch = checkpoint["epoch"]
        epoch = init_epoch
        model.load_state_dict(checkpoint["model_dict"])
        # load G
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        global_step = checkpoint["global_step"]
        print("=> loaded checkpoint (epoch {})".format(checkpoint["epoch"]))
        del checkpoint
    else:
        global_step, epoch, init_epoch = 0, 0, 0

    for epoch in range(init_epoch, args.num_epoch + 1):
        train_sampler.set_epoch(epoch)
        for iteration, (lowres, hires) in enumerate(data_loader):
            # x_1 = image = target = hires
            x_1 = hires.to(device, non_blocking=True)
            cond = lowres.to(device, non_blocking=True)
            model.zero_grad()
            with torch.no_grad():
                # map to 16x16 latent dim
                z_1 = first_stage_model.encode(x_1).latent_dist.sample().mul_(args.scale_factor)
                # c = condition = lowres
                c = first_stage_model.encode(cond).latent_dist.sample().mul_(args.scale_factor)
            # sample t as batch dim and reshape
            t = torch.rand((z_1.size(0),), device=device)
            t = t.view(-1, 1, 1, 1)
            # z_0 is gaussian we sample from and want to transport to z1
            # needs to have same dimension as target tensor z1
            z_0 = torch.randn_like(z_1)
            # linear interpolation for OT-CMF with added tolerances to avoid edge cases
            v_t = (1 - t) * z_1 + (1e-5 + (1 - 1e-5) * t) * z_0
            u = (1 - 1e-5) * z_0 - z_1 # target velocity
            # bring condition up to shape via simple interpolation
            c = F.interpolate(c, size=v_t.shape[-2:]).to(device, non_blocking=True)
            v_t_cond = torch.cat((v_t, c), dim=1)
            loss = F.mse_loss(model(t.squeeze(), v_t_cond), u)
            loss.backward()
            optimizer.step()
            global_step += 1
            if iteration % 100 == 0:
                if rank == 0:
                    print("epoch {} iteration{}, Loss: {}".format(epoch, iteration, loss.item()))

        if not args.no_lr_decay:
            scheduler.step()

        if rank == 0:
            with torch.no_grad():
                rand = torch.randn_like(z_1)[:4]
                b, h, w = 4, x_1.shape[2], x_1.shape[3]
                
                c = first_stage_model.encode(cond).latent_dist.sample().mul_(args.scale_factor)
                c = F.interpolate(c, size=rand.shape[-2:]).to(device, non_blocking=True)
                
                model_cond = WrapperCondFlow(model, c)
                # sample first then decode with pretrained model
                fake_sample = sample_from_model(model_cond, rand)[-1]
                fake_image = first_stage_model.decode(fake_sample / args.scale_factor).sample
            
            torchvision.utils.save_image(
                hires,
                os.path.join(exp_path, "image_epoch_hires_{}.png".format(epoch)),
                normalize=True,
            )
            torchvision.utils.save_image(
                fake_image,
                os.path.join(exp_path, "image_epoch_{}.png".format(epoch)),
                normalize=True,
            )
            del fake_image, fake_sample, model_cond, c
            if args.save_content:
                if epoch % args.save_content_every == 0:
                    print("Saving content.")
                    content = {
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "args": args,
                        "model_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    }

                    torch.save(content, os.path.join(exp_path, "content.pth"))

            if epoch % args.save_ckpt_every == 0:
                if args.use_ema:
                    optimizer.swap_parameters_with_ema(store_params_in_ema=True)

                torch.save(
                    model.state_dict(),
                    os.path.join(exp_path, "model_{}.pth".format(epoch)),
                )
                if args.use_ema:
                    optimizer.swap_parameters_with_ema(store_params_in_ema=True)


def init_processes(rank, size, fn, args):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = args.master_address
    os.environ["MASTER_PORT"] = args.master_port
    torch.cuda.set_device(args.local_rank)
    gpu = args.local_rank
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=size)
    fn(rank, gpu, args)
    dist.barrier()
    cleanup()


def cleanup():
    dist.destroy_process_group()


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser("ddgan parameters")
    parser.add_argument("--seed", type=int, default=1024, help="seed used for initialization")
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--image_size", type=int, default=32, help="size of image")
    parser.add_argument("--scale_factor", type=float, default=0.18215, help="size of image")
    parser.add_argument("--num_in_channels", type=int, default=3, help="in channel image")
    parser.add_argument("--num_out_channels", type=int, default=3, help="in channel image")
    parser.add_argument("--nf", type=int, default=256, help="channel of image")
    parser.add_argument("--centered", action="store_false", default=True, help="-1,1 scale")
    parser.add_argument("--resamp_with_conv", type=bool, default=True)
    parser.add_argument(
        "--num_res_blocks",
        type=int,
        default=2,
        help="number of resnet blocks per scale",
    )
    parser.add_argument("--num_heads", type=int, default=4, help="number of head")
    parser.add_argument("--num_head_upsample", type=int, default=-1, help="number of head upsample")
    parser.add_argument("--num_head_channels", type=int, default=-1, help="number of head channels")
    parser.add_argument(
        "--attn_resolutions",
        nargs="+",
        type=int,
        default=(16,),
        help="resolution of applying attention",
    )
    parser.add_argument(
        "--ch_mult",
        nargs="+",
        type=int,
        default=(1, 1, 2, 2, 4, 4),
        help="channel mult",
    )
    parser.add_argument("--dropout", type=float, default=0.0, help="drop-out rate")
    parser.add_argument("--num_classes", type=int, default=None, help="num classes")
    parser.add_argument("--use_scale_shift_norm", type=bool, default=True)
    parser.add_argument("--resblock_updown", type=bool, default=False)
    parser.add_argument("--use_new_attention_order", type=bool, default=False)
    parser.add_argument("--pretrained_autoencoder_ckpt", type=str, default="stabilityai/sd-vae-ft-mse")
    # geenrator and training
    parser.add_argument("--exp", default="experiment_cifar_default", help="name of experiment")
    parser.add_argument("--dataset", default="cifar10", help="name of dataset")
    parser.add_argument("--num_timesteps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128, help="input batch size")
    parser.add_argument("--num_epoch", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate g")
    parser.add_argument("--beta1", type=float, default=0.5, help="beta1 for adam")
    parser.add_argument("--beta2", type=float, default=0.9, help="beta2 for adam")
    parser.add_argument("--no_lr_decay", action="store_true", default=False)
    parser.add_argument("--use_ema", action="store_true", default=False, help="use EMA or not")
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="decay rate for EMA")
    parser.add_argument("--save_content", action="store_true", default=False)
    parser.add_argument(
        "--save_content_every",
        type=int,
        default=10,
        help="save content for resuming every x epochs",
    )
    parser.add_argument("--save_ckpt_every", type=int, default=25, help="save ckpt every x epochs")
    # ddp
    parser.add_argument(
        "--num_proc_node",
        type=int,
        default=1,
        help="The number of nodes in multi node env.",
    )
    parser.add_argument("--num_process_per_node", type=int, default=1, help="number of gpus")
    parser.add_argument("--node_rank", type=int, default=0, help="The index of node.")
    parser.add_argument("--local_rank", type=int, default=0, help="rank of process in the node")
    parser.add_argument("--master_address", type=str, default="127.0.0.1", help="address for master")
    parser.add_argument("--master_port", type=str, default="6255", help="address for master")

    # torch.multiprocessing.set_start_method('spawn', force=True)# good solution !!!!

    args = parser.parse_args()

# for debugging
# args.exp = "superres_kl"
# args.dataset="celeba_256"
# args.batch_size=64
# args.lr=5e-5
# args.scale_factor=0.18215
# args.num_epoch=500
# args.image_size=128
# args.num_in_channels=8
# args.num_out_channels=4
# args.ch_mult=(1, 2, 3, 4)
# args.attn_resolution=(16, 8)
# args.num_process_per_node=1
# args.save_content=True

    args.world_size = args.num_proc_node * args.num_process_per_node
    size = args.num_process_per_node

    if size > 1:
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_rank = global_rank
            print("Node rank %d, local proc %d, global proc %d" % (args.node_rank, rank, global_rank))
            p = Process(target=init_processes, args=(global_rank, global_size, train, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        print("starting in debug mode")

        init_processes(0, size, train, args)
tents through decoder
