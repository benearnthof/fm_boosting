# Unconditional Training will look like this: 
from fm_boosting.datasets import ImageDataset
from fm_boosting.models import UNet, LFM
from fm_boosting.wrappers import Trainer

model = Unet(dim = 64)

lfm = LFM(model) # adjust default values to your liking

img_dataset = ImageDataset(
    folder = '/data/oxford_flowers/jpg',
    image_size = 256
)

trainer = Trainer(
    lfm,
    dataset = img_dataset,
    num_train_steps = 100_000,
    results_folder = '/your/results/folder/'   # samples will be saved periodically to this folder
)

trainer()
