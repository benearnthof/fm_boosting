# dataset classes
from utils import exists
from pathlib import Path
from typing import List

from torch.utils.data import Dataset
import torchvision.transforms as T
from torch import nn

from functools import partial

from PIL import Image

class ImageDataset(Dataset):
    def __init__(
        self,
        folder: str | Path,
        image_size: int,
        exts: List[str] = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        super().__init__()
        if isinstance(folder, str):
            folder = Path(folder)

        assert folder.is_dir()

        self.folder = folder
        self.image_size = image_size

        self.paths = [p for ext in exts for p in folder.glob(f'**/*.{ext}')]

        self.maybe_convert_fn = partial(self.convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(self.maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])
    
    def convert_image_to_fn(self, img_type, image):
        if image.mode == img_type:
            return image
        return image.convert(img_type)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)



class SuperresDataset(ImageDataset):
    def __init__(
        self,
        folder: str | Path,
        image_size: int, # lowres
        target_size: int, # hires
        exts: List[str] = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        super().__init__(folder, image_size)
        self.target_size = target_size

        self.transform_hires = T.Compose([
            T.Lambda(self.maybe_convert_fn),
            T.Resize(target_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(target_size),
            T.ToTensor()
        ])
    
    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        # return in source, target format => lowres, hires
        return self.transform(img), self.transform_hires(img)
        

