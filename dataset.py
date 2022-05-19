import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import random
from torchvision.transforms import transforms, functional


class ImageFolderVimeo(Dataset):
    def __init__(self, root, transform=None, split="train"):
        from tqdm import tqdm
        self.mode = split
        self.transform = transform
        self.samples = []
        split_dir = Path('./data/vimeo_septuplet/sequences')
        for sub_f in tqdm(split_dir.iterdir()):
            if sub_f.is_dir():
                for sub_sub_f in Path(sub_f).iterdir():
                    self.samples += list(sub_sub_f.iterdir())

        if not split_dir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

    def __getitem__(self, index):
        img = Image.open(self.samples[index]).convert("RGB")
        if self.transform:
            return self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)


class Kodak24Dataset(Dataset):
    def __init__(self, root, transform=None, split="kodak24"):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = [f for f in splitdir.iterdir() if f.is_file()]
        self.transform = transform
        self.mode = split

    def __getitem__(self, index):
        img = Image.open(self.samples[index]).convert("RGB")
        if self.transform:
            return self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)