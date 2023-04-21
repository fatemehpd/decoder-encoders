"""a module to import data as pytorch into the framework"""
import os
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import nibabel as nib
import numpy as np
import torchvision
import torchvision.transforms as TF


class CTDataset(Dataset):
    """a class to import CT scans into the framework"""

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        resize=True,
        resize_size: tuple[int, int] = (128, 128),
    ) -> None:
        """_summary_

        Args:
            image_dir (str): pass image directory.
            mask_dir (str): pass mask directory.
            resize (bool, optional): if you need resize switch this
            to True. Defaults to True.
            resize_size (tuple[int, int], optional): if resize is Ture
            pass output dimensions. Defaults to (128, 128).
        """ 
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
        self.maskes = os.listdir(mask_dir)
        self.resize = resize
        self.resize_size = resize_size

        if resize:
            self.resize_func = TF.Resize(size=resize_size)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> torch.Tensor:
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.maskes[index])

        image = np.load(img_path)
        mask = np.load(mask_path)

        """convert images from numpy to tensor"""
        image = transforms.Compose([transforms.ToTensor()])(image)
        mask = transforms.Compose([transforms.ToTensor()])(mask)

        """because of 3D structure in pytorch we should make sure that
        input dimensions are equal to 5"""
        image = torch.unsqueeze(image, dim=1).float()
        mask = torch.unsqueeze(mask, dim=1).float()

        if self.resize:
            image = Resize(image) / 255.0  # normalize value between 0 and 1
            mask = torch.round(
                Resize(mask) / 255.0
            )  # make sure indexes are 1 and 0
        else:
            image = image / 255.0  # normalize value between 0 and 1
            mask = torch.round(mask / 255.0)  # make sure indexes are 1 and 0

        # change below code proportionally to your dataset and your goal
        mask = torch.cat((mask, 1 - mask), dim=1)

        return image, mask


# TODO: add proper tets function
