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
        resize: bool = False,
        gray2RGB: bool = True,
        classification:bool = False,
        resize_size: tuple[int, int] = (128, 128)
    ) -> None:
        #NOTE: this class is for loading slices not whole ct image
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
        self.gray2RGB = gray2RGB
        self.classification = classification
        self.resize_size = resize_size
        self.preprocess = TF.Compose([transforms.ToTensor()])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if resize:
            self.resize_func = TF.Resize(size=resize_size)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> torch.Tensor:
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.maskes[index])
        
        # NOTE: data will cast to uint8
        image = np.round(np.load(img_path)).astype(np.uint8)
        mask = np.round(np.load(mask_path)).astype(np.uint8)

        image = self.preprocess(image)
        mask = self.preprocess(mask)

        if self.resize:
            image = self.resize_func(image)
            mask = self.resize_func(mask)

        if len(image.shape)==2:
                image = torch.unsqueeze(image, dim=0)
                
        if not self.classification:
            if len(mask.shape)==2:
                mask = torch.unsqueeze(mask, dim=0)  
            # NOTE: change below code proportionally to your dataset and your goal
            mask = torch.cat((mask, 1 - mask), dim=0)
        else:
            if mask.max() == 1:
                mask= torch.tensor(1)
            else:
                mask=torch.tensor(0)
        if self.gray2RGB: 
            # NOTE: image is a slice of brain
            if len(image.shape)==2:
                image = torch.unsqueeze(image, dim=1)
            image = torch.cat((image, image, image), dim=0)
        
        return image.to(self.device), mask.to(self.device)
