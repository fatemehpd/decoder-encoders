
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


def window_ct (ct_scan, w_level=40, w_width=120):
    w_min = w_level - w_width / 2
    w_max = w_level + w_width / 2
    num_slices=ct_scan.shape[2]
    for s in range(num_slices):
        slice_s = ct_scan[:,:,s]
        slice_s = (slice_s - w_min)*(255/(w_max-w_min)) #or slice_s = (slice_s - (w_level-(w_width/2)))*(255/(w_width))
        slice_s[slice_s < 0]=0
        slice_s[slice_s > 255] = 255
        #slice_s=np.rot90(slice_s)
        ct_scan[:,:,s] = slice_s      

    return ct_scan

class CTDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])

        image = nib.load(img_path)
        image = image.get_fdata()
        
        window_specs=[40,120] #Brain window
        image = window_ct(image, window_specs[0], window_specs[1])
     

        mask = nib.load(mask_path)
        mask = mask.get_fdata()
       

        image = transforms.Compose([transforms.ToTensor()])(image)
        mask = transforms.Compose([transforms.ToTensor()])(mask)
        image = torch.unsqueeze(image, dim=1).float() 
        mask = torch.unsqueeze(mask, dim=1).float() 


        return image, mask
        


    

