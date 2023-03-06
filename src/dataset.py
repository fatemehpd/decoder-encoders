
import os
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import nibabel as nib
import numpy as np
import torchvision
import torchvision.transforms as TF

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
        self.maskes = os.listdir(mask_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.maskes[index])
        
        image = np.load(img_path)
        mask = np.load(mask_path)

        image = transforms.Compose([transforms.ToTensor()])(image)
        mask = transforms.Compose([transforms.ToTensor()])(mask)
        image = torch.unsqueeze(image, dim=1).float() 
        mask = torch.unsqueeze(mask, dim=1).float() 

        Resize = TF.Resize(size = (128,128))

        image = Resize(image)
        mask = Resize(mask)

        return image, mask
    
    
if __name__=="__main__":
    my_tensor = torch.randn((34,1,512, 512))
    print("first" + str(my_tensor.size()))
    Resize = TF.Resize(size = (128,128))
    print("second"+str(T(my_tensor).size()))



    

