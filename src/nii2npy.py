"""a program to convert a nifti file to a numpy form

Atributes:
------------
    path: str
        the directory where the nifti files are located
    images_path: list of str
        the path of nii images as a list 

Methods: 
------------
    convert(w_level=40, w_width=120, save_to="..\converted dataset"):
        convert and save nifti files to a numpy file
"""

import os
import nibabel as nib
import glob
import numpy as np


class nii2npy():
    def __init__(self, path):
        """get the path of the nifti file
        Args:
            path (str): the directory of the nifti file
        """
        self.dirPath = os.path.dirname(__file__)
        self.path = os.path.join(self.dirPath, path)  # obtain folder directory
        self.images_path = glob.glob(os.path.join(self.path, "*.nii"))

    def _window(self, ct_scan, w_level, w_width):
        """extract bone information or subdural information
        Args:
            ct_scan (npy): 3-D array of CT-scan image
            w_level (int, optional): find it in dataset instruction.
            w_width (int, optional): find it in dataset instruction.

        Returns:
            numpy array: extracted information from original CT-scan image
        """
        w_min = w_level - w_width / 2
        w_max = w_level + w_width / 2
        num_slices = ct_scan.shape[2]
        for s in range(num_slices):
            slice_s = ct_scan[:, :, s]
            slice_s = (slice_s - w_min)*(255/(w_max-w_min))
            slice_s[slice_s < 0] = 0
            slice_s[slice_s > 255] = 255
            ct_scan[:, :, s] = slice_s
        return ct_scan

    def _get_name(self, path: str, extentinon: str = ".nii") -> str:
        """Returns a name to save numpy array"""
        splited_path = path.split("\\")
        return splited_path[-1].replace(extentinon, "")

    def convert(self, w_level=40, w_width=120,
                dir_name="converted_dataset", isMask = False):
        """convert and save nifti as numpy array

        Args:
            w_level (int, optional): check dataset instruction.
            Defaults to 40.
            w_width (int, optional):check dataset instruction.
            Defaults to 120.
            save_to (str, optional):path to save your files.
            Defaults to "..\converted dataset".
        """
        save_to = self.dirPath
        while os.listdir(save_to).count("__init__.py"):
            save_to, _  = os.path.split(save_to)
        paths = dir_name.split("\\")
        for path in paths:
            if not os.path.exists(os.path.join(save_to, path)):
                save_to = os.path.join(save_to, path)
                os.mkdir(save_to)
            else:
                save_to = os.path.join(save_to, path)
        for path in self.images_path:
            ct = nib.load(path)
            ct = ct.get_fdata()
            if not isMask:
                ct = self._window(ct, w_level, w_width)
            name = self._get_name(path)
            np.save(os.path.join(save_to, name), ct)
            print(str(name) +'   '+str(ct.max()) +"   " +str(ct.min()))


if __name__ == "__main__":
    #farayand path
    mask_paths = ["..\\dataSet\\train_masks", "..\\dataSet\\val_masks"]
    train_paths = ["..\\dataSet\\train_cts", "..\\dataSet\\val_cts"]         
    
    #mohammad path
    #paths = ["..\dataSet\ct_scans", "..\dataSet\masks"]
    
    #test path
    #paths = ["..\\dataSet\\masks"]

    for path in train_paths:
        print("path to canvert is: " + path)
        data = nii2npy(path)
        save_path = os.path.join("converted_dataset", path.split("\\")[-1])
        data.convert(dir_name=save_path, isMask=False)

    for path in mask_paths:
        print("path to canvert is: " + path)
        data = nii2npy(path)
        save_path = os.path.join("converted_dataset", path.split("\\")[-1])
        data.convert(dir_name=save_path, isMask=True)
