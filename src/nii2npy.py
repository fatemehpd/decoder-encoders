"""a program to convert a nifti file to a numpy form

Atributes:
    path: str
        the directory where the nifti files are located
    images_path: list of str
        the path of nii images as a list 

Methods:

    

    
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
      self.path = os.path.join(os.path.dirname(
          __file__), path)  # obtain folder directory
      self.images_path = glob.glob(os.path.join(self.path, "*.nii"))

  def _window(self, ct_scan, w_level, w_width):
      """extract bone information or subdural information
      Args:
          ct_scan (npy): 3-D array of CT-scan image
          w_level (int, optional): find it in dataset instruction.
          w_width (int, optional): find it in dataset instruction.

      Returns:
          numpy array: extractet information from original CT-scan image
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
    
  def _get_name(self, path:str, extentinon:str =".nii" )->str:
    """Returns a name to save numpy array"""
    splited_path = path.split("\\")
    return splited_path[-1].replace(extentinon,"")
    
  def convert(self,w_level=40, w_width=120, save_to="..\converted dataset"):
    """convert and save nifti as numpy array

    Args:
        w_level (int, optional): check dataset instruction.
        Defaults to 40.
        w_width (int, optional):check dataset instruction.
        Defaults to 120.
        save_to (str, optional):path to save your files.
        Defaults to "..\converted dataset".
    """
    
    path = os.path.join(os.path.dirname(__file__),save_to)
    if ~os.path.exists(path):
      os.mkdir(path)
    for path in self.images_path:
      ct = nib.load(path)
      ct = ct.get_fdata()
      ct = self._window(ct,w_level,w_width)
      name = self._get_name(path)
      np.save(os.path.join(os.path.dirname(__file__),save_to,name),ct)
if __name__ == "__main__":
  path = "..\dataSet\ct_scans"
  data = nii2npy(path)
  data.convert()

