<<<<<<< Updated upstream
# # -*- coding: utf-8 -*-
# """
# Created on Thu Jul 13 14:13:27 2023

# @author: Mohammad
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from functools import reduce


# class DiceLoss(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(DiceLoss, self).__init__()

#     def forward(self, inputs, targets, smooth= 1):
        

#         # comment out if your model contains a sigmoid or equivalent activation layer
#         #inputs = torch.sigmoid(inputs)
#         dice = torch.tensor(0.,requires_grad=True)
#         # flatten label and prediction tensors
#         # for i in range(inputs.shape[1]):
#         #     dice = dice + self.dice_function(inputs[:,i,:,:], targets[:,i,:,:])
        
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)
#         intersection = (inputs * targets).sum()
#         dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
=======
# # -*- coding: utf-8 -*-
# """
# Created on Thu Jul 13 14:13:27 2023

# @author: Mohammad
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from functools import reduce


# class DiceLoss(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(DiceLoss, self).__init__()

#     def forward(self, inputs, targets, smooth= 1):
        

#         # comment out if your model contains a sigmoid or equivalent activation layer
#         #inputs = torch.sigmoid(inputs)
#         dice = torch.tensor(0.,requires_grad=True)
#         # flatten label and prediction tensors
#         # for i in range(inputs.shape[1]):
#         #     dice = dice + self.dice_function(inputs[:,i,:,:], targets[:,i,:,:])
        
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)
#         intersection = (inputs * targets).sum()
#         dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
>>>>>>> Stashed changes
#         return 1 - dice