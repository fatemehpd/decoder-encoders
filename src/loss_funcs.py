# TODO: add documantations and comments

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
class dice_func(nn.Module):
    def __init__(self):
        super(dice_func, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth) / \
            ((inputs+ targets).sum() + smooth)
        return 1 - dice

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        # comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        dice = 0
        inputs = (inputs > 0.5)
        for i in range(targets.shape[1]):
            dice += dice_func()(inputs[:,i:i+1,:,:],targets[:,i:i+1,:,:],smooth=smooth)
        

        return dice


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth)/(union + smooth)

        return 1 - IoU


class weighted_CrossEntropyLoss(nn.Module):
    #for 2 classes
    def __init__(self,weight):
        super(weighted_CrossEntropyLoss, self).__init__()
        self.weight=weight
    def forward(self, input,target):
        input = input[:,0,:,:]
        target = target[:,0,:,:]
        x =-(self.weight[0]*target*torch.log(input+1e-10)+self.weight[1]*(1-target)*torch.log(1-input+1e-10))
        x=x.sum()
        x = x / reduce(lambda x,y: x*y, input.shape)
        return x


class Combined_Loss(nn.Module):
    def __init__(self, weight=None, size_average=True,CE_weight = None):
        super(Combined_Loss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.BCE_loss_func = nn.BCEWithLogitsLoss()
        self.CE_loss_func = nn.CrossEntropyLoss(weight = CE_weight)
        self.dice_loss_func = DiceLoss()
        self.IoU_loss_func =  IoULoss()
        self.WCE_loss_func = weighted_CrossEntropyLoss(weight = CE_weight)

    def forward(self, inputs, targets):

        CE_loss = self.CE_loss_func(inputs, targets)
        WCE_loss = self.WCE_loss_func(nn.Softmax2d()(inputs), targets)
        Dice_Loss = self.dice_loss_func(nn.Softmax2d()(inputs), targets)

        loss = [10*Dice_Loss , 4*CE_loss]

        return sum(loss), loss 



if __name__ == "__main__":
    img1 = torch.zeros(1,2,5,5)
    img1[0,0,0:2,0:2] = 1
    img1[0,1,3:5,3:5] =1
    img2 = torch.zeros(1,2,5,5)
    img2[0,0,0,0]=1
    img3 = torch.zeros(1,2,5,5)
    img3[0,0,0:2,0:2] = 1

    loss0 = Combined_Loss()(img1,img2)
    loss1 = Combined_Loss()(img1,img3)
    loss2 = Combined_Loss(CE_weight = torch.tensor([1,0]))(img1, img3)
    loss3 = Combined_Loss(CE_weight = torch.tensor([0,1]))(img1, img3)
    loss4 = Combined_Loss(CE_weight = torch.tensor([0.01,1]))(img1, img3)


    loss4 = Combined_Loss(CE_weight = torch.tensor([50,50]))(img1, img2)




