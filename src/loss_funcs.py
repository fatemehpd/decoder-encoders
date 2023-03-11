# TODO: add documantations and comments

import torch
import torch.nn as nn
import unittest
from torch.autograd import Variable

class DiceLoss(nn.Module):
    """
    Dice loss for binary segmentation problems. Calculates the Dice coefficient
    between the predicted and target tensors and returns 1 - dice.

    Args:
        weight (Tensor, optional): A manual rescaling weight given to each class.
        size_average (bool, optional): If True, loss is averaged over the batch size
            and spatial dimensions. If False, loss is summed over the batch size
            and spatial dimensions.

    Inputs:
        - inputs (torch.Tensor): A tensor of predicted labels with shape (batch_size, 1, height, width).
        - targets (torch.Tensor): A tensor of ground truth labels with shape (batch_size, 1, height, width).

    Returns:
        - loss (torch.Tensor): The Dice loss between the predicted and target tensors.
        
    """

    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        """
        Calculate the Dice loss between the predicted and target tensors.

        Args:
            - inputs (torch.Tensor): A tensor of predicted labels with shape (batch_size, 1, height, width).
            - targets (torch.Tensor): A tensor of ground truth labels with shape (batch_size, 1, height, width).
            - smooth (float, optional): A smoothing value added to the numerator and denominator to avoid
              division by zero.

        Returns:
            - loss (torch.Tensor): The Dice loss between the predicted and target tensors.
        """

        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth) / \
            (inputs.sum() + targets.sum() + smooth)

        return 1 - dice

class IoULoss(nn.Module):
    """
    Intersection over Union (IoU) Loss.

    Args:
        weight (torch.Tensor): Optional weight tensor to apply to the loss.
        size_average (bool): If True, the losses are averaged over each loss element in the batch.

    Returns:
            torch.Tensor: Computed IoU loss.
    """

    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        """
        Computes the Intersection over Union (IoU) Loss.

        Args:
            inputs (torch.Tensor): Input tensor (predicted output) of size (N x C x H x W).
            targets (torch.Tensor): Target tensor (ground truth) of size (N x C x H x W).
            smooth (float): A smoothing value to avoid division by zero errors.

        Returns:
            torch.Tensor: Computed IoU loss.
        """

        # Apply sigmoid activation to predicted output (comment out if not needed)
        inputs = torch.sigmoid(inputs)

        # Flatten both tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Calculate intersection, union and IoU
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection
        IoU = (intersection + smooth)/(union + smooth)

        # Return the loss (1 - IoU)
        return 1 - IoU

class Combined_Loss(nn.Module):
    """
    This class defines a custom loss function that combines three loss functions:
    1. Cross Entropy Loss (SCE_loss)
    2. Dice Loss (Dice_Loss)
    3. Intersection over Union (IoU) Loss (IoU_Loss)

    The final loss used is the Dice Loss.

    Args:
        weight (tensor, optional): A manual rescaling weight given to each class.
                                   Default: None
        size_average (bool, optional): By default, the losses are averaged over observations
                                       for each minibatch. However, if the field size_average is
                                       set to False, the losses are instead summed for each minibatch.
                                       Default: True
    """

    def __init__(self, weight=None, size_average=True):
        super(Combined_Loss, self).__init__()

    def forward(self, inputs, targets):
        """
        Computes the combined loss using three different loss functions and returns the
        Dice Loss as the final loss.

        Args:
            inputs (tensor): The input tensor for the loss function.
            targets (tensor): The target tensor for the loss function.

        Returns:
            loss (tensor): The Combined Loss.
        """

        SCE_loss = nn.CrossEntropyLoss()(inputs, targets)

        Dice_Loss = DiceLoss()(inputs, targets)

        IoU_Loss = IoULoss()(inputs, targets)

        loss = Dice_Loss

        return loss

class TestLosses(unittest.TestCase):

    def test_dice_loss(self):
        dice_loss = DiceLoss()
        inputs = Variable(torch.randn((3, 1, 512, 512)), requires_grad=True)
        targets = Variable(torch.randn((3, 1, 512, 512)), requires_grad=True)
        loss = dice_loss(inputs.float(), targets.float())
        assert loss.item() >= 0.0
        loss.backward()
        assert inputs.grad is not None

    def test_iou_loss(self):
        iou_loss = IoULoss()
        inputs = Variable(torch.randn((3, 1, 512, 512)), requires_grad=True)
        targets = Variable(torch.randn((3, 1, 512, 512)), requires_grad=True)
        loss = iou_loss(inputs.float(), targets.float())
        assert loss.item() >= 0.0
        loss.backward()
        assert inputs.grad is not None

    def test_combined_loss(self):
        combined_loss = Combined_Loss()
        inputs = Variable(torch.randn((3, 1, 512, 512)), requires_grad=True)
        targets = Variable(torch.randn((3, 1, 512, 512)), requires_grad=True)
        loss = combined_loss(inputs.float(), targets.float())
        assert loss.item() >= 0.0
        loss.backward()
        assert inputs.grad is not None


if __name__ == '__main__':
    unittest.main()
