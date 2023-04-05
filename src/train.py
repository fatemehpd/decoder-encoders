# TODO: add documantations and comments

import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from loss_funcs import DiceLoss, IoULoss, Combined_Loss 
from model import UNET2D
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders, 
    save_predictions_as_imgs,
)

# Hyperparameters etc.
LEARNING_RATE = 1e-4

# use GPU if available, otherwise use CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 

NUM_EPOCHS = 100
NUM_WORKERS = 2
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128

 # flag to enable memory pinning for faster data transfer to GPU
PIN_MEMORY = True

# flag to load a saved model checkpoint
LOAD_MODEL = True

TRAIN_IMG_DIR = "./converted_dataset/train_cts"
TRAIN_MASK_DIR = "./converted_dataset/train_masks"
VAL_IMG_DIR = "./converted_dataset/val_cts"
VAL_MASK_DIR = "./converted_dataset/val_masks"


def train_fn(loader, model, optimizer, loss_fn, scaler):
    """
    Function to train the model for one epoch using a given data loader.

    Args:
        loader (torch.utils.data.DataLoader): data loader for training data

        model (nn.Module): the model to be trained

        optimizer (torch.optim.Optimizer): optimizer to update model 
        weights

        loss_fn (callable): loss function to compute the loss

        scaler (torch.cuda.amp.GradScaler): gradient scaler for mixed 
        precision training
    """
    
    loop = tqdm(loader)
    
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)

        targets = targets.float().to(device=DEVICE)
        # forward
        with torch.cuda.amp.autocast():  # enable mixed precision training
            pred = model(data)
            loss = loss_fn(pred, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop with current loss value
        loop.set_postfix(loss=loss.item())


def main():
    """
    Main function to load data, define model and optimizer, and train the model.
    """

    model = UNET2D(in_channels=1, out_channels=1).to(DEVICE)

    # define the loss functions
    loss_fn1 = nn.BCEWithLogitsLoss()  # binary cross-entropy loss with logits
    loss_fn2 = IoULoss()  # custom Intersection over Union (IoU) loss
    loss_fn3 = DiceLoss()  # custom Dice loss
    loss_fn4 = nn.MSELoss()  # mean squared error loss
    loss_combined = Combined_Loss()  # combined loss function

    # initialize the optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


    # observe input and target data after gathering hyperparameters
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    # This class is used to enable mixed precision training on CUDA-enabled GPUs
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        print(f"Number of Epoch: {epoch}")

        train_fn(train_loader, model, optimizer, loss_fn1, scaler)

        # save model
        if(epoch % 3 == 0):
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint)

            # print some examples to a folder
            save_predictions_as_imgs(
                val_loader, model, folder="./saved_images", device=DEVICE
            )


if __name__ == "__main__":
    main()
