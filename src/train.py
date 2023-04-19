# TODO: add documantations and comments

import torch
import numpy as np
import os
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from loss_funcs import DiceLoss, IoULoss, Combined_Loss 
from model import *
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    save_predictions_as_imgs,
)

# Hyperparameters etc.
LEARNING_RATE = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 1000
NUM_WORKERS = 2
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_IMG_DIR = "./converted_dataset/train_cts"
TRAIN_MASK_DIR = "./converted_dataset/train_masks"
VAL_IMG_DIR = "./converted_dataset/val_cts"
VAL_MASK_DIR = "./converted_dataset/val_masks"


def train_fn(loader, model, optimizer, loss_fn, scaler, losses):
    loop = tqdm(loader)
    sigmoid = nn.Sigmoid()
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            pred = model(data)
            loss, _ = loss_fn(pred, targets)
            
        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        losses.append(loss.item())


def main():

    model = UNET2D(in_channels=1, out_channels=1).to(DEVICE)
    if LOAD_MODEL:
        epoch_losses = np.load(os.path.join(os.path.dirname(__file__), '..\\losses\\epoch_losses.npy')).tolist()
        val_losses = np.load(os.path.join(os.path.dirname(__file__), '..\\losses\\val_losses.npy')).tolist()
    else:
        epoch_losses = []
        val_losses = []

    losses = []
    
    loss_fn1 = nn.BCEWithLogitsLoss()
    loss_fn2 = IoULoss()
    loss_fn3 = DiceLoss()
    loss_fn4 = nn.MSELoss()
    loss_combined = Combined_Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

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

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        print(f"Number of Epoch: {epoch}")
        # print(losses)
        if losses != []:
            epoch_losses.append(np.average(losses))
            losses = []
            np.save(os.path.join(os.path.dirname(__file__), '..\\losses\\epoch_losses'), epoch_losses)
  

        train_fn(train_loader, model, optimizer, loss_combined, scaler, losses)

        # save model
        if(epoch % 3 == 0):
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint)

            # print some examples to a folder
            save_predictions_as_imgs(
                val_loader, model, loss_combined, val_losses, folder="./saved_images", device=DEVICE 
            )


if __name__ == "__main__":
    main()
