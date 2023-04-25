# TODO: add documantations and comments
import os
import torch
import torch.nn as nn
import numpy as np
import torchvision
from dataset import CTDataset
from torch.utils.data import DataLoader


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """to save parameters of checkpoint"""
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    """to load parameters from last checkpoint"""
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


# TODO: add method to load parameters from specific checkpoint


def get_loaders(
    train_dir:str,
    train_maskdir:str,
    val_dir:str,
    val_maskdir:str,
    batch_size:int,
    num_workers:int=4,
    pin_memory:bool=True,
    shuffle:bool=True
):
    train_ds = CTDataset(image_dir=train_dir, mask_dir=train_maskdir)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
    )

    val_ds = CTDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def save_predictions_as_imgs(
    loader, model, loss_fn, val_losses, folder="./saved_images", device="cuda"
):
    losses = []
    model.eval()
    for idx, (x, y) in enumerate(loader):
        # y = y.to(device=device)
        # x = x.to(device=device)

        x = x.to(device=device)
        with torch.no_grad():
            preds = model(x)
            loss, _ = loss_fn(preds, y.to(device=device))
            preds = nn.Softmax2d()(preds)
            preds = preds[:, 0, :, :].unsqueeze(1)
            preds = (preds > 0.5).float()

        losses.append(loss.item())
        torchvision.utils.save_image(preds, f"{folder}/{idx}_pred.png")
        torchvision.utils.save_image(
            y[:, 0, :, :].unsqueeze(1), f"{folder}/{idx}_save.png"
        )

    val_losses.append(np.average(losses))
    np.save(
        os.path.join(os.path.dirname(__file__), "..\\losses\\val_losses"), val_losses
    )

    model.train()
