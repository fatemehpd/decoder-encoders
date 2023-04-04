import torch
import torch.nn as nn
import torchvision
from dataset import CTDataset
from torch.utils.data import DataLoader


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    '''to save parameters of checkpoint '''
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    ''' to load parameters from last checkpoint '''
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

# TODO: add method to load parameters from specific checkpoint


def get_loaders(train_dir, train_maskdir, val_dir,
                val_maskdir, num_workers=4, pin_memory=True):
    '''Returns train and validation data loaders with specified data directories, 
    number of workers for data loading, and whether to use pinned memory for data.'''
                
    train_ds = CTDataset(image_dir=train_dir, mask_dir=train_maskdir)
    train_loader = DataLoader(train_ds,batch_size=None,num_workers=num_workers,
        pin_memory=pin_memory,shuffle=True)

    val_ds = CTDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def save_predictions_as_imgs(
        loader, model, folder="./saved_images", device="cuda"):
    '''Saves predicted masks and actual masks for images in the specified data 
    loader to the specified folder location using the trained model.'''

    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)  # send data to device
        with torch.no_grad():
            preds = model(x)  # get model predictions
            preds = nn.Sigmoid()(preds)  # apply sigmoid activation to predictions
            preds = (preds > 0.5).float()  # convert predictions to binary (0 or 1)
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")  # save predicted masks
        torchvision.utils.save_image(y, f"{folder}/save_{idx}.png")  # save actual masks

    model.train()
