# TODO: add documantations and comments
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

    model.eval()
    for idx, (x, y) in enumerate(loader):
        #y = y.to(device=device)
        #x = x.to(device=device)

        x = x.to(device=device)
        with torch.no_grad():
            preds = model(x)
            preds = nn.Sigmoid()(preds)
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, f"{folder}/{idx}_pred.png")
        torchvision.utils.save_image(y, f"{folder}/{idx}_save.png")

    model.train()
