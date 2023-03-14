'''models that use in this segmentation and classification'''
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

"""two serial cnn

Returns:
    tensor: feature matrix
"""


class Double2DConv(nn.Module):
    """set input and output channels, kernel and stride sizes 

    Args:
        nn (_type_): _description_
    """

    def __init__(self, in_channels, out_channels,
                 kernel_size=[3, 3], stride_size=1, padding=1):
        """generate 2 serial convolution layers

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            kernel_size (list, optional): Defaults to [3, 3].
            stride_size (int, optional): Defaults to 1.
            padding (int, optional): Defaults to 1.
        """
        super(Double2DConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride_size, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size,
                      stride_size, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        '''copmute the output of 2 cnn layers'''
        return self.conv(x)


class UNET2D(nn.Module):
    """basic Unet network implementation based on the below paper
    https://doi.org/10.48550/arXiv.1505.04597
    """

    def __init__(self, in_channels=3, out_channels=3,
                 features=[64, 128, 256, 512]):
        """setup 2-D unet network

        Args:
            in_channels (int, optional): Defaults to 3.
            out_channels (int, optional):  Defaults to 3.
            features (list, optional): Defaults to [64, 128, 256, 512].
        """
        super(UNET2D, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.softmax = nn.Softmax2d()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        # Down part of UNET
        for feature in features:
            self.downs.append(Double2DConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(Double2DConv(feature*2, feature))  # two cnn on top

        self.bottleneck = Double2DConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        """generate a segmentation image 

        Args:
            x (tensor): the image that would be segmented

        Returns:
            segmented image 
        """
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        x = self.final_conv(x)
        
        x = self.softmax(x)
        print(x.shape)
        # x = self.sigmoid(x)

        return x


def test():
    x = torch.randn((3, 1, 512, 512))
    x = x.to(device=DEVICE)
    
    model = UNET2D(in_channels=1, out_channels=2).to(device=DEVICE)
    preds = model(x)
    print("torch.cuda.max_memory_reserved: %fGB" %
          (torch.cuda.max_memory_reserved(0)/1024/1024/1024))
    


if __name__ == "__main__":
    test()
