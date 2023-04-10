"""models that use in this segmentation and classification"""
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Double2DConv(nn.Module):
    """two serial 2D CNN"""

    def __init__(
        self,
        in_channels,
        out_channels,
        batchNorm=True,
        kernel_size=[3, 3],
        stride_size=1,
        padding=1,
        activation=nn.ReLU(inplace=True),
    ):
        """generate 2 serial convolution layers

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            batchNorm (boolean, optuional): have batchnormalization
            layer or not
            kernel_size (list, optional): Defaults to [3, 3].
            stride_size (int, optional): Defaults to 1.
            padding (int, optional): Defaults to 1.
            activation (torch.nn.modules.activation, optional) specific
            activation function
        """
        super(Double2DConv, self).__init__()
        if batchNorm:
            self.conv = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride_size,
                    padding,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                activation,
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size,
                    stride_size,
                    padding,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                activation,
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride_size,
                    padding,
                    bias=False,
                ),
                activation,
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size,
                    stride_size,
                    padding,
                    bias=False,
                ),
                activation,
            )

    def forward(self, x):
        """copmute the output of 2 cnn layers"""
        return self.conv(x)


class UNET2D(nn.Module):
    """basic Unet network implementation based on the below paper
    https://doi.org/10.48550/arXiv.1505.04597"""

    def __init__(
        self, in_channels=3, out_channels=3, features=[64, 128, 256, 512]
    ):
        """setup 2-D U_NET network

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

        # Down part of UNET
        for feature in features:
            self.downs.append(Double2DConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2,
                    feature,
                    kernel_size=2,
                    stride=2,
                )
            )
            self.ups.append(
                Double2DConv(feature * 2, feature)
            )  # two cnn on top

        self.bottleneck = Double2DConv(features[-1], features[-1] * 2)
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
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        x = self.final_conv(x)
        # TODO: make change in code to mak this class modular
        # check number of classes to segment and toggle softmax
        # check kind of loss function and toggle sigmoid
        # get input to print output shape or not
        # x = self.softmax(x)
        # print(x.shape)
        # x = self.sigmoid(x)

        return x


class Double3DConv(nn.Module):
    """two serial 3D CNN"""

    def __init__(
        self,
        in_channels,
        out_channels,
        batchNorm=False,
        kernel_size=3,
        stride_size=1,
        padding=1,
        dilation=1,
        activation=nn.ReLU(inplace=True),
    ):
        """generate 2 serial convolution layers

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            batchNorm (boolean, optuional): have batchnormalization
            layer or not
            kernel_size (list, optional): Defaults to [3, 3].
            stride_size (int, optional): Defaults to 1.
            padding (int, optional): Defaults to 1.
            activation (torch.nn.modules.activation, optional) specific
            activation function
        """
        super(Double3DConv, self).__init__()
        if batchNorm:
            self.conv = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride_size,
                    padding=padding,
                    dilation=dilation,
                    bias=False,
                ),
                nn.BatchNorm3d(out_channels),
                activation,
                nn.Conv3d(
                    out_channels,
                    out_channels,
                    kernel_size,
                    stride=stride_size,
                    padding=padding,
                    dilation=dilation,
                    bias=False,
                ),
                nn.BatchNorm3d(out_channels),
                activation,
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride_size,
                    padding=padding,
                    dilation=dilation,
                    bias=False,
                ),
                activation,
                nn.Conv3d(
                    out_channels,
                    out_channels,
                    kernel_size,
                    stride=stride_size,
                    padding=padding,
                    dilation=dilation,
                    bias=False,
                ),
                activation,
            )

    def forward(self, x):
        """copmute the output of 2 CNN layers"""
        return self.conv(x)


class encoder3D(nn.Module):
    """a 3D encoder with bottleneck"""

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        kernel=[3, 3, 3, 3],
        padding=[1, 1, 1, 1],
        stride=[1, 1, 1, 1],
        batchNorm=True,
        features=[64, 128, 256, 512],
    ):
        """
        Args:
            in_channels (int, optional): Defaults to 3.
            out_channels (int, optional): Defaults to 3.
            kernel (list, optional): Defaults to [3, 3, 3, 3].
            padding (list, optional): Defaults to [1, 1, 1, 1].
            stride (list, optional): Defaults to [1, 1, 1, 1].
            batchNorm (bool, optional): if you want batch normalization
            layers after 3D CNN Defaults to True.
            features (list, optional): number of feature after each
            maxpool to [64, 128, 256, 512].
        """
        super(encoder3D).__init__()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.in_channels = in_channels
        self.batchNorm = batchNorm
        self.padding = padding
        for i, feature in enumerate(features):
            self.downs.append(
                Double3DConv(
                    self.in_channels,
                    feature,
                    batchNorm=self.batchNorm,
                    kernel_size=kernel[i],
                    stride_size=stride[i],
                    padding=self.padding[i],
                )
            )
            self.in_channels = feature
        self.bottleneck = Double3DConv(features[-1], features[-1] * 2)

    def forward(self, x):

        for down in self.downs:
            x = down(x)
            x = self.pool(x)
        x = self.bottleneck(x)


class UNET3D(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        kernel=[3, 3, 3, 3],
        padding=[1, 1, 1, 1],
        stride=[1, 1, 1, 1],
        batchNorm=True,
        features=[64, 128, 256, 512],
    ):
        """
        Args:
            in_channels (int, optional): Defaults to 3.
            out_channels (int, optional): Defaults to 3.
            kernel (list, optional): Defaults to [3, 3, 3, 3].
            padding (list, optional): Defaults to [1, 1, 1, 1].
            stride (list, optional): Defaults to [1, 1, 1, 1].
            batchNorm (bool, optional): if you want batch normalization
            layers after 3D CNN Defaults to True.
            features (list, optional): number of feature after each
            maxpool to [64, 128, 256, 512].
        """
        super(UNET3D, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.in_channels = in_channels
        self.batchNorm = batchNorm
        self.padding = padding

        # Down part of UNET
        for i, feature in enumerate(features):
            self.downs.append(
                Double3DConv(
                    self.in_channels,
                    feature,
                    batchNorm=self.batchNorm,
                    kernel_size=kernel[i],
                    stride_size=stride[i],
                    padding=self.padding[i],
                )
            )
            self.in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose3d(
                    feature * 2,
                    feature,
                    kernel_size=(1, 2, 2),
                    stride=(1, 2, 2),
                )
            )
            self.ups.append(Double3DConv(feature * 2, feature))

        self.bottleneck = Double3DConv(features[-1], features[-1] * 2)
        self.final_conv = Double3DConv(features[0], out_channels=out_channels)

    def forward(self, x):

        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        x = self.final_conv(x)
class upsample3D(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels = 8,
        up_size = 48,
        dilations=[1, 2, 4, 8],
        batchNorm=True,
        
    ):
 
        super(upsample3D, self).__init__()
        self.ds = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.in_channels = in_channels
        self.batchNorm = batchNorm
        
        for dilation in dilations:
            self.ds.append(Double3DConv(in_channels = in_channels, 
            out_channels = out_channels,
            dilation= dilation,
            padding= dilation
            ))

        self.upsample = nn.Upsample(size= up_size, mode='nearest')

            
    def forward(self, x):
         
        x = self.upsample(x)
        ds = []

        for d in self.ds:
            ds.append(d(x))

        

        return x


def test():
    # TODO: add comment about specifications of test function and replace
    # test function to test folder
    x = torch.randn((8, 50, 12, 12))
    print(x)

    m = nn.Upsample(size=48, mode='nearest')

    model = upsample3D(in_channels= 8).to(device=DEVICE)
    preds = model(x)
    print(preds.shape)
    # x = x.to(device=DEVICE)

    model = UNET2D(in_channels=1, out_channels=1).to(device=DEVICE)
    # preds = model(x)
    print(
        "torch.cuda.max_memory_reserved: %fGB"
        % (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024)
    )


if __name__ == "__main__":
    img1 = torch.randint(0, 10, (1, 3, 10))
    print(img1.shape)

    img2 = torch.randint(0, 10, (1, 3, 10))
    print(img2.shape)

    img3 = torch.cat((img1, img2), dim=0)
    print(img3.shape)

    img4 = torch.randn(1, 1, 50, 128, 128)
    print(img4.shape)
    model = UNET3D(in_channels=1, out_channels=1)
    model(img4)
