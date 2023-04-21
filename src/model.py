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


class UNETEncoder2D(nn.Module):
    """UNETEncoder2D is a series of double CNN 2D and maxpooling
    this module doesn't calculate bottleneck of unet"""

    def __init__(
        self,
        in_channels=3,
        kernel=[3, 3, 3, 3],
        padding=[1, 1, 1, 1],
        stride=[1, 1, 1, 1],
        batchNorm=True,
        features=[64, 128, 256, 512],
    ):
        super(UNETEncoder2D, self).__init__()

        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.skip_connections = []

        for feature in features:
            self.downs.append(
                Double2DConv(in_channels, feature, batchNorm=batchNorm)
            )
            in_channels = feature

    def forward(self, x):
        self.skip_connections = []
        for down in self.downs:
            x = down(x)
            self.skip_connections.append(x)
            x = self.pool(x)
        return x, self.skip_connections


class UNETDecoder2D(nn.Module):
    """UNETDecoder2D is a series of double CNN 2D and transpose
    convolution. this module doesn't calculate final CNN layer of unet"""

    def __init__(
        self,
        out_channels=3,
        kernel=[3, 3, 3, 3],
        padding=[1, 1, 1, 1],
        stride=[1, 1, 1, 1],
        batchNorm=True,
        features=[64, 128, 256, 512],
    ):
        super(UNETDecoder2D, self).__init__()

        self.ups = nn.ModuleList()
        self.out_channels = out_channels
        self.initConv = nn.Conv2d(
            features[-1] * 4, features[-1] * 2, 3, padding=1
        )

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
                Double2DConv(feature * 2, feature, batchNorm=batchNorm)
            )

    def forward(self, x, skip_connections):
        skip_connections = skip_connections[::-1]
        x = self.initConv(x)
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)
        return x


class UNET2D(nn.Module):
    """basic Unet network implementation based on the below paper
    https://doi.org/10.48550/arXiv.1505.04597"""

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        features=[64, 128, 256, 512]
        # TODO: add kernel padding stride batchnorm
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
        #x = self.softmax(x)
        # print(x.shape)
        #x = self.sigmoid(x)

        return x


class Double3DConv(nn.Module):
    """two serial 3D CNN"""

    def __init__(
        self,
        in_channels,
        out_channels,
        batchNorm=True,
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
        """copmute the output of 3 CNN layers"""
        if len(x.shape) == 4:
            x = x.unsqueeze(0)
        return torch.squeeze(self.conv(x), 0)


class UNETEncoder3D(nn.Module):
    """UNETEncoder3D is a series of double CNN 2D and maxpooling
    this module doesn't calculate bottleneck of unet"""

    def __init__(
        self,
        in_channels=3,
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
        super(UNETEncoder3D, self).__init__()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.in_channels = in_channels
        self.batchNorm = batchNorm
        self.padding = padding
        self.skip_connections = []

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

    def forward(self, x):
        self.skip_connections = []

        for down in self.downs:
            x = down(x)
            self.skip_connections.append(x)
            x = self.pool(x)
        return x, self.skip_connections


class UNETDecoder3D(nn.Module):
    """UNETDecoder2D is a series of double CNN 2D and transpose
    convolution. this module doesn't calculate final CNN layer of unet"""

    def __init__(
        self,
        out_channels=3,
        kernel=[3, 3, 3, 3],
        padding=[1, 1, 1, 1],
        stride=[1, 1, 1, 1],
        batchNorm=True,
        features=[64, 128, 256, 512],
    ):
        super(UNETDecoder3D, self).__init__()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.out_channels = out_channels
        self.batchNorm = batchNorm
        self.padding = padding
        self.initConv = nn.Conv3d(
            features[-1] * 4, features[-1] * 2, 3, padding=1
        )

        for i, feature in enumerate(reversed(features)):
            self.ups.append(
                nn.ConvTranspose3d(
                    feature * 2,
                    feature,
                    kernel_size=(1, 2, 2),
                    stride=(1, 2, 2),
                )
            )
            j = len(features) - i - 1
            self.ups.append(
                Double3DConv(
                    feature * 2,
                    feature,
                    batchNorm=self.batchNorm,
                    kernel_size=kernel[j],
                    stride_size=stride[j],
                    padding=self.padding[j],
                )
            )

    def forward(self, x, skip_connections):
        x = self.initConv(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            if len(x.shape) == 4:
                concat_skip = torch.cat((skip_connection, x), dim=0)
            elif len(x.shape) == 5:
                concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)
        return x


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
        for i, feature in enumerate(reversed(features)):
            self.ups.append(
                nn.ConvTranspose3d(
                    feature * 2,
                    feature,
                    kernel_size=(1, 2, 2),
                    stride=(1, 2, 2),
                )
            )
            j = len(features) - i - 1
            self.ups.append(
                Double3DConv(
                    feature * 2,
                    feature,
                    batchNorm=self.batchNorm,
                    kernel_size=kernel[j],
                    stride_size=stride[j],
                    padding=self.padding[j],
                )
            )
        self.bottleneck = Double3DConv(
            features[-1],
            features[-1] * 2,
            batchNorm=self.batchNorm,
            kernel_size=(3, 3, 3),
            # stride_size=1,
            padding=(1, 1, 1),
        )
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

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
            if len(x.shape) == 4:
                concat_skip = torch.cat((skip_connection, x), dim=0)
            elif len(x.shape) == 5:
                concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        x = self.final_conv(x)
        # TODO:add sigmoid
        #
        # return nn.Sigmoid()(x)
        return x


class UPSAMPLE3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels = 4,
        up_size = 128,
        dilations=[1, 2, 4, 8],
        batchNorm=True,
    ):
        """
        Args:
            in_channels (int): The number of input channels.
            out_channels (int, optional): The number of output channels. Defaults to 4.
            up_size (int, optional): The desired size of the upsampled output. Defaults to 128.
            dilations (list of int, optional): A list of dilation values for the convolutional layers. Defaults to [1, 2, 4, 8].
            batchNorm (bool, optional): Whether or not to use batch normalization. Defaults to True.
        """
        super(UPSAMPLE3D, self).__init__()

        # Define a list to store the dilated convolutional layers
        self.ds = nn.ModuleList()

        # Save the number of input channels and whether or not to use batch normalization
        self.in_channels = in_channels
        self.batchNorm = batchNorm

        # Create the first convolutional layer
        self.first_conv = Double3DConv(in_channels, out_channels)

        # Create the dilated convolutional layers
        for dilation in dilations:
            self.ds.append(Double3DConv(in_channels = out_channels, 
            out_channels = out_channels,
            dilation= dilation,
            padding= dilation
            ))

        # Create the upsampling layer
        self.upsample = nn.Upsample(size= up_size, mode='nearest')

        # Create the fully connected layers
        self.FClayers = nn.Sequential(
            nn.Linear(in_features= out_channels*4, out_features= out_channels*32),
            nn.Linear(in_features= out_channels*32, out_features= 1),
        )

    def forward(self, x):

        # Apply the first convolutional layer to the input
        x = self.first_conv(x)

        # Upsample the output
        x = self.upsample(x)

        # Create a list to store the output of each dilated convolutional layer
        ds = []

        # Apply each dilated convolutional layer to the upsampled output
        for d in self.ds:
            ds.append(d(x))

        # Concatenate the output of each dilated convolutional layer with the upsampled output
        for idx, d in enumerate(self.ds):
            if(idx != 0):
                x = torch.cat((ds[idx], x), dim=0)
            else:
                x = ds[idx]

        # Permute the output tensor to prepare it for the fully connected layers
        x = x.permute(1, 2, 3, 0)

        # Apply the fully connected layers to the output tensor
        x = self.FClayers(x)

        return x


class xnet(nn.Module):
    """
    A 2D-3D hybrid UNet architecture for medical image segmentation.

    Args:
        in_channels (int): number of input channels. Default is 3.
        out_channels (int): number of output channels. Default is 3.
        kernel (list[int]): size of convolutional kernels. Default is [3, 3, 3, 3].
        padding (list[int]): size of zero padding added to all sides of the input. Default is [1, 1, 1, 1].
        stride (list[int]): stride of convolutional kernels. Default is [1, 1, 1, 1].
        batchNorm (bool): whether to use batch normalization after convolutional layers. Default is True.
        features (list[int]): number of output features for each convolutional layer. Default is [64, 128, 256, 512].
    """

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
        super(xnet, self).__init__()

        # 3D encoder
        self.UNETEncoder3D = UNETEncoder3D(
            in_channels=in_channels,
            kernel=kernel,
            padding=padding,
            stride=stride,
            batchNorm=batchNorm,
            features=features,
        )
        # 2D encoder
        self.UNETEncoder2D = UNETEncoder2D(
            in_channels=in_channels,
            kernel=kernel,
            padding=padding,
            stride=stride,
            batchNorm=batchNorm,
            features=features,
        )
        # 3D decoder
        self.UNETDecoder3D = UNETDecoder3D(
            out_channels=out_channels,
            kernel=kernel,
            padding=padding,
            stride=stride,
            batchNorm=batchNorm,
            features=features,
        )
        # 2D decoder
        self.UNETDecoder2D = UNETDecoder2D(
            out_channels=out_channels,
            kernel=kernel,
            padding=padding,
            stride=stride,
            batchNorm=batchNorm,
            features=features,
        )

        # 3D and 2D bottlenecks
        self.bottleneck3D = Double3DConv(
            features[-1], features[-1] * 2, batchNorm=batchNorm, kernel_size=3
        )
        self.bottleneck2D = Double2DConv(
            features[-1], features[-1] * 2, batchNorm=batchNorm, kernel_size=3
        )

        # Final 3D convolutional layers

        """we supposed that the features[0] index is equal to 64"""
        """TODO: make this code modular"""
        self.finalConv3D = nn.Sequential(
            nn.Conv3d(
                features[0],
                features[0] // 4,
                3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(features[0] // 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                features[0] // 4,
                features[0] // 16,
                3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(features[0] // 16),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                features[0] // 16,
                features[0] // 64,
                3,
                padding=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
        )
        self.finalConv2D = nn.Sequential(
            nn.Conv2d(
                features[0],
                features[0] // 4,
                3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(features[0] // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                features[0] // 4,
                features[0] // 16,
                3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(features[0] // 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                features[0] // 16,
                features[0] // 64,
                3,
                padding=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
        )
        self.finalBatchNorm = nn.BatchNorm2d(out_channels*2)
        self.finalConv = nn.Conv2d(
            out_channels * 2, out_channels, kernel_size=1
        )

    def forward(self, x: torch.Tensor):
        # permute x to have dimensions in 3d form
        x3D = x.permute(1, 0, 2, 3)

        # pass x through the 2D and 3D UNET encoders to obtain features at different resolutions, as well as skip connections
        x2D, skip_connections2D = self.UNETEncoder2D(x)
        x3D, skip_connections3D = self.UNETEncoder3D(x3D)

        # pass the features through the bottleneck layers
        x2D = self.bottleneck2D(x2D).permute(1, 0, 2, 3)
        x3D = self.bottleneck3D(x3D)

        # concatenate the bottleneck outputs
        concat_bottleNeck = torch.cat((x2D, x3D), dim=0)
        x2D = concat_bottleNeck.permute(1, 0, 2, 3)
        x3D = concat_bottleNeck

        # pass the concatenated bottleneck output through the UNET decoders
        x2D = self.UNETDecoder2D(x2D, skip_connections2D)
        x3D = self.UNETDecoder3D(x3D, skip_connections3D)

        # apply final convolutional layers to each output stream
        x2D = self.finalConv2D(x2D)
        x3D = torch.squeeze(self.finalConv3D(x3D.unsqueeze(0)),0).permute(1, 0, 2, 3)

        # concatenate the final outputs of the 2D and 3D streams
        concat_finalLayer = torch.cat((x2D, x3D), dim=1)
        # x = self.finalBatchNorm(concat_finalLayer)
        x = self.finalConv(concat_finalLayer)
        return x


class XNET_UPSAMPLE(nn.Module):
    """
    XNET_UPSAMPLE is a 3D-2D U-Net style architecture that uses a 3D encoder 
    and a 2D encoder to extract spatio-temporal and 2D spatial features 
    respectively, and then concatenates the feature maps and performs a 2D 
    upsampling operation to recover the original input resolution. 
    This architecture is designed for medical image segmentation tasks.
    
    Args:
    - in_channels (int): Number of input channels (default: 3)
    - out_channels (int): Number of output channels (default: 3)
    - kernel (list of ints): List of kernel sizes for the convolutions in the
      encoder and decoder (default: [3, 3, 3, 3])
    - padding (list of ints): List of padding sizes for the convolutions in the
      encoder and decoder (default: [1, 1, 1, 1])
    - stride (list of ints): List of stride sizes for the convolutions in the
      encoder and decoder (default: [1, 1, 1, 1])
    - batchNorm (bool): Whether or not to use batch normalization (default: True)
    - features (list of ints): List of feature map sizes for the convolutions in
      the encoder and decoder (default: [64, 128, 256, 512])
    - upsample_out_channels (int): Number of output channels for the 3D-to-2D
      upsampling operation (default: 4)
    """
    
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        kernel=[3, 3, 3, 3],
        padding=[1, 1, 1, 1],
        stride=[1, 1, 1, 1],
        batchNorm=True,
        features=[64, 128, 256, 512],
        upsample_out_channels=4
    ):
        super(XNET_UPSAMPLE, self).__init__()

        # 3D encoder
        self.UNETEncoder3D = UNETEncoder3D(
            in_channels=in_channels,
            kernel=kernel,
            padding=padding,
            stride=stride,
            batchNorm=batchNorm,
            features=features,
        )

        # 2D encoder
        self.UNETEncoder2D = UNETEncoder2D(
            in_channels=in_channels,
            kernel=kernel,
            padding=padding,
            stride=stride,
            batchNorm=batchNorm,
            features=features,
        )

        # 3D-to-2D upsampling layer
        self.UPSAMPLE3D = UPSAMPLE3D(
            in_channels=features[-1] * 4,
            out_channels=upsample_out_channels,
        )

        # 2D decoder
        self.UNETDecoder2D = UNETDecoder2D(
            out_channels=out_channels,
            kernel=kernel,
            padding=padding,
            stride=stride,
            batchNorm=batchNorm,
            features=features,
        )

        self.bottleneck3D = Double3DConv(
            features[-1], features[-1] * 2, batchNorm=batchNorm, kernel_size=3
        )
        self.bottleneck2D = Double2DConv(
            features[-1], features[-1] * 2, batchNorm=batchNorm, kernel_size=3
        )

        """we supposed that the features[0] index is equal to 64"""
        """TODO: make this code modular"""
        self.finalConv2D = nn.Sequential(
            nn.Conv2d(
                features[0],
                features[0] // 4,
                3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(features[0] // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                features[0] // 4,
                features[0] // 16,
                3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(features[0] // 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                features[0] // 16,
                features[0] // 64,
                3,
                padding=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
        )
        self.finalBatchNorm = nn.BatchNorm2d(out_channels*2)
        self.finalConv = nn.Conv2d(
            out_channels * 2, out_channels, kernel_size=1
        )

    def forward(self, x: torch.Tensor):
        # Permute x to change dimensions of channel and batch 
        x3D = x.permute(1, 0, 2, 3)

        # Apply the 2D and 3D encoders
        x2D, skip_connections2D = self.UNETEncoder2D(x)
        x3D, _ = self.UNETEncoder3D(x3D)

        # Apply the 2D and 3D bottlenecks
        x2D = self.bottleneck2D(x2D).permute(1, 0, 2, 3)
        x3D = self.bottleneck3D(x3D)

        # Concatenate the bottleneck outputs
        concat_bottleNeck = torch.cat((x2D, x3D), dim=0)
        x2D = concat_bottleNeck.permute(1, 0, 2, 3)
        x3D = concat_bottleNeck
        
        # Apply the 2D decoder and 3D upsampling
        x2D = self.UNETDecoder2D(x2D, skip_connections2D)
        x3D = self.UPSAMPLE3D(x3D).permute(0, 3, 1, 2)

        # Apply the final 2D convolutional layer
        x2D = self.finalConv2D(x2D)

        # Concatenate the 2D and 3D outputs
        concat_finalLayer = torch.cat((x2D, x3D), dim=1)
        # x = self.finalBatchNorm(concat_finalLayer)
        x = self.finalConv(concat_finalLayer)
        return x


def test():
    # this a test for monitoring amount of vram that occupied for training the model
    print(
        "torch.cuda.max_memory_reserved: %fGB"
        % (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024)
    )


if __name__ == "__main__":
    # img1 = torch.randint(0, 10, (1, 3, 10))
    # print(img1.shape)

    # img2 = torch.randint(0, 10, (1, 3, 10))
    # print(img2.shape)

    # img3 = torch.cat((img1, img2), dim=0)
    # print(img3.shape)

    img4 = torch.randn(50, 1, 128, 128)
    print(img4.shape)

    # here you can check all of the main models that written in this files and see the output result 
    model = XNET_UPSAMPLE(in_channels=1, out_channels=1)
    preds = model(img4)
    print(preds.shape)
    # test()
