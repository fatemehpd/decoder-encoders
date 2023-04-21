"""models that use in this segmentation and classification"""
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Double2DConv(nn.Module):
    """two serial 2D CNN"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        batchNorm: bool = True,
        kernel_size: list[int, int] = [3, 3],
        stride_size: int = 1,
        padding: int = 1,
        activation: torch.nn.modules.activation = nn.ReLU(inplace=True),
    ) -> None:
        """
        this class create two serial 2D CNN

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            batchNorm (boolean, optuional): have batchnormalization
            layer or not
            kernel_size (list[int,int], optional): Defaults to [3, 3].
            stride_size (int, optional): Defaults to 1.
            padding (int, optional): Defaults to 1.
            activation (torch.nn.modules.activation, optional): specific
            activation function. Defaults to nn.ReLU(inplace=True).
        """
        super(Double2DConv, self).__init__()
        # TODO: remove sequential form of this class
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """copmute the output of 2 cnn layers"""
        return self.conv(x)


class UNETEncoder2D(nn.Module):
    """UNETEncoder2D is a series of double CNN 2D and maxpooling
    this module doesn't calculate bottleneck of unet"""

    def __init__(
        self,
        in_channels: int = 3,
        kernel: list[int] = [3, 3, 3, 3],
        padding: list[int] = [1, 1, 1, 1],
        stride: list[int] = [1, 1, 1, 1],
        batchNorm: bool = True,
        features: list[int] = [64, 128, 256, 512],
    ) -> None:
        """this class use multiple Double CNN 2D and create a simple
        encoder. return of forward method of this class is an image with
        2*features[-1] channels and skip connections that saved after
        each maxpool2D. this class dosent contain bottleneck layer.

        Args:
            in_channels (int, optional): number of input channels. Defaults to 3.
            kernel (list[int], optional): Defaults to [3, 3, 3, 3].
            padding (list[int], optional): Defaults to [1, 1, 1, 1].
            stride (list[int], optional): Defaults to [1, 1, 1, 1].
            batchNorm (bool, optional): _description_. Defaults to True.
            features (list[int], optional): number of features
            after each doubleConv2D. Defaults to [64, 128, 256, 512].
        """
        super(UNETEncoder2D, self).__init__()

        self.downs = nn.ModuleList()  # list of CNNs
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.skip_connections = []

        for feature in features:
            self.downs.append(
                Double2DConv(in_channels, feature, batchNorm=batchNorm)
            )  # TODO: make above initialization dependent on other
            #      argumans of __init__
            in_channels = feature

    def forward(
        self, x: torch.tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        self.skip_connections = []
        for down in self.downs:  # loop(conv -> maxpool)
            x = down(x)
            self.skip_connections.append(x)
            x = self.pool(x)
        return x, self.skip_connections


class UNETDecoder2D(nn.Module):
    """UNETDecoder2D is a series of double CNN 2D and transpose
    convolution. this module doesn't calculate final CNN layer of unet"""

    def __init__(
        self,
        out_channels: int = 3,
        kernel: list[int] = [3, 3, 3, 3],
        padding: list[int] = [1, 1, 1, 1],
        stride: list[int] = [1, 1, 1, 1],
        batchNorm=True,
        features: list[int] = [64, 128, 256, 512],
    ) -> None:
        """
        this class contains the structure of a simple decoder.
        you should consider that forward method requiers
        skip connections that collected from encoder part.

        Args:
            out_channels (int, optional): Defaults to 3.
            kernel (list[int], optional): Defaults to [3, 3, 3, 3].
            padding (list[int], optional): Defaults to [1, 1, 1, 1].
            stride (list[int], optional): Defaults to [1, 1, 1, 1].
            batchNorm (bool, optional): Defaults to True.
            features (list[int], optional):number of features
            in indexes of skip connection. Defaults to [64, 128, 256, 512].
        """
        super(UNETDecoder2D, self).__init__()

        self.ups = nn.ModuleList()
        self.out_channels = out_channels
        self.initConv = nn.Conv2d(
            features[-1] * 4, features[-1] * 2, 3, padding=1
        )  # NOTE: consider that initConv is customized for X_NET

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

    def forward(
        self, x: torch.tensor, skip_connections: list[torch.Tensor]
    ) -> torch.Tensor:
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

    # TODO: reconstruct with decoder and encoder classes see xnet class
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        features: list[int] = [64, 128, 256, 512]
        # TODO: add kernel padding stride batchnorm
    ) -> None:
        """this class contatin structure of 2-D U_NET network

        Args:
            in_channels (int, optional): Defaults to 3.
            out_channels (int, optional):  Defaults to 3.
            features (list[int], optional): Defaults to [64, 128, 256, 512].
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
        in_channels: int,
        out_channels: int,
        batchNorm: bool = True,
        kernel_size: int = 3,
        stride_size: int = 1,
        padding: int = 1,
        dilation: int = 1,
        activation: torch.nn.modules.activation = nn.ReLU(inplace=True),
    ) -> None:
        """this class create two serial 3D CNN

        Args:
            in_channels (int):
            out_channels (int):
            batchNorm (bool, optional): Defaults to True.
            kernel_size (int, optional): Defaults to 3.
            stride_size (int, optional): Defaults to 1.
            padding (int, optional): Defaults to 1.
            dilation (int, optional): Defaults to 1.
            activation (torch.nn.modules.activation, optional):
            Defaults to nn.ReLU(inplace=True).
        """
        super(Double3DConv, self).__init__()
        # TODO: remove sequential form of this class
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
        out_channels=4,
        up_size=128,
        dilations=[1, 2, 4, 8],
        batchNorm=True,
    ):
        super(UPSAMPLE3D, self).__init__()
        self.ds = nn.ModuleList()
        self.in_channels = in_channels
        self.batchNorm = batchNorm

        self.first_conv = Double3DConv(in_channels, out_channels)

        for dilation in dilations:
            self.ds.append(
                Double3DConv(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    dilation=dilation,
                    padding=dilation,
                )
            )

        self.upsample = nn.Upsample(size=up_size, mode="nearest")

        self.FClayers = nn.Sequential(
            nn.Linear(
                in_features=out_channels * 4, out_features=out_channels * 32
            ),
            nn.Linear(in_features=out_channels * 32, out_features=1),
        )

    def forward(self, x):
        x = self.first_conv(x)
        x = self.upsample(x)
        ds = []

        for d in self.ds:
            ds.append(d(x))

        for idx, d in enumerate(self.ds):
            if idx != 0:
                x = torch.cat((ds[idx], x), dim=0)
            else:
                x = ds[idx]

        x = x.permute(1, 2, 3, 0)
        x = self.FClayers(x)

        return x


class xnet(nn.Module):
    """Some Information about xnet"""

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
        self.UNETEncoder3D = UNETEncoder3D(
            in_channels=in_channels,
            kernel=kernel,
            padding=padding,
            stride=stride,
            batchNorm=batchNorm,
            features=features,
        )
        self.UNETEncoder2D = UNETEncoder2D(
            in_channels=in_channels,
            kernel=kernel,
            padding=padding,
            stride=stride,
            batchNorm=batchNorm,
            features=features,
        )
        self.UNETDecoder3D = UNETDecoder3D(
            out_channels=out_channels,
            kernel=kernel,
            padding=padding,
            stride=stride,
            batchNorm=batchNorm,
            features=features,
        )
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
        self.finalBatchNorm = nn.BatchNorm2d(out_channels * 2)
        self.finalConv = nn.Conv2d(
            out_channels * 2, out_channels, kernel_size=1
        )

    def forward(self, x: torch.Tensor):
        x3D = x.permute(1, 0, 2, 3)

        x2D, skip_connections2D = self.UNETEncoder2D(x)
        x3D, skip_connections3D = self.UNETEncoder3D(x3D)

        x2D = self.bottleneck2D(x2D).permute(1, 0, 2, 3)
        x3D = self.bottleneck3D(x3D)

        concat_bottleNeck = torch.cat((x2D, x3D), dim=0)
        x2D = concat_bottleNeck.permute(1, 0, 2, 3)
        x3D = concat_bottleNeck

        x2D = self.UNETDecoder2D(x2D, skip_connections2D)
        x3D = self.UNETDecoder3D(x3D, skip_connections3D)

        x2D = self.finalConv2D(x2D)
        x3D = torch.squeeze(self.finalConv3D(x3D.unsqueeze(0)), 0).permute(
            1, 0, 2, 3
        )

        print(x2D.shape)
        print(x3D.shape)

        concat_finalLayer = torch.cat((x2D, x3D), dim=1)
        # x = self.finalBatchNorm(concat_finalLayer)
        x = self.finalConv(concat_finalLayer)
        return x


class XNET_UPSAMPLE(nn.Module):
    """Some Information about xnet"""

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        kernel=[3, 3, 3, 3],
        padding=[1, 1, 1, 1],
        stride=[1, 1, 1, 1],
        batchNorm=True,
        features=[64, 128, 256, 512],
        upsample_out_channels=4,
    ):
        super(XNET_UPSAMPLE, self).__init__()
        self.UNETEncoder3D = UNETEncoder3D(
            in_channels=in_channels,
            kernel=kernel,
            padding=padding,
            stride=stride,
            batchNorm=batchNorm,
            features=features,
        )
        self.UNETEncoder2D = UNETEncoder2D(
            in_channels=in_channels,
            kernel=kernel,
            padding=padding,
            stride=stride,
            batchNorm=batchNorm,
            features=features,
        )
        self.UPSAMPLE3D = UPSAMPLE3D(
            in_channels=features[-1] * 4,
            out_channels=upsample_out_channels,
        )
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
        self.finalBatchNorm = nn.BatchNorm2d(out_channels * 2)
        self.finalConv = nn.Conv2d(
            out_channels * 2, out_channels, kernel_size=1
        )

    def forward(self, x: torch.Tensor):
        x3D = x.permute(1, 0, 2, 3)

        x2D, skip_connections2D = self.UNETEncoder2D(x)
        x3D, _ = self.UNETEncoder3D(x3D)

        x2D = self.bottleneck2D(x2D).permute(1, 0, 2, 3)
        x3D = self.bottleneck3D(x3D)

        concat_bottleNeck = torch.cat((x2D, x3D), dim=0)
        x2D = concat_bottleNeck.permute(1, 0, 2, 3)
        x3D = concat_bottleNeck

        x2D = self.UNETDecoder2D(x2D, skip_connections2D)
        x3D = self.UPSAMPLE3D(x3D).permute(0, 3, 1, 2)

        x2D = self.finalConv2D(x2D)

        concat_finalLayer = torch.cat((x2D, x3D), dim=1)
        # x = self.finalBatchNorm(concat_finalLayer)
        x = self.finalConv(concat_finalLayer)
        return x


def test():
    # TODO: add comment about specifications of test function and replace
    # test function to test folder
    x = torch.randn(1024, 50, 12, 12)
    # print(x[0].shape)
    model = UPSAMPLE3D(in_channels=1024)
    preds = model(x)
    print(preds.shape)
    # x = x.to(device=DEVICE)

    # model = UNET2D(in_channels=1, out_channels=1).to(device=DEVICE)
    # preds = model(x)
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
    model = XNET_UPSAMPLE(in_channels=1, out_channels=1)
    preds = model(img4)
    print(preds.shape)
    # test()
