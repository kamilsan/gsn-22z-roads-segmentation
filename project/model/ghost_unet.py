from typing import List, Tuple
import torch
import torch.nn.functional as F
from torch import nn
import math
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


class SqueezeExcitation(nn.Module):

    def __init__(self, input_channels, reduction_ratio: float = 16) -> None:
        super().__init__()

        self.channels = input_channels
        excitation_bottleneck_channels = int(self.channels / reduction_ratio)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.shrink = nn.Linear(self.channels, excitation_bottleneck_channels)
        self.blow = nn.Linear(excitation_bottleneck_channels, self.channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Squeeze
        z = self.avg_pool(x).view((-1, self.channels))

        # Excitation
        s = F.relu(self.shrink(z))
        s = torch.sigmoid(self.blow(s)).view((-1, self.channels, 1, 1))

        # Scale
        y = s * x

        return y


class GhostModule(nn.Module):
    # Mostly stolen from https://github.com/huawei-noah/Efficient-AI-Backbones/blob/master/ghostnet_pytorch/ghostnet.py
    # (original authors' implementation)
    # as paper doesn't quite explain what those cheap operations are.
    # Adapted for Ghost-UNet
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int = 1, ratio: float = 2, dw_size: int = 3, stride: int = 1, relu: bool = True) -> None:
        super().__init__()

        reduced_channels = math.ceil(output_channels / ratio)
        # nearly equal to `output_channels` but multiple
        # of `reduced_channels` for sure, which is required
        # for depth-wise conv
        cheap_channels = reduced_channels * (ratio - 1)

        # as cheap conv may produce a little bit more channels
        # than it is necessary, we store the amount of required channels
        # so as to be able to discard unneeded ones later
        self.output_channels = output_channels

        self.normal_conv = nn.Sequential(
            nn.Conv2d(input_channels, reduced_channels, kernel_size,
                      stride, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.LeakyReLU(inplace=True) if relu else nn.Identity()
        )
        self.cheap_conv = nn.Sequential(
            nn.Conv2d(reduced_channels, cheap_channels, dw_size, 1,
                      padding=dw_size//2, groups=reduced_channels, bias=False),
            nn.BatchNorm2d(cheap_channels),
            nn.LeakyReLU(inplace=True) if relu else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.normal_conv(x)
        x2 = self.cheap_conv(x1)

        x = torch.cat([x1, x2], dim=1)
        x = x[:, :self.output_channels, :, :]

        return x


class GhostLayer(nn.Module):
    # NOTE: As there is no implementation and paper is not reproducible at all
    # many sizes here were choosen arbitraily
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, use_se: bool = True, use_pooling: bool = True) -> None:
        super().__init__()

        self.use_se = use_se
        self.use_pooling = use_pooling
        self.skip_conv = stride == 1

        self.gm1 = GhostModule(in_channels, out_channels)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        # TODO: Figure 1 suggests that when stride = 1 dw conv is skipped, yet on Figure 2 it is always used...
        if not self.skip_conv:
            self.conv = nn.Conv2d(out_channels, out_channels, kernel_size,
                                  1, padding=kernel_size//2, groups=out_channels, bias=False)
            self.batch_norm2 = nn.BatchNorm2d(out_channels)

        # TODO: Also, sizes do not match. For GL6 input size is 19x19, kernel size 2, stride 1 and output is 20x20
        # similar thing with layer 7 - size also grows by 1 pixel
        # if dw-conv should really be skipped, some padding must happen inside Ghost Module

        if use_se:
            self.se = SqueezeExcitation(out_channels)

        self.gm2 = GhostModule(out_channels, out_channels, relu=False)

        if use_pooling:
            self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gm1(x)
        x = self.batch_norm1(x)
        x = F.leaky_relu(x)

        if not self.skip_conv:
            x = self.conv(x)
            x = self.batch_norm2(x)
            x = F.leaky_relu(x)

        if self.use_se:
            x = self.se(x)

        x = self.gm2(x)

        if self.use_pooling:
            x = self.pool(x)

        return x


class GhostUNetDConv(nn.Module):
    # At least this is easy
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.conv1 = nn.Conv2d(in_features, out_features,
                               kernel_size=3, padding='same')
        self.batch_norm1 = nn.BatchNorm2d(out_features)

        self.conv2 = nn.Conv2d(out_features, out_features,
                               kernel_size=3, padding='same')
        self.batch_norm2 = nn.BatchNorm2d(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)

        return x


class GhostUNet(nn.Module):

    def __init__(self, image_size: Tuple[int, int], out_channels: int, **kwargs) -> None:
        super().__init__()

        self.image_size = image_size
        self.num_classes = out_channels

        self.enc_layer1 = GhostLayer(3, 64, 2, 2, use_se=False)
        self.enc_layer2 = GhostLayer(64, 125, 2, 2, use_se=True)
        self.enc_layer3 = GhostLayer(125, 256, 2, 2, use_se=False)
        self.enc_layer4 = GhostLayer(256, 415, 2, 2, use_se=True)
        self.enc_layer5 = GhostLayer(415, 612, 2, 2, use_se=False)
        self.enc_layer6 = GhostLayer(612, 950, 2, 1, use_se=True)
        self.enc_layer7 = GhostLayer(
            950, 1024, 2, 1, use_se=False, use_pooling=False)

        self.up_conv1 = nn.ConvTranspose2d(
            1024, out_channels, kernel_size=4, stride=2)
        self.double_conv1 = GhostUNetDConv(out_channels, out_channels)

        self.up_conv2 = nn.ConvTranspose2d(
            950 + out_channels, out_channels, kernel_size=4, stride=2)
        self.double_conv2 = GhostUNetDConv(out_channels, out_channels)

        self.up_conv3 = nn.ConvTranspose2d(
            415 + out_channels, out_channels, kernel_size=4, stride=2)
        self.double_conv3 = GhostUNetDConv(out_channels, out_channels)

        self.up_conv4 = nn.ConvTranspose2d(
            256 + out_channels, out_channels, kernel_size=4, stride=2)
        self.double_conv4 = GhostUNetDConv(out_channels, out_channels)

        self.final_conv = nn.ConvTranspose2d(
            125 + out_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc_layer1(x)

        features1 = self.enc_layer2(x)
        features2 = self.enc_layer3(features1)
        features3 = self.enc_layer4(features2)

        x = self.enc_layer5(features3)

        features4 = self.enc_layer6(x)

        x = self.enc_layer7(features4)

        x = self.up_conv1(x)
        x = self.double_conv1(x)
        x = self.concat_with_features(x, features4)

        x = self.up_conv2(x)
        x = self.double_conv2(x)
        x = self.concat_with_features(x, features3)

        x = self.up_conv3(x)
        x = self.double_conv3(x)
        x = self.concat_with_features(x, features2)

        x = self.up_conv4(x)
        x = self.double_conv4(x)
        x = self.concat_with_features(x, features1)

        x = self.final_conv(x)

        return TF.resize(x, self.image_size, interpolation=InterpolationMode.NEAREST)

    def concat_with_features(self, x: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        target_size = x.shape[-2:]
        features_resized = TF.resize(
            features, target_size, interpolation=InterpolationMode.NEAREST)
        result = torch.cat([x, features_resized], dim=1)

        return result
