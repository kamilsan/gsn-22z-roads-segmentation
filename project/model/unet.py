from typing import List, Tuple
import torch
import torch.nn.functional as F
from torch import nn


class UNetConvBlock(nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.conv1 = nn.Conv2d(in_features, out_features,
                               kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(out_features, out_features,
                               kernel_size=3, padding='same')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)

        return x


class UNet(nn.Module):

    def __init__(self, image_size: Tuple[int, int],
                 in_channels: int,
                 out_channels: int, channels: List[int] = [64, 128, 256, 512, 1024],
                 **kwargs) -> None:
        super().__init__()

        # Since input image is downscaled by a factor of 2
        # at every pooling, input image sizes should be divisible
        # by 2^number of pooling operations
        total_downscale_factor = 2 ** len(channels)
        if image_size[0] % total_downscale_factor != 0 or \
           image_size[1] % total_downscale_factor != 0:
            raise Exception(
                f'Input image size should be divisible by {total_downscale_factor}')

        self.num_classes = out_channels
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        full_channels_list = [in_channels] + channels
        self.down_blocks = nn.ModuleList([UNetConvBlock(
            full_channels_list[idx],
            full_channels_list[idx + 1]) for idx in range(len(full_channels_list) - 1)])

        full_channels_list.reverse()
        self.up_blocks = nn.ModuleList([UNetConvBlock(
            full_channels_list[idx],
            full_channels_list[idx + 1]) for idx in range(len(full_channels_list) - 1)])
        self.up_convs = nn.ModuleList([nn.ConvTranspose2d(full_channels_list[idx], full_channels_list[idx + 1],
                                      kernel_size=2, stride=2) for idx in range(len(full_channels_list) - 1)])

        self.final_conv = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features_list: List[torch.Tensor] = []

        # Encoder
        for down_block in self.down_blocks[:-1]:
            features = down_block(x)
            features_list.append(features)
            x = self.pool(features)

        # Bottleneck
        x = self.down_blocks[-1](x)

        features_list.reverse()

        # Decoder
        for features, up_block, up_conv in zip(features_list, self.up_blocks, self.up_convs):
            x = up_conv(x)
            x = torch.cat([x, features], dim=1)
            x = up_block(x)

        x = self.final_conv(x)

        return x
