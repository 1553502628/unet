import torch
from typing import Optional
from torch import nn
from torch.nn.parameter import Parameter

class ConvModule(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            groups: int = 1):
        super().__init__()
        layers = []
        # Convolution Layer
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class att(nn.Module):
    def __init__(self, channel, k_size=3, h_kernel_size: int = 11, v_kernel_size: int = 11):
        super(att, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.h_conv = ConvModule(channel, channel, (1, h_kernel_size), 1,
                                 (0, h_kernel_size // 2), groups=channel)
        self.v_conv = ConvModule(channel, channel, (v_kernel_size, 1), 1,
                                 (v_kernel_size // 2, 0), groups=channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.h_conv(x)
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-2).transpose(-1, -2)).unsqueeze(-1).transpose(-2,-3)
        #y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.avg_pool(y)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        y = x * y.expand_as(x)
        y = self.v_conv(y)

        return y