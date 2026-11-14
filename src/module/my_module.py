import torch
from torch import nn

from src.component.att import att
from src.component.kanconv import KANConvLayer
from src.component.pc import Partial_conv3


class my_module1(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.att = att(in_channels)
        self.kan = KANConvLayer(in_channels,out_channels)
        self.pc = Partial_conv3(out_channels,4,'split_cat')

    def forward(self,x):
        x1 = self.att(x)
        x2 = self.kan(x1)
        x3 = self.pc(x2)
        return x3