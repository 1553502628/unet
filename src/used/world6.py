from torch import nn
from src.component.att import att
from src.component.kanconv import KANConvLayer


class world6(nn.Module):

    def __init__(self, in_channels, out_channels, size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.size = size
        self.att = att(in_channels)
        self.kan = KANConvLayer(in_channels, out_channels)
        self.up = nn.Upsample(size=(self.size, self.size), mode='bilinear')

    def forward(self, x):
        x1 = self.att(x)
        x2 = self.kan(x1)
        x3 = self.up(x2)
        return x3