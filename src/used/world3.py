from torch import nn
from src.component.att import att
from src.component.kanconv import KANConvLayer
from src.component.pc import Partial_conv3



class world3(nn.Module):

    def __init__(self, in_channels, out_channels, size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.size = size
        self.relu = nn.ReLU()
        self.att = att(in_channels)
        self.kan = KANConvLayer(in_channels, out_channels)
        self.pc = Partial_conv3(out_channels, 4)
        self.up = nn.Upsample(size=(self.size,self.size), mode='bilinear')

    def forward(self, x):
        x1 = self.att(x)
        x2 = self.kan(x1)
        x3 = self.pc(x2)
        x4 = x2+x3
        x5 = self.relu(x4)
        x6 = self.up(x5)
        return x6