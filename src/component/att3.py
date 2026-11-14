import torch
from torch import nn
from torch.nn.parameter import Parameter

class att(nn.Module):
    def __init__(self, channel, k_size=3):
        super(att, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-2).transpose(-1, -2)).unsqueeze(-1).transpose(-2,-3)
        #y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.avg_pool(y)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        y = x * y.expand_as(x)

        return x * y.expand_as(x)