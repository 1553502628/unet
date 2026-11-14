import torch
from thop import profile
import os
import time
import datetime
import argparse
import numpy as np


from src.models.UNet_2Plus import UNet_2Plus
from src.models.UNet_3Plus import UNet_3Plus
from src.module.unet import UNet
from src.module.unet10 import UNet10
from src.module.unet11 import UNet11
from src.module.unet15 import UNet15
from src.module.unet16 import UNet16
from src.module.unet17 import UNet17
from src.module.unet18 import UNet18
from src.module.unet19 import UNet19
from src.module.unet20 import UNet20
from src.module.unet21 import UNet21
from src.module.unet22 import UNet22
from src.module.unet23 import UNet23
from src.module.unet24 import UNet24
from src.networks import nnUNet, UNetPP, UNetPPP, VNet


model = UNet(in_channels=3, num_classes=1)
input = torch.randn(2,3,480,480)
flops , params = profile(model, inputs=(input,))
print(f"FLOPs: {flops/1e9:.2f} G, Params: {params/1e6:.2f} M")