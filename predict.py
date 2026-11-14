import os
import time

import torch
from torchvision import transforms
import numpy as np
from PIL import Image


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
from src.module.masku1 import MASKU1
from src.SURMT.Networks import SURMT


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    classes = 1  # exclude background

    weights_path = "save_weights/par9/best_model.pth"

    img_load = './data/mmwhsmri/train_data/image'

    roi_load = './data/mmwhsmri/train_data/mask'

    save_path = './prediction/mmwhsmri'

    # weights_path = "save_weights/nnunetbusi/best_model.pth"
    # img_load = './BUSI/test_data/image'
    # roi_load = './BUSI/test_data/mask'
    # save_path = './prediction/BUSI/nnunet'
    img_path_list = os.listdir(img_load)
    roi_mask_path_list = os.listdir(roi_load)
    for i in range(len(img_path_list)):
        img_path = img_load + '//' + img_path_list[i]
        roi_mask_path = roi_load + '//' + roi_mask_path_list[i]
        assert os.path.exists(weights_path), f"weights {weights_path} not found."
        assert os.path.exists(img_path), f"image {img_path} not found."
        assert os.path.exists(roi_mask_path), f"image {roi_mask_path} not found."

        mean = (0.709, 0.381, 0.224)
        std = (0.127, 0.079, 0.043)

        # load image
        original_img = Image.open(img_path).convert('RGB')

        # from pil image to tensor and normalize
        data_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=mean, std=std)])
        img = data_transform(original_img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)
        img_height, img_width = img.shape[-2:]

        # pad input to multiples of 32 for U-Net skip-connection alignment
        stride = 32
        pad_h = (stride - img_height % stride) % stride
        pad_w = (stride - img_width % stride) % stride
        if pad_h > 0 or pad_w > 0:
            img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), mode='reflect')

        # get devices
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # create model
        model = UNet(in_channels=3, num_classes=2)
        #model = SURMT(channels=3, classes=2)

        # load weights
        checkpoint = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model.to(device)

        # load roi mask
        roi_img = Image.open(roi_mask_path).convert('L')
        roi_img = roi_img.resize((img_width, img_height))
        roi_img = np.array(roi_img)


        model.eval()
        with torch.no_grad():
            # init model (padded size to avoid skip-connection mismatch)
            init_h, init_w = img.shape[-2:]
            init_img = torch.zeros((1, 3, init_h, init_w), device=device)
            model(init_img)

            output = model(img.to(device))
            # crop back to original size
            #output = output[..., :img_height, :img_width]
            print("{}/{}".format(i + 1,len(img_path_list)))


            prediction = output['out'].argmax(1).squeeze(0)
            #prediction = output.argmax(1).squeeze(0)
            prediction = prediction.to("cpu").numpy().astype(np.uint8)
            prediction[prediction == 1] = 255
            prediction[prediction == 0] = 0
            mask = Image.fromarray(prediction)
            mask.save(save_path + '//' + img_path_list[i])


if __name__ == '__main__':
    main()
