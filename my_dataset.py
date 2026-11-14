import os
import numpy as np
import torch
from torch.utils.data import Dataset


class SynapseDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(SynapseDataset, self).__init__()
        # 确定读取 train_npz 还是 val_npz
        self.flag = "train_npz" if train else "val_npz"
        self.data_root = os.path.join(root, "Synapse", self.flag)
        assert os.path.exists(self.data_root), f"Path '{self.data_root}' does not exist."

        self.transforms = transforms

        # 获取所有 .npz 文件
        self.files = [os.path.join(self.data_root, f) for f in os.listdir(self.data_root) if f.endswith('.npz')]

    def __getitem__(self, idx):
        # 加载 npz 文件
        data = np.load(self.files[idx])
        image, label = data['image'], data['label']

        # 扩展单通道维度 (H, W) -> (1, H, W)
        image = np.expand_dims(image, axis=0)

        # 转换为 PyTorch Tensor
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()

        # 数据增强 (针对 Tensor 的基础增强)
        if self.transforms is not None:
            image, label = self.transforms(image, label)

        return image, label

    def __len__(self):
        return len(self.files)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = torch.stack(images, dim=0)
        batched_targets = torch.stack(targets, dim=0)
        return batched_imgs, batched_targets