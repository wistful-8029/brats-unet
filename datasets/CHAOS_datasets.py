import os
import torch
from torch.utils.data import Dataset
import random
import numpy as np
from torchvision.transforms import transforms
import h5py
from os.path import join, isfile, basename
from glob import glob


class CHAOS_h5_Dataset(Dataset):
    def __init__(self, data_root, type, modal, image_size=256, output_size=(256, 256), data_aug=True):
        self.data_root = data_root

        self.image_size = image_size
        self.data_aug = data_aug
        self.output_size = output_size
        self.type = type
        self.modal = modal  # [T1DUAL_InPhase、T1DUAL_OutPhase、T2SPIR]

        self.train_list = [1, 2, 3, 5, 8, 10, 13, 15, 19, 20, 21, 22, 31, 32, 33, 34, 36, 37, 38, 39]  # 官方给的训练集序列

        self.train_part = self.train_list[:-4]  # 训练
        self.val_part = self.train_list[-4:]  # 训练过程中用于验证的部分

        self.h5_files = []

        if type == 'train':
            self.h5_path = join(data_root, 'MR_' + self.modal, 'Train')

            for i in self.train_part:
                file_path = str(i) + '_h5_' + self.modal + '.h5'
                self.h5_files.append(join(self.h5_path, file_path))

            # self.h5_files = sorted(glob(join(self.h5_path, '*.h5'), recursive=True))
        elif type == 'val':
            self.h5_path = join(data_root, 'MR_' + self.modal, 'Train')

            for i in self.val_part:
                file_path = str(i) + '_h5_' + self.modal + '.h5'
                self.h5_files.append(join(self.h5_path, file_path))
        elif type == 'test':
            self.h5_path = join(data_root, 'MR_' + self.modal, 'Test')
            self.h5_files = sorted(glob(join(self.h5_path, '*.h5'), recursive=True))

    def __len__(self):
        return len(self.h5_files)

    def __getitem__(self, index):
        if self.type == 'train':
            h5f = h5py.File(self.h5_files[index], 'r')
            image = h5f['image'][:]
            label = h5f['label'][:]

            image, label = self.data_trans(image, label)

            return image, label

        elif self.type == 'val':
            h5f = h5py.File(self.h5_files[index], 'r')
            image = h5f['image'][:]
            label = h5f['label'][:]

            image, label = self.data_trans(image, label)

            return image, label
        elif self.type == 'test':
            h5f = h5py.File(self.h5_files[index], 'r')
            image = h5f['image'][:]

            image = self.data_trans(image)

            return image

    # elif
    def data_trans(self, img, label=None):
        """
            将数据转换为Tensor张量，并保持深度维度为偶数
        """
        if label is not None:
            if img.shape[-1] % 2 == 0:
                # 无需做任何变换
                return torch.tensor(img), torch.tensor(label, dtype=torch.int64)
            else:
                img_tensor = torch.tensor(img)
                label_tensor = torch.tensor(label, dtype=torch.int64)
                canvas_img_tensor = torch.zeros(img.shape[:-1]).unsqueeze(dim=-1)  # 创建一个全0张量
                img_tensor = torch.cat((img_tensor, canvas_img_tensor), dim=-1)
                canvas_label_tensor = torch.zeros(label.shape[:-1]).unsqueeze(dim=-1)
                label_tensor = torch.cat((label_tensor, canvas_label_tensor), dim=-1)
                label_tensor = label_tensor.to(torch.int64)

                return img_tensor, label_tensor
        else:
            if img.shape[-1] % 2 == 0:
                # 无需做任何变换
                return torch.tensor(img)
            else:
                img_tensor = torch.tensor(img)
                canvas_img_tensor = torch.zeros(img.shape[:-1]).unsqueeze(dim=-1)  # 创建一个全0张量
                img_tensor = torch.cat((img_tensor, canvas_img_tensor), dim=-1)

                return img_tensor


if __name__ == '__main__':
    data_root = '/mnt/sda3/yigedabuliu/lkq/data/MR/CHAOS/h5_datasets/'
    tr_dataset = CHAOS_h5_Dataset(data_root, data_aug=True, type='train', modal='MR_T1DUAL_InPhase')

    d1 = tr_dataset[0]
    image, label = d1
    print(image.shape)
    print(label.shape)
    print(np.unique(label))
