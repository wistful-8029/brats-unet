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
        self.modal = modal  # [MR_T1DUAL_InPhase、MR_T1DUAL_OutPhase、MR_T2SPIR]

        if type == 'train':
            self.h5_path = join(data_root, self.modal, 'Train')

            # print(f'h5 path:{self.h5_path}')

            self.h5_files = sorted(glob(join(self.h5_path, '*.h5'), recursive=True))


        elif type == 'test':
            self.h5_path = join(data_root, self.modal, 'Test')
            self.h5_files = sorted(glob(join(self.h5_path, '*.h5'), recursive=True))
        pass

    def __len__(self):
        return len(self.h5_files)

    def __getitem__(self, index):
        if self.type == 'train':
            h5f = h5py.File(self.h5_files[index], 'r')
            image = h5f['image'][:]
            label = h5f['label'][:]

            image, label = self.data_trans(image, label)

            return image, label

        elif self.type == 'test':
            pass

    # elif
    def data_trans(self, img, label):
        """
            将数据转换为Tensor张量，并保持深度维度为偶数
        """
        if img.shape[-1] % 2 == 0:
            # 无需做任何变换
            return torch.tensor(img), torch.tensor(label)
        else:
            img_tensor = torch.tensor(img)
            label_tensor = torch.tensor(label)

            canvas_img_tensor = torch.zeros(img.shape[:-1]).unsqueeze(dim=-1)  # 创建一个全0张量

            img_tensor = torch.cat((img_tensor, canvas_img_tensor), dim=-1)

            canvas_label_tensor = torch.zeros(label.shape[:-1]).unsqueeze(dim=-1)

            label_tensor = torch.cat((label_tensor, canvas_label_tensor), dim=-1)

            return img_tensor, label_tensor

if __name__ == '__main__':
    from torchvision import transforms

    data_root = '/mnt/sda3/yigedabuliu/lkq/data/MR/CHAOS/h5_datasets/'
    tr_dataset = CHAOS_h5_Dataset(data_root, data_aug=True, type='train', modal='MR_T1DUAL_InPhase')

    d1 = tr_dataset[0]
    image, label = d1
    print(image.shape)
    print(label.shape)
    print(np.unique(label))
