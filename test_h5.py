import h5py
import numpy as np

p = '/mnt/sda3/yigedabuliu/lkq/data/Task01_BrainTumour/h5_datasets/BraTS2021_00000_mri_norm2.h5'
h5f = h5py.File(p, 'r')
image = h5f['image'][:]
label = h5f['label'][:]
print('image shape:', image.shape, '\t', 'label shape', label.shape)
print('label set:', np.unique(label))

# image shape: (4, 240, 240, 155)          label shape (240, 240, 155)
# label set: [0 1 2 4]
