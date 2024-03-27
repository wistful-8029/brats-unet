import os
from sklearn.model_selection import train_test_split

# 预处理输出地址
data_path = "/mnt/sda3/yigedabuliu/lkq/data/Task01_BrainTumour_h5"
train_and_test_ids = os.listdir(data_path)

train_ids, val_test_ids = train_test_split(train_and_test_ids, test_size=0.2,random_state=21)
val_ids, test_ids = train_test_split(val_test_ids, test_size=0.5,random_state=21)
print("Using {} images for training, {} images for validation, {} images for testing.".format(len(train_ids),len(val_ids),len(test_ids)))

train_ids.sort()
val_ids.sort()
test_ids.sort()

with open('/mnt/sda3/yigedabuliu/lkq/data/train.txt','w') as f:
    f.write('\n'.join(train_ids))

with open('/mnt/sda3/yigedabuliu/lkq/data/valid.txt','w') as f:
    f.write('\n'.join(val_ids))

with open('/mnt/sda3/yigedabuliu/lkq/data/test.txt','w') as f:
    f.write('\n'.join(test_ids))
