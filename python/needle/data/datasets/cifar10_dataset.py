import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset
# CIFAR-10 数据集路径
data_path = "/home/ljq/gpu/hw4/data/cifar-10-batches-py"

# CIFAR-10 类别
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 读取单个批次文件
def load_batch(batch_file):
    with open(batch_file, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        data = batch[b'data']
        labels = batch[b'labels']
        # 还原图像的形状 (num_samples, 3, 32, 32)
        data = data.reshape(-1, 3, 32, 32)
        return data, labels
    
def load_training_data(data_path):
    train_data = []
    train_labels = []
    # 读取 5 个训练批次
    for i in range(1, 6):
        batch_file = os.path.join(data_path, f'data_batch_{i}')
        data, labels = load_batch(batch_file)
        train_data.append(data)
        train_labels += labels
    # 将所有批次的数据合并
    train_data = np.concatenate(train_data)
    return train_data, train_labels

# 加载测试数据
def load_test_data(data_path):
    test_file = os.path.join(data_path, 'test_batch')
    return load_batch(test_file)
class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        self.train = train
        if(train):
            self.data, self.label = load_training_data(base_folder)
        else:
            self.data, self.label = load_test_data(base_folder)
        self.transforms = transforms
        self.p = p
        #raise NotImplementedError()
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        if isinstance(index, (np.ndarray, list)):
        # 如果是数组或列表，返回批量数据
            return (self.data[index], [self.label[i] for i in index])
        else:
        # 如果是单个索引，直接返回单个数据
            return (self.data[index], self.label[index])
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return len(self.data)
        raise NotImplementedError()
        ### END YOUR SOLUTION
