import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

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

# 加载训练数据
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

# 展示一张图片
def show_image(img, label):
    # 转换 (3, 32, 32) -> (32, 32, 3)
    img = np.transpose(img, (1, 2, 0))
    plt.imshow(img)
    plt.title(f'Label: {classes[label]}')
    plt.show()

# 加载训练和测试数据
train_data, train_labels = load_training_data(data_path)
test_data, test_labels = load_test_data(data_path)

# 打印训练数据和标签的形状
print(f'Training data shape: {train_data.shape}')  # 应为 (50000, 3, 32, 32)
print(f'Training labels length: {len(train_labels)}')  # 应为 50000
print(f'Test data shape: {test_data.shape}')  # 应为 (10000, 3, 32, 32)
print(f'Test labels length: {len(test_labels)}')  # 应为 10000

# 展示一张训练图片
show_image(train_data[0], train_labels[0])
