import numpy as np

# 创建一个形状为 (3, 2, 3, 5) 的数组
new_arr = np.zeros((3, 2, 3, 5))

# 创建一个形状为 (2, 3, 5) 的数组，包含不同的值
args = [np.arange(30).reshape(2, 3, 5) for _ in range(3)]

# 定义索引
idxes = [0, slice(0, 2, None), slice(0, 3, None), slice(0, 5, None)]
new_arr[tuple(idxes)] = args[0]
print(new_arr)
