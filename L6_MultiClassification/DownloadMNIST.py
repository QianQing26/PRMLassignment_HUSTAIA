import torch
from torchvision import datasets, transforms
import numpy as np
import os

# 设置保存文件路径
save_path = "mnist_numpy_data.npz"

# 下载 MNIST 数据集
transform = transforms.Compose([transforms.ToTensor()])

# 下载训练集和测试集
train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

# 提取数据和标签
X_train = train_dataset.data.numpy()  # 训练集特征 (28x28 灰度图)
y_train = train_dataset.targets.numpy()  # 训练集标签

X_test = test_dataset.data.numpy()  # 测试集特征 (28x28 灰度图)
y_test = test_dataset.targets.numpy()  # 测试集标签

# 保存为 .npz 文件
np.savez(save_path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
print(f"MNIST dataset saved as {save_path}.")
