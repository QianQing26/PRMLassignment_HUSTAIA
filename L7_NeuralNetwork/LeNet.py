import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=5,
            stride=1,
            padding=2,
        )
        self.sigmoid1 = nn.Sigmoid()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0
        )
        self.sigmoid2 = nn.Sigmoid()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.sigmoid3 = nn.Sigmoid()
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.sigmoid4 = nn.Sigmoid()
        self.fc3 = nn.Linear(in_features=84, out_features=10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 第一层卷积池化
        x = self.conv1(x)
        x = self.sigmoid1(x)
        x = self.pool1(x)
        # 第二层卷积池化
        x = self.conv2(x)
        x = self.sigmoid2(x)
        x = self.pool2(x)
        # 展平
        x = x.view(x.size(0), -1)
        # 全连接层
        x = self.fc1(x)
        x = self.sigmoid3(x)
        x = self.fc2(x)
        x = self.sigmoid4(x)
        x = self.fc3(x)
        # x = self.softmax(x)
        return x
