import numpy as np
from PLA import PLA
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class OVOwithPLA:
    def __init__(self, X_train, y_train, learning_rate=0.5, max_iter=1500):
        self.X_train = X_train
        self.y_train = y_train
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.classifiers = {}
        self.classes = np.unique(y_train)

        # 构建所有二分类器
        for i, class_1 in enumerate(self.classes):
            for class_2 in self.classes[i + 1 :]:
                idx = np.isin(y_train, [class_1, class_2])  # 筛选两类数据的索引
                X = X_train[idx]
                y = np.where(y_train[idx] == class_1, 1, -1)
                self.classifiers[(class_1, class_2)] = PLA(
                    X, y, self.learning_rate, self.max_iter
                )

    def predict(self, X):
        n_samples = X.shape[0]
        votes = np.zeros((n_samples, len(self.classes)))  # 每个样本对每个类的投票计数

        # 获取所有二分类器的预测结果
        for (class_1, class_2), classifier in self.classifiers.items():
            y_pred = classifier.predict(X)
            votes[:, class_1] += (y_pred == 1).astype(int)
            votes[:, class_2] += (y_pred == -1).astype(int)

        # 返回投票数最多的类别
        return self.classes[np.argmax(votes, axis=1)]
