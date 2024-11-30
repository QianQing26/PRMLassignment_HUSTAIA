# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt


class PLA:
    def __init__(self, X: np.ndarray, y: np.ndarray, eta=1, max_iter=1500):
        """
        初始化PLA模型.

        Parameters:
        - X: (n_samples, n_features) array, 输入数据
        - y: (n_samples,) array, 输入标签  +1 or -1
        - eta: float, 学习率
        - max_iter: int, 最大迭代次数
        """
        self.w = np.ones(X.shape[1])  # 初始化权重为全1
        self.b = 0.0  # 初始化偏置为0
        self.eta = eta
        self.max_iter = max_iter
        self.X = X
        self.y = y
        self.fit()

    def predict(self, X: np.ndarray):
        """
        对输入的数据X预测类别

        Parameters:
        - X: (n_samples, n_features) array, 输入数据

        Returns:
        - (n_samples,) array, 预测标签  +1 or -1
        """
        return np.sign(X.dot(self.w) + self.b)

    def fit(self):
        """
        PLA模型训练，更新权重w和偏置b，在构造函数中调用

        Returns:
        - (n_features,) array, 学习到的权重
        - float, learned bias
        """
        for _ in range(self.max_iter):
            # 对当前权重w和偏置b，对训练数据进行预测
            y_pred = self.predict(self.X)
            # 遍历训练集，对错误的样本更新
            idx = -1
            for i in range(self.X.shape[0]):
                if y_pred[i] != self.y[i]:
                    idx = i
                    break
            if idx == -1:
                # 所有样本都正确，停止训练
                break
            # 更新权重w和偏置b
            self.w += self.eta + self.X[idx] * self.y[idx]
            self.b += self.eta * self.y[idx]
        return self.w, self.b

    def plot(self):
        """
        画出数据点和分类面

        """
        # 画出标签为1的点和标签为-1的点
        plt.scatter(
            self.X[self.y == 1, 0],
            self.X[self.y == 1, 1],
            c="r",
            marker="o",
            label="+1",
        )
        plt.scatter(
            self.X[self.y == -1, 0],
            self.X[self.y == -1, 1],
            c="b",
            marker="x",
            label="-1",
        )
        # 画出分割超平面
        x1_min, x1_max = self.X[:, 0].min(), self.X[:, 0].max()
        scale = (x1_max - x1_min) / 10
        x1 = np.arange(x1_min - scale, x1_max + scale, scale)
        x2 = (-self.w[0] * x1 - self.b) / self.w[1]
        plt.plot(x1, x2, c="g")
        plt.legend()
        plt.show()


"""
# Example:
if __name__ == "__main__":
    # generate data
    np.random.seed(0)
    X1 = np.random.multivariate_normal([-5, 0], np.eye(2), size=200)
    Y1 = np.ones(X1.shape[0])
    X2 = np.random.multivariate_normal([0, 5], np.eye(2), size=200)
    Y2 = -np.ones(X2.shape[0])
    X = np.vstack((X1, X2))
    y = np.hstack((Y1, Y2))
    # train PLA
    pla = PLA(X, y, eta=1, max_iter=1000)
    # plot
    pla.plot()
"""
