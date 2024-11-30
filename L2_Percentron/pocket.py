# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt


class Pocket:
    def __init__(self, X: np.ndarray, y: np.ndarray, eta=1, max_iter=1500):
        """
        初始化Pocket模型.

        Parameters:
        - X: (n_samples, n_features) array, 输入数据
        - y: (n_samples,) array, 输入标签  +1 or -1
        - eta: float, 学习率
        - max_iter: int, 最大迭代次数
        """
        self.w = np.ones(X.shape[1])
        self.b = 0.0
        self.eta = eta
        self.max_iter = max_iter
        self.X = X
        self.y = y
        self.fit()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        对输入的数据X预测类别

        Parameters:
        - X: (n_samples, n_features) array, 输入数据

        Returns:
        - (n_samples,) array, 预测标签   +1 or -1
        """
        return np.sign(X.dot(self.w) + self.b)

    def model_predict(self, X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:

        return np.sign(X.dot(w) + b)

    def fit(self):
        """
        Pocket模型训练，更新权重w和偏置b，在构造函数中调用

        Returns:
        - (n_features,) array, 学习到的权重
        - float, 学习到的偏置
        """

        w = self.w
        b = self.b
        y_pred = self.predict(self.X)
        least_fault = np.sum(y_pred != self.y)
        for _ in range(self.max_iter):
            idx = -1
            y_pred = self.model_predict(self.X, w, b)
            fault_count = 0
            for i in range(self.X.shape[0]):
                if y_pred[i] != self.y[i]:
                    idx = i
                    fault_count += 1
            if idx == -1:
                self.w = w
                self.b = b
                break
            if fault_count < least_fault:
                least_fault = fault_count
                self.w = w
                self.b = b
            w += self.eta + self.y[idx] * self.X[idx]
            b += self.eta * self.y[idx]
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
    pla = Pocket(X, y, eta=1, max_iter=1000)
    # plot
    pla.plot()
"""
