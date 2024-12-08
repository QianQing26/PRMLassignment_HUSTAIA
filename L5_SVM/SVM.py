import numpy as np
import matplotlib.pyplot as plt
import cvxopt as cvx
import cvxopt.solvers as cvx_solver
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures


class PrimalSVM:
    def __init__(self):
        self.w = None
        self.b = None
        self.SupportVector = None

    def fit(self, X, y):
        d = X.shape[1]
        X0 = X
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        y = y.reshape(-1, 1)
        P = np.eye(d + 1)
        P[0, 0] = 0
        P = cvx.matrix(P)
        q = cvx.matrix(np.zeros((d + 1, 1)))
        G = cvx.matrix(-1 * y * X)
        h = cvx.matrix(-1 * np.ones((X.shape[0], 1)))
        solution = cvx_solver.qp(P, q, G, h)
        self.w = np.array(solution["x"][1:]).reshape(-1, 1)
        self.b = np.array(solution["x"][0])
        self.SupportVector = self.FindSupportVector(X0, y)
        print(self.SupportVector)

    def FindSupportVector(self, X, y):
        distances = y * (np.dot(X, self.w) + self.b)
        SupportVectorIndices = np.where(np.isclose(distances, 1, atol=1e-4))[0]
        return X[SupportVectorIndices]

    def predict(self, X):
        return np.sign(X.dot(self.w) + self.b)


class DualSVM:
    def __init__(self):
        self.alpha = None
        self.SupportVector = None
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y = y.reshape(-1, 1)
        Q = cvx.matrix(((y * X).dot((y * X).T)).astype(np.double))
        p = cvx.matrix(-1 * np.ones((n_samples, 1)))
        G = cvx.matrix(-1 * np.eye(n_samples))
        h = cvx.matrix(np.zeros((n_samples, 1)))
        A = cvx.matrix((y.T).astype(np.double))
        b = cvx.matrix(0.0)
        solution = cvx_solver.qp(Q, p, G, h, A, b)
        # print(solution)
        self.alpha = np.array(solution["x"]).reshape(1, -1)[0]
        self.SupportVector = X[self.alpha > 1e-4]
        self.w = np.sum((self.alpha * (y.reshape(1, -1)[0])).reshape(-1, 1) * X, axis=0)
        # print(self.w)
        index = np.random.choice(np.where(self.alpha > 1e-4)[0])
        self.b = y[index] - np.dot(self.w.T, X[index])
        # print(self.b)

    def predict(self, X):
        return np.sign(X.dot(self.w) + self.b)

    def visualize(self, X, y, show=False):
        sns.set(style="whitegrid", palette="muted")  # 设置背景和调色板

        # 绘制正类和负类数据点
        sns.scatterplot(
            x=X[y == 1, 0],
            y=X[y == 1, 1],
            color="red",
            edgecolor="black",
            s=40,
            alpha=0.4,
            label="Class +1",
        )

        sns.scatterplot(
            x=X[y == -1, 0],
            y=X[y == -1, 1],
            color="blue",
            edgecolor="black",
            s=40,
            alpha=0.4,
            label="Class -1",
        )

        # 绘制支持向量
        sns.scatterplot(
            x=self.SupportVector[:, 0],
            y=self.SupportVector[:, 1],
            color="green",
            marker="x",
            s=200,
            label="Support Vectors",
            linewidth=1.2,
        )

        # 绘制决策边界和边缘
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min = -self.w[0] / self.w[1] * x_min - self.b / self.w[1]
        y_max = -self.w[0] / self.w[1] * x_max - self.b / self.w[1]
        plt.plot([x_min, x_max], [y_min, y_max], "k-", lw=2, label="Decision Boundary")

        # 上边缘和下边缘
        y_min_upper = -self.w[0] / self.w[1] * x_min - (self.b - 1) / self.w[1]
        y_max_upper = -self.w[0] / self.w[1] * x_max - (self.b - 1) / self.w[1]
        plt.plot(
            [x_min, x_max],
            [y_min_upper, y_max_upper],
            "k--",
            lw=1,
            label="Margin Boundary",
        )

        y_min_lower = -self.w[0] / self.w[1] * x_min - (self.b + 1) / self.w[1]
        y_max_lower = -self.w[0] / self.w[1] * x_max - (self.b + 1) / self.w[1]
        plt.plot([x_min, x_max], [y_min_lower, y_max_lower], "k--", lw=1)

        # 设置图形标签
        plt.xlabel("Feature 1", fontsize=12)
        plt.ylabel("Feature 2", fontsize=12)
        plt.title("Decision Boundary and Data Points", fontsize=14)
        plt.legend(fontsize=12, loc="best")

        # 显示图像
        if show:
            plt.show()


class KernelSVM:
    def __init___(self, kernel, degree=2):
        self.alpha = None
        self.SupportVector = None
        self.w = None
        self.b = None
        if kernel == "Poly":
            self.kernel = PolynomialFeatures(degree=degree, include_bias=False)

    def Phi(self, x):
        pass

    def K(self, X1, X2):
        pass

    def fit(self, X, y):
        pass


if __name__ == "__main__":
    X = np.array([[1, 1], [2, 2], [2, 0], [0, 0], [1, 0], [0, 1]])
    y = np.array([1, 1, 1, -1, -1, -1])
    svm = DualSVM()
    svm.fit(X, y)
    svm.visualize(X, y, show=True)
