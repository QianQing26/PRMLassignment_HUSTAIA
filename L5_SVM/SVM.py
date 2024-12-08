import numpy as np
import matplotlib.pyplot as plt
import cvxopt as cvx
import cvxopt.solvers as cvx_solver
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
import kernel
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


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

    def __init__(
        self, kernel="rbf", C=1.0, degree=3, gamma="scale", coef0=0, random_state=42
    ):
        """
        初始化 SVM 模型
        kernel: 核函数类型，支持 'linear', 'poly', 'rbf', 'sigmoid'
        C: 正则化参数
        degree: 多项式核的阶数，仅在 kernel='poly' 时有效
        gamma: 核函数的超参数，仅在 rbf 和 poly 核时有效
        coef0: 核函数的常数项，仅在 poly 和 sigmoid 核时有效
        """
        self.kernel = kernel
        self.C = C
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.random_state = random_state

        # 初始化 SVC 模型
        self.model = SVC(
            kernel=self.kernel,
            C=self.C,
            degree=self.degree,
            gamma=self.gamma,
            coef0=self.coef0,
            random_state=self.random_state,
        )

    def fit(self, X, y):
        """
        训练 SVM 模型
        X: 特征矩阵
        y: 标签
        """
        # 标准化数据
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 训练模型
        self.model.fit(X_scaled, y)

    def predict(self, X):
        """
        预测新数据
        X: 新样本特征矩阵
        """
        # 标准化数据
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        return self.model.predict(X_scaled)

    def visualize(self, X, y, show=False):
        """
        绘制分类面间隔面
        """
        sns.set(style="whitegrid", palette="muted")

        # 标准化数据
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
        y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01)
        )
        Z = self.model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.75, cmap="coolwarm")
        plt.scatter(
            X_scaled[y == 1, 0],
            X_scaled[y == 1, 1],
            edgecolors="k",
            marker="o",
            cmap=plt.cm.RdYlBu,
            s=80,
        )
        plt.scatter(
            X_scaled[y == 0, 0],
            X_scaled[y == 0, 1],
            edgecolors="k",
            marker="o",
            cmap=plt.cm.RdYlBu,
            s=80,
        )
        plt.xlabel("Feature 1", fontsize=12)
        plt.ylabel("Feature 2", fontsize=12)
        plt.title("Classification Boundary", fontsize=14)

        if show:
            plt.show()


# class KernelSVM:
#     def __init__(self, kernel=kernel.PolynomialKernel(degree=2)):
#         self.alpha = None
#         self.SupportVector = None
#         self.kernel = kernel
#         self.w = None
#         self.b = None

#     def fit(self, X, y):
#         self.X = X
#         self.y = y
#         n_samples, n_features = X.shape
#         y = y.reshape(-1, 1)
#         Q = cvx.matrix((self.kernel(y * X, y * X)).astype(np.double))
#         # print(Q)
#         p = cvx.matrix(-1 * np.ones((n_samples, 1)).astype(np.double))
#         G = cvx.matrix(-1 * np.eye(n_samples).astype(np.double))
#         h = cvx.matrix(np.zeros((n_samples, 1)).astype(np.double))
#         A = cvx.matrix((y.T).astype(np.double))
#         b = cvx.matrix(0.0)
#         solution = cvx_solver.qp(Q, p, G, h, A, b)
#         self.alpha = np.array(solution["x"]).reshape(1, -1)[0]
#         self.SupportVector = X[self.alpha > 1e-4]
#         supportY = y[self.alpha > 1e-4]
#         supportAlpha = self.alpha[self.alpha > 1e-4]

#         self.w = (supportY.reshape(1, -1)[0] * supportAlpha.reshape(1, -1)[0]).reshape(
#             -1, 1
#         )

#         self.b = supportY[0] - np.sum(
#             self.w * self.kernel(self.SupportVector[0], self.SupportVector), axis=0
#         )

#     def predict(self, X):
#         return np.sign(self.kernel(X, self.w.reshape(1, -1)) + self.b)


if __name__ == "__main__":
    X = np.array([[2, 2], [-2, -2], [2, -2], [-2, 2]])
    y = np.array([1, 1, -1, -1])
    svm = DualSVM()
    svm.fit(X, y)
    print(svm.w)
    print(svm.b)
    print(svm.alpha)
    print(svm.SupportVector)
    # svm.visualize(X, y, show=True)
