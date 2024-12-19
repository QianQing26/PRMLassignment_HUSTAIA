import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
        cvx.solvers.options["show_progress"] = False
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
        solution = cvx_solver.qp(P, q, G, h, show_progress=False)
        self.w = np.array(solution["x"][1:]).reshape(-1, 1)
        self.b = np.array(solution["x"][0])
        self.SupportVector = self.FindSupportVector(X0, y)

    def FindSupportVector(self, X, y):
        distances = y * (np.dot(X, self.w) + self.b)
        SupportVectorIndices = np.where(np.isclose(distances, 1, atol=1e-4))[0]
        return X[SupportVectorIndices]

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


class DualSVM:
    def __init__(self):
        self.alpha = None
        self.SupportVector = None
        self.w = None
        self.b = None

    def fit(self, X, y):
        cvx.solvers.options["show_progress"] = False
        n_samples, n_features = X.shape
        y = y.reshape(-1, 1)
        Q = cvx.matrix(((y * X).dot((y * X).T)).astype(np.double))
        p = cvx.matrix(-1 * np.ones((n_samples, 1)))
        G = cvx.matrix(-1 * np.eye(n_samples))
        h = cvx.matrix(np.zeros((n_samples, 1)))
        A = cvx.matrix((y.T).astype(np.double))
        b = cvx.matrix(0.0)
        solution = cvx_solver.qp(Q, p, G, h, A, b, show_progress=False)
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
    def __init__(self, kernel="rbf", C=99999, degree=4, gamma="scale"):
        self.kernel = kernel
        self.C = C
        self.degree = degree
        self.gamma = gamma
        self.model = None
        self.SupportVector = None

    def fit(self, X, y):
        self.model = SVC(
            kernel=self.kernel, C=self.C, degree=self.degree, gamma=self.gamma
        )
        self.model.fit(X, y)
        self.SupportVector = np.array(self.model.support_vectors_)

    def predict(self, X):
        return self.model.predict(X)

    def visualize(self, X, y, show=False):
        sns.set(style="whitegrid", palette="muted")
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

        ax = plt.gca()
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        XX, YY = np.meshgrid(
            np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100)
        )
        Z = self.model.decision_function(np.c_[XX.ravel(), YY.ravel()])
        Z = Z.reshape(XX.shape)
        ax.contour(
            XX,
            YY,
            Z,
            levels=[0],
            linewidths=2,
            colors="black",
        )  # 分类面
        ax.contour(
            XX, YY, Z, levels=[-1, 1], linewidths=1, linestyles="--", colors="gray"
        )  # 间隔面
        legend_elements = [
            Line2D([0], [0], color="black", lw=2, label="Decision Boundary"),
            Line2D(
                [0], [0], color="gray", lw=1, linestyle="--", label="Margin Boundary"
            ),
        ]
        ax.legend(handles=ax.get_legend_handles_labels()[0] + legend_elements)
        plt.title(f"{self.kernel.capitalize()} Kernel SVM")
        plt.xlabel("Feature 1", fontsize=12)
        plt.ylabel("Feature 2", fontsize=12)
        if show:
            plt.show()
