import numpy as np
import matplotlib.pyplot as plt


class Fisher:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.w = None
        self.threshold = None
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.fit()

    def fit(self):
        X1 = self.X[self.y == 1]
        X2 = self.X[self.y == -1]
        mu1 = np.mean(X1, axis=0)
        mu2 = np.mean(X2, axis=0)
        S1 = np.cov(X1.T)
        S2 = np.cov(X2.T)
        # print(S1, S2)
        Sw = S1 + S2
        self.w = np.linalg.pinv(Sw) @ (mu1 - mu2)
        self.threshold = self.w @ (mu1 + mu2) / 2
        # print(self.w, self.threshold)

    def projection(self, X):
        return X @ self.w

    def predict(self, X):
        return np.sign(self.projection(X) - self.threshold)

    def plot2classes(self, X, y, show=False):
        plt.scatter(
            X[y == 1, 0],
            X[y == 1, 1],
            c="red",
            marker="o",
            edgecolors="k",
            s=25,
            alpha=0.7,
            label="+1",
        )
        plt.scatter(
            X[y == -1, 0],
            X[y == -1, 1],
            c="blue",
            marker="x",
            label="-1",
            s=25,
            alpha=0.7,
        )
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min = (self.threshold - self.w[0] * x_min) / self.w[1]
        y_max = (self.threshold - self.w[0] * x_max) / self.w[1]
        plt.xlabel("Feature 1", fontsize=12)
        plt.ylabel("Feature 2", fontsize=12)
        plt.plot([x_min, x_max], [y_min, y_max], "k--", label="Decision Boundary")
        plt.legend(fontsize=12, loc="best")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.title("Fisher's Linear Discriminant", fontsize=14)
        if show:
            plt.show()

    def visualize(self, X, y, show=False):
        projections = self.projection(X)
        plt.hist(
            projections[y == 1],
            bins=30,
            alpha=0.6,
            label="+1",
            color="red",
        )
        plt.hist(
            projections[y == -1],
            bins=30,
            alpha=0.6,
            label="-1",
            color="blue",
        )
        plt.axvline(
            self.threshold,
            linestyle="--",
            label="Decision Boundary",
            linewidth=2,
            color="black",
        )
        plt.legend(loc="best")
        plt.xlabel("Projection Value", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.title("Projections of data points after projection", fontsize=14)
        if show:
            plt.show()


# if __name__ == "__main__":
# X = np.array([[1, 5], [2, 4], [3, 8], [2, 4], [9, 4], [10, 10]])
# y = np.array([1, 1, -1, -1, 1, 1])
# fisher = Fisher(X, y)
# fisher.fit()
# y_pred = fisher.predict(X)
# print(y_pred)
