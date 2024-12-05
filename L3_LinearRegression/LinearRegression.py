import numpy as np
import matplotlib.pyplot as plt


def MSEfunction(y1: np.array, y2: np.array):
    return np.mean((y1 - y2) ** 2)


class GeneralizedInverse:
    def __init__(self, X, y):
        self.y = y
        self.X0 = X
        self.X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.w = np.linalg.inv(self.X.T.dot(self.X)).dot(self.X.T).dot(self.y)
        self.loss = MSEfunction(y, self.predict(X))

    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return X.dot(self.w)

    def classify(self, X):
        return np.sign(self.predict(X))

    def plot_2classes(self, X, y, show=False):
        plt.scatter(
            X[y == 1, 0],
            X[y == 1, 1],
            c="r",
            marker="o",
            label="+1",
        )
        plt.scatter(
            X[y == -1, 0],
            X[y == -1, 1],
            c="b",
            marker="x",
            label="-1",
        )
        Xmin = np.min(X[:, 0])
        Xmax = np.max(X[:, 0])
        x = np.linspace(Xmin - 0.1 * (Xmax - Xmin), Xmax + 0.1 * (Xmax - Xmin), 100)
        y = -self.w[0] / self.w[2] - self.w[1] / self.w[2] * x
        plt.plot(
            x,
            y,
            c="b",
            label="decision boundary",
        )
        plt.legend()
        plt.title("Generalized Inverse Method   loss=" + str(round(self.loss, 5)))
        if show:
            plt.show()


class LinearRegression:
    def __init__(self, X, y):
        self.y = y
        self.X = X
        self.w = np.random.randn(self.X.shape[1])
        self.b = np.random.randn()
        self.N = self.X.shape[0]
        self.loss = None

    def predict(self, X):
        return X.dot(self.w) + self.b

    def classify(self, X):
        return np.sign(self.predict(X))

    def train(self, lr=0.01, num_epochs=30, plot_loss=False):
        loss_history = []
        for epoch in range(num_epochs):
            y_pred = self.predict(self.X)
            loss = MSEfunction(self.y, y_pred)
            loss_history.append(loss)
            gradw = 2 / self.N * self.X.T.dot(y_pred - self.y)
            gradb = 2 / self.N * (y_pred - self.y).sum()
            self.w -= lr * gradw
            self.b -= lr * gradb
        self.loss = loss
        if plot_loss:
            plt.plot(loss_history)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Linear Regression   loss=" + str(round(loss, 5)))
            plt.show()

    def plot_2classes(self, X, y, show=False):
        plt.scatter(
            X[y == 1, 0],
            X[y == 1, 1],
            c="r",
            marker="o",
            label="+1",
        )
        plt.scatter(
            X[y == -1, 0],
            X[y == -1, 1],
            c="b",
            marker="x",
            label="-1",
        )
        Xmin = np.min(X[:, 0])
        Xmax = np.max(X[:, 0])
        x = np.linspace(Xmin - 0.1 * (Xmax - Xmin), Xmax + 0.1 * (Xmax - Xmin), 100)
        y = -self.w[0] / self.w[1] * x - self.b / self.w[1]
        plt.plot(
            x,
            y,
            c="b",
            label="decision boundary",
        )
        plt.legend()
        plt.title("Linear Regression   loss=" + str(round(self.loss, 5)))
        if show:
            plt.show()


"""
if __name__ == "__main__":
    np.random.seed(0)
    X1 = np.random.multivariate_normal([-5, 0], np.eye(2), size=200)
    Y1 = np.ones(X1.shape[0])
    X2 = np.random.multivariate_normal([0, 5], np.eye(2), size=200)
    Y2 = -np.ones(X2.shape[0])
    X = np.vstack((X1, X2))
    y = np.hstack((Y1, Y2))

    # model1 = GeneralizedInverse(X, y)
    # print(model1.loss)
    # model1.plot_2classes(X, y)
    # print(model1.w)

    model2 = LinearRegression(X, y)
    model2.train()
    model2.plot_2classes(X, y)
"""
