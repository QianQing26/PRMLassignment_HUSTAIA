import numpy as np
import matplotlib.pyplot as plt
import optimizer
import seaborn as sns


class logistic:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.w = np.zeros(X.shape[1])
        self.b = 0

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b - 0.5)

    def predict_prob(self, X):
        return self.sigmoid(np.dot(X, self.w) + self.b)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def loss(self, X, y):
        loss = np.sum(np.log(1 + np.exp(-y * (np.dot(X, self.w) + self.b))))
        return loss

    def compute_gradient(self, X, y, w, b):
        logits = y * (np.dot(X, w) + b)
        probs = self.sigmoid(logits)
        dlogits = -y * (1 - probs)
        dw = np.dot(dlogits, X) / X.shape[0]
        db = np.sum(dlogits) / X.shape[0]
        return {"w": dw, "b": db}

    def train(self, optimizer, num_epoch=100, batch_size=16, plot=False):
        params = {"w": self.w, "b": self.b}
        loss_history = []
        for epoch in range(num_epoch):
            indices = np.arange(self.X.shape[0])
            np.random.shuffle(indices)
            for i in range(0, self.X.shape[0], batch_size):
                X_batch = self.X[indices[i : i + batch_size]]
                y_batch = self.y[indices[i : i + batch_size]]
                grads = self.compute_gradient(
                    X_batch, y_batch, params["w"], params["b"]
                )
                optimizer.update(params, grads)
            self.w = params["w"]
            self.b = params["b"]
            loss = self.loss(self.X, self.y)
            print(f"Epoch {epoch+1} : loss = {loss}")
            if plot:
                loss_history.append(self.loss(self.X, self.y))
        if plot:
            # plt.plot(loss_history, label="Training Loss", color="blue", lw=1.5)
            sns.set_style("whitegrid")
            sns.lineplot(
                x=range(1, num_epoch + 1),
                y=loss_history,
                label="Training Loss",
                color="blue",
                linewidth=2,
            )
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title("Trainning Loss Over Epochs")
            plt.grid(alpha=0.3)
            plt.legend(fontsize=12, loc="best")
            plt.show()

    def plot2classes(self, X, y, show=False):
        sns.set(style="whitegrid")
        # plt.figure(figsize=(8, 6))
        sns.scatterplot(
            x=X[y == 1, 0],
            y=X[y == 1, 1],
            color="red",
            edgecolor="k",
            s=50,
            alpha=0.7,
            label="Class +1",
        )
        sns.scatterplot(
            x=X[y == -1, 0],
            y=X[y == -1, 1],
            color="blue",
            edgecolor="k",
            s=50,
            alpha=0.7,
            label="Class -1",
        )
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min = -self.w[0] / self.w[1] * x_min - self.b / self.w[1]
        y_max = -self.w[0] / self.w[1] * x_max - self.b / self.w[1]
        plt.plot(
            [x_min, x_max],
            [y_min, y_max],
            "k--",
            lw=2,
            label="Decision Boundary",
        )
        plt.xlabel("Feature 1", fontsize=12)
        plt.ylabel("Feature 2", fontsize=12)
        plt.title("Decision Boundary and Data Points", fontsize=14)
        plt.legend(fontsize=12)
        if show:
            plt.show()

    def visualize_prob(self, X, y, show=False):
        sns.set(style="whitegrid")
        prob = self.predict_prob(X)
        # plt.figure(figsize=(8, 6))
        sns.histplot(
            prob[y == 1],
            bins=20,
            kde=True,
            color="red",
            alpha=0.6,
            label="Class +1",
        )
        sns.histplot(
            prob[y == -1],
            bins=20,
            kde=True,
            color="blue",
            alpha=0.6,
            label="Class -1",
        )
        plt.axvline(
            x=0.5, color="k", linestyle="--", linewidth=2, label="Threshold 0.5"
        )
        plt.xlabel("Predicted Probability", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.title("Predicted Probability Distribution", fontsize=14)
        plt.legend(fontsize=12)
        if show:
            plt.show()
