import numpy as np
from sklearn.preprocessing import OneHotEncoder
import optimizer


# class SoftmaxClassifier:
#     def __init__(self):
#         self.w = None
#         self.b = None
#         self.n_samples = None
#         self.n_features = None
#         self.n_classes = None

#     def predict(self, X):
#         s = np.dot(X, self.w) + self.b
#         probs = self.softmax(s)
#         return np.argmax(probs, axis=1)

#     def softmax(self, X):
#         Y = np.exp(X - np.max(X, axis=1, keepdims=True))
#         return Y / np.sum(Y, axis=1, keepdims=True)

#     def cross_entropy(self, probs, y):
#         return -np.mean(np.sum(y * np.log(probs + 1e-9), axis=1))

#     def compute_gradient(self, X, y, probs):
#         """
#         计算损失函数相对于权重和偏置的梯度
#         """
#         m = X.shape[0]
#         dw = np.dot(X.T, (probs - y)) / m
#         db = np.sum(probs - y, axis=0) / m
#         return {"w": dw, "b": db}

#     def fit(self, X, y, optimizer=optimizer.Adam(), batch_size=8, num_epochs=10):
#         loss_history = []
#         self.n_samples, self.n_features = X.shape
#         self.n_classes = len(np.unique(y))
#         self.w = np.zeros((self.n_features, self.n_classes))
#         self.b = np.zeros(self.n_classes)
#         # One-hot encoding
#         y_onehot = (
#             OneHotEncoder(categories="auto").fit_transform(y.reshape(-1, 1)).toarray()
#         )
#         # print(self.w)
#         params = {"w": self.w, "b": self.b}

#         # print(params)
#         # return
#         for epoch in range(num_epochs):
#             # Shuffle the data
#             idx = np.random.permutation(self.n_samples)
#             X_shuffled = X[idx]
#             y_shuffled = y_onehot[idx]
#             for i in range(0, self.n_samples, batch_size):
#                 X_batch = X_shuffled[i : i + batch_size]
#                 y_batch = y_shuffled[i : i + batch_size]
#                 # Forward pass
#                 probs = self.softmax(np.dot(X_batch, self.w) + self.b)
#                 # Compute loss and gradients
#                 loss = self.cross_entropy(probs, y_batch)
#                 loss_history.append(loss)
#                 grads = self.compute_gradient(X_batch, y_batch, probs)
#                 # Update parameters
#                 params = optimizer.update(params, grads)
#                 self.w = params["w"]
#                 self.b = params["b"]
#         return loss_history


import numpy as np


class SoftmaxClassifier:
    def __init__(self):
        self.w = None
        self.b = None
        self.n_samples = None
        self.n_features = None
        self.n_classes = None

    def predict(self, X):
        s = np.dot(X, self.w) + self.b
        probs = self.softmax(s)
        return np.argmax(probs, axis=1)

    def softmax(self, X):
        exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))  # 数值稳定性优化
        return exp_X / np.sum(exp_X, axis=1, keepdims=True)

    def cross_entropy(self, probs, y):
        return -np.mean(np.sum(y * np.log(probs + 1e-12), axis=1))  # 加入防护值

    def compute_gradient(self, X, y, probs):
        m = X.shape[0]
        dw = np.dot(X.T, (probs - y)) / m
        db = np.sum(probs - y, axis=0) / m
        return {"w": dw, "b": db}

    def fit(self, X, y, optimizer, batch_size=4, num_epochs=100, verbose=True):
        loss_history = []
        self.n_samples, self.n_features = X.shape
        self.n_classes = len(np.unique(y))
        self.w = np.random.randn(self.n_features, self.n_classes) * 0.01  # 随机初始化
        self.b = np.zeros(self.n_classes)
        y_onehot = np.eye(self.n_classes)[y]  # 用 NumPy 实现 one-hot 编码

        params = {"w": self.w, "b": self.b}
        # print(params["w"])

        for epoch in range(num_epochs):
            # Shuffle the data
            idx = np.random.permutation(self.n_samples)
            X_shuffled = X[idx]
            y_shuffled = y_onehot[idx]

            for i in range(0, self.n_samples, batch_size):
                X_batch = X_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]
                # Forward pass
                probs = self.softmax(np.dot(X_batch, self.w) + self.b)
                # Compute loss and gradients
                loss = self.cross_entropy(probs, y_batch)
                loss_history.append(loss)
                grads = self.compute_gradient(X_batch, y_batch, probs)
                # Update parameters
                optimizer.update(params, grads)
                # print(params)
                self.w = params["w"]
                self.b = params["b"]

            if verbose:
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")

        return loss_history
