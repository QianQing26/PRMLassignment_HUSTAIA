import numpy as np


def hinge_loss(w, X, y, lambda_reg=0.1):
    return np.sum(np.maximum(0, 1 - y * (X @ w))) + 0.5 * w.T @ w * lambda_reg


import numpy as np


def hinge_loss_grad(w, X, y, lambda_reg=0.1):
    # 计算每个样本的hinge损失梯度
    n = X.shape[0]
    gradient = np.zeros_like(w)

    for i in range(n):
        # 计算每个样本的损失
        if 1 - y[i] * np.dot(X[i], w) > 0:
            gradient -= y[i] * X[i]

    # 加上正则化梯度
    gradient += lambda_reg * w
    return gradient


X = np.array([[1, 1], [2, 2], [2, 0], [0, 0], [1, 0], [0, 1]])
y = np.array([1, 1, 1, -1, -1, -1])
X = np.hstack((np.ones((X.shape[0], 1)), X))
w0 = np.zeros(X.shape[1])

max_iter = 200000
eta = 0.0005

for i in range(max_iter):
    w = w0 - eta * hinge_loss_grad(w0, X, y)
    if np.linalg.norm(w - w0) < 1e-5:
        break
    w0 = w

print(w)
