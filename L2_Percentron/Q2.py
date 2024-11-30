# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from PLA import PLA
from pocket import Pocket
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate data
np.random.seed(42)
X1 = np.random.multivariate_normal([-5, 3], np.eye(2), size=200)
Y1 = np.ones(X1.shape[0])
X2 = np.random.multivariate_normal([0, 5], np.eye(2), size=200)
Y2 = -np.ones(X2.shape[0])
X = np.vstack((X1, X2))
Y = np.hstack((Y1, Y2))

# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# T在训练集上训练PLA和Pocket算法
max_iter = 1000
pla = PLA(X_train, Y_train, max_iter=max_iter)
pocket = Pocket(X_train, Y_train, max_iter=max_iter)


# 画图
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.title("PLA")
plt.scatter(
    X_train[Y_train == 1, 0],
    X_train[Y_train == 1, 1],
    c="r",
    marker="o",
    label="+1",
)
plt.scatter(
    X_train[Y_train == -1, 0],
    X_train[Y_train == -1, 1],
    c="b",
    marker="x",
    label="-1",
)
x1_min, x1_max = pla.X[:, 0].min(), pla.X[:, 0].max()
scale = (x1_max - x1_min) / 10
x1 = np.arange(x1_min - scale, x1_max + scale, scale)
x2 = (-pla.w[0] * x1 - pla.b) / pla.w[1]
plt.plot(x1, x2, c="g")
plt.legend()

plt.subplot(122)
plt.title("Pocket")
plt.scatter(
    X_train[Y_train == 1, 0],
    X_train[Y_train == 1, 1],
    c="r",
    marker="o",
    label="+1",
)
plt.scatter(
    X_train[Y_train == -1, 0],
    X_train[Y_train == -1, 1],
    c="b",
    marker="x",
    label="-1",
)
x1_min, x1_max = pla.X[:, 0].min(), pla.X[:, 0].max()
scale = (x1_max - x1_min) / 10
x1 = np.arange(x1_min - scale, x1_max + scale, scale)
x2 = (-pocket.w[0] * x1 - pocket.b) / pla.w[1]
plt.plot(x1, x2, c="g")
plt.legend()

plt.show()

# 计算PLA和Pocket算法在测试集上的准确率
Y_pred_pla = pla.predict(X_test)
Y_pred_pocket = pocket.predict(X_test)
acc_pla = accuracy_score(Y_test, Y_pred_pla)
acc_pocket = accuracy_score(Y_test, Y_pred_pocket)
print("PLA accuracy:", acc_pla)
print("Pocket accuracy:", acc_pocket)
