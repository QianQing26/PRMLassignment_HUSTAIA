# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from LinearRegression import LinearRegression, GeneralizedInverse

# Generate data
np.random.seed(42)
X1 = np.random.multivariate_normal([-5, 0], np.eye(2), size=200)
Y1 = np.ones(X1.shape[0])
X2 = np.random.multivariate_normal([0, 5], np.eye(2), size=200)
Y2 = -np.ones(X2.shape[0])
X = np.vstack((X1, X2))
Y = np.hstack((Y1, Y2))

# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

model1 = GeneralizedInverse(X=X_train, y=Y_train)
model2 = LinearRegression(X=X_train, y=Y_train)
model2.train(lr=0.0025, num_epochs=60, plot_loss=True)

# 广义逆
y_pred11 = model1.classify(X_train)
acc11 = accuracy_score(Y_train, y_pred11)
print("Accuracy of Generalized Inverse Method on Training Set:", acc11)
y_pred12 = model1.classify(X_test)
acc12 = accuracy_score(Y_test, y_pred12)
print("Accuracy of Generalized Inverse Method on Testing Set:", acc12)

# 线性回归
y_pred21 = model2.classify(X_train)
acc21 = accuracy_score(Y_train, y_pred21)
print("Accuracy of Linear Regression with GD on Training Set:", acc21)
y_pred22 = model2.classify(X_test)
acc22 = accuracy_score(Y_test, y_pred22)
print("Accuracy of Linear Regression with GD on Testing Set:", acc22)


# Plot the decision boundary
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.title("Generalized Inverse Method")
model1.plot_2classes(X, Y)
plt.subplot(1, 2, 2)
plt.title("Linear Regression with GD")
model2.plot_2classes(X, Y)
plt.show()
