from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import SVM
import kernel
from sklearn.datasets import make_circles


# Generate data
# np.random.seed(1919)
# X1 = np.random.multivariate_normal([-5, 0], np.eye(2), size=200)
# Y1 = np.ones(X1.shape[0])
# X2 = np.random.multivariate_normal([0, 5], np.eye(2), size=200)
# Y2 = -np.ones(X2.shape[0])
# X = np.vstack((X1, X2))
# Y = np.hstack((Y1, Y2))
X, Y = make_circles(n_samples=100, shuffle=True, noise=0.1, factor=0.5)

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# print("Training data shape:", X_train.shape)
# print("Testing data shape:", Y_train.shape)
# model = SVM.DualSVM()
# model.fit(X_train, Y_train)
# Y_pred = model.predict(X_test)
# accuracy = accuracy_score(Y_test, Y_pred)
# print("Accuracy:", accuracy)

# model.visualize(X, Y)
# plt.show()

model = SVM.KernelSVM(kernel="rbf")
model.fit(X_train, Y_train)
# print(model.alpha)
# for i in range(len(model.alpha)):
#     if model.alpha[i] > 1e-4:
#         plt.scatter(
#             X[i][0], X[i][1], s=100, marker="o", color="red" if Y[i] == 1 else "blue"
#         )
# plt.show()
model.visualize(X_train, Y_train)
plt.show()
# print(model.w)
# print(model.b)
# Y_pred = model.predict(X_test)
# print(Y_pred)
# accuracy = accuracy_score(Y_test, Y_pred)
# print("Accuracy:", accuracy)
