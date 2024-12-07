from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from Fisher import Fisher

# Generate data
np.random.seed(114514)
X1 = np.random.multivariate_normal([-5, 0], np.eye(2), size=200)
Y1 = np.ones(X1.shape[0])
X2 = np.random.multivariate_normal([0, 5], np.eye(2), size=200)
Y2 = -np.ones(X2.shape[0])
X = np.vstack((X1, X2))
Y = np.hstack((Y1, Y2))

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

model = Fisher(X_train, Y_train)

print(f"最佳投影向量为w = {model.w}")
print(f"最佳分类阈值为s = {model.threshold}")

# 在训练集上测试
y_pred = model.predict(X_train)
acc = accuracy_score(Y_train, y_pred)
print(f"训练集上的准确率为{acc}")

# 在测试集上测试
y_pred = model.predict(X_test)
acc = accuracy_score(Y_test, y_pred)
print(f"测试集上的准确率为{acc}")

# model.plot2classes(X, Y, show=True)
plt.figure(figsize=(10, 5.5))
plt.subplot(121)
model.plot2classes(X, Y, show=False)
plt.subplot(122)
model.visualize(X, Y, show=False)
plt.tight_layout()
plt.show()
