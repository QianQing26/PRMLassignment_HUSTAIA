from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import SVM


# Generate data
np.random.seed(810)
X1 = np.random.multivariate_normal([-5, 0], np.eye(2), size=500)
Y1 = np.ones(X1.shape[0])
X2 = np.random.multivariate_normal([0, 5], np.eye(2), size=500)
Y2 = -np.ones(X2.shape[0])
X = np.vstack((X1, X2))
Y = np.hstack((Y1, Y2))

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


models = {
    "PrimalSVM": SVM.PrimalSVM(),
    "DualSVM  ": SVM.DualSVM(),
    "RBF_SVM  ": SVM.KernelSVM(kernel="rbf"),
    "4thPoly_SVM": SVM.KernelSVM(kernel="poly", degree=4),
}


for name, model in models.items():
    # Train the model
    model.fit(X_train, Y_train)

    Y_pred_train = model.predict(X_train)
    train_acc = accuracy_score(Y_pred_train, Y_train)
    Y_pred_test = model.predict(X_test)
    test_acc = accuracy_score(Y_pred_test, Y_test)

    print(f"{name}\tTrain Acc:{train_acc:.4f}\tTest Acc:{test_acc:.4f}")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, (name, model) in enumerate(models.items()):
    plt.sca(axes[i])
    model.visualize(X, Y, show=False)
    axes[i].set_title(f"{name}    acc:{accuracy_score(model.predict(X), Y):.4f}")

plt.tight_layout()
plt.show()


# 打印支撑向量
for name, model in models.items():
    print(f"{name} support vectors: \n{model.SupportVector}")
