import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Softmax import SoftmaxClassifier
import numpy as np
import optimizer
import matplotlib.pyplot as plt

np.random.seed(12)


# 加载 MNIST 数据
def load_mnist(filepath):
    with np.load(filepath) as data:
        X_train = data["X_train"]
        y_train = data["y_train"]
        X_test = data["X_test"]
        y_test = data["y_test"]
    return X_train, y_train, X_test, y_test


# 加载数据
X_train, y_train, X_test, y_test = load_mnist("mnist_numpy_data.npz")

# 打印数据维度
print(f"Train features shape: {X_train.shape}, Train labels shape: {y_train.shape}")
print(f"Test features shape: {X_test.shape}, Test labels shape: {y_test.shape}")

X_train = X_train.reshape((X_train.shape[0], -1))
X_test = X_test.reshape((X_test.shape[0], -1))

model = SoftmaxClassifier()
# optimizer = optimizer.Adam(learning_rate=0.005)
optimizer = optimizer.Adam(learning_rate=0.01)
losses, accuracies = model.fit(
    X_train, y_train, optimizer=optimizer, num_epochs=10, batch_size=256
)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {accuracy}")
plt.figure()
plt.subplot(1, 2, 1)
plt.plot(accuracies)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy")
plt.grid()
plt.subplot(1, 2, 2)
plt.plot(losses)
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.show()

# # 在测试集上抽取10个样本，并可视化
# sample_indices = np.random.choice(X_test.shape[0], 10)
# sample_images = X_test[sample_indices]
# sample_labels = y_test[sample_indices]
# sample_predictions = model.predict(sample_images)

# fig, axes = plt.subplots(2, 5, figsize=(10, 5))
# for i, (ax, img, label, pred) in enumerate(
#     zip(axes.flatten(), sample_images, sample_labels, sample_predictions)
# ):
#     ax.imshow(img.reshape((28, 28)), cmap="gray")
#     ax.set_title(f"Label: {label}, Prediction: {pred}")
#     ax.axis("off")
# plt.tight_layout()
# plt.show()


# 找到分类正确和分类错误的样本索引
correct_indices = np.where(y_pred == y_test)[0]
incorrect_indices = np.where(y_pred != y_test)[0]

# 从正确分类和错误分类中各抽取10个样本
correct_sample_indices = np.random.choice(correct_indices, 10, replace=False)
incorrect_sample_indices = np.random.choice(incorrect_indices, 10, replace=False)

# 可视化分类正确的样本
correct_images = X_test[correct_sample_indices]
correct_labels = y_test[correct_sample_indices]
correct_predictions = y_pred[correct_sample_indices]

fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, (ax, img, label, pred) in enumerate(
    zip(axes.flatten(), correct_images, correct_labels, correct_predictions)
):
    ax.imshow(img.reshape((28, 28)), cmap="gray")
    ax.set_title(f"Label: {label}, Prediction: {pred}")
    ax.axis("off")
plt.suptitle("Correctly Classified Samples")
plt.tight_layout()
plt.show()

# 可视化分类错误的样本
incorrect_images = X_test[incorrect_sample_indices]
incorrect_labels = y_test[incorrect_sample_indices]
incorrect_predictions = y_pred[incorrect_sample_indices]

fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, (ax, img, label, pred) in enumerate(
    zip(axes.flatten(), incorrect_images, incorrect_labels, incorrect_predictions)
):
    ax.imshow(img.reshape((28, 28)), cmap="gray")
    ax.set_title(f"Label: {label}, Prediction: {pred}")
    ax.axis("off")
plt.suptitle("Incorrectly Classified Samples")
plt.tight_layout()
plt.show()
