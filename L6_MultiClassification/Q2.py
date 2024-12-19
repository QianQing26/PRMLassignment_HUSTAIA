import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Softmax import SoftmaxClassifier
import numpy as np
import optimizer
import matplotlib.pyplot as plt


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
optimizer = optimizer.GradientDescent(learning_rate=0.1)
losses, accuracies = model.fit(
    X_train, y_train, optimizer=optimizer, num_epochs=10, batch_size=256
)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {accuracy}")
plt.figure()
plt.subplot(1, 2, 1)
plt.plot(accuracies)
plt.title("Accuracy")
plt.subplot(1, 2, 2)
plt.plot(losses)
plt.title("Loss")
plt.show()

# 在测试集上抽取10个样本，并可视化
sample_indices = np.random.choice(X_test.shape[0], 10)
sample_images = X_test[sample_indices]
sample_labels = y_test[sample_indices]
sample_predictions = model.predict(sample_images)

fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, (ax, img, label, pred) in enumerate(
    zip(axes.flatten(), sample_images, sample_labels, sample_predictions)
):
    ax.imshow(img.reshape((28, 28)), cmap="gray")
    ax.set_title(f"Label: {label}, Prediction: {pred}")
    ax.axis("off")
plt.tight_layout()
plt.show()
