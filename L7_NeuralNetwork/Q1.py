import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from neuralnetwork import NeuralNetwork
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 加载IRIS数据集
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.int64))
y_test = torch.from_numpy(y_test.astype(np.int64))

# 设置超参数
input_size = X.shape[1]
hidden_size = [10, 10]
output_size = len(np.unique(y))
learning_rate = 0.01
epochs = 30
batch_size = 32

# 定义模型
model = NeuralNetwork(input_size, hidden_size, output_size, activation_fn="relu")

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
losses = []
for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(X_train.size(0))
    for i in range(0, X_train.size(0), batch_size):
        indices = permutation[i : i + batch_size]
        batch_X, batch_Y = X_train[indices], y_train[indices]

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_Y)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}: loss = {loss.item():4f}")

# 测试模型
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    _, y_pred = torch.max(outputs.data, 1)
    y_pred = y_pred.numpy()
    label = y_test.numpy()
    accuracy = accuracy_score(label, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")

# 绘制损失函数
plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()
