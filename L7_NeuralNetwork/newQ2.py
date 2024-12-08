import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from LeNet import LeNet  # 假定 LeNet 已实现
from sklearn.metrics import accuracy_score
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

# 设置超参数
learning_rate = 0.001
epochs = 10
batch_size = 256
train_flag = True

# 数据加载与处理
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # MNIST是灰度图像，单通道归一化
    ]
)

train_dataset = datasets.MNIST(
    root="./data", train=True, transform=transform, download=True
)
test_dataset = datasets.MNIST(
    root="./data", train=False, transform=transform, download=True
)
print("数据加载完成")
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

# 定义模型
model = LeNet().to(device)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print("模型定义完成")


def compute_accuracy(loader, model, device):
    """计算数据集的分类精度"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def train(model, train_loader, test_loader, optimizer, criterion, device, epochs):
    """训练模型并记录损失和精度"""
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        with tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch"
        ) as progress_bar:
            for images, labels in progress_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                optimizer.zero_grad()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

        train_losses.append(running_loss / len(train_loader))
        train_acc = compute_accuracy(train_loader, model, device)
        train_accuracies.append(train_acc)
        test_acc = compute_accuracy(test_loader, model, device)
        test_accuracies.append(test_acc)

        print(
            f"Epoch {epoch+1}/{epochs}, Train Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}"
        )

    plot_loss_and_accuracy(train_losses, train_accuracies, test_accuracies)


def plot_loss_and_accuracy(train_losses, train_accuracies, test_accuracies):
    """绘制损失和精度曲线"""
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="best")

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(test_accuracies, label="Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")

    plt.tight_layout()
    plt.show()


def visualize_misclassified_samples(model, test_loader, device, num_samples=10):
    """可视化测试集中识别错误的样本"""
    model.eval()
    misclassified_images = []
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # 找到预测错误的样本
            misclassified_idx = (predicted != labels).nonzero(as_tuple=True)[0]
            for idx in misclassified_idx:
                misclassified_images.append(images[idx].cpu().numpy())
                true_labels.append(labels[idx].item())
                predicted_labels.append(predicted[idx].item())

                if len(misclassified_images) >= num_samples:
                    break
            if len(misclassified_images) >= num_samples:
                break

    plot_misclassified_images(
        misclassified_images, true_labels, predicted_labels, num_samples
    )


def plot_misclassified_images(images, true_labels, predicted_labels, num_samples):
    """绘制识别错误样本的图像"""
    plt.figure(figsize=(12, 6))
    for i in range(num_samples):
        plt.subplot(2, 5, i + 1)
        image = images[i].squeeze()
        plt.imshow(image, cmap="gray")
        plt.title(f"True: {true_labels[i]}\nPred: {predicted_labels[i]}", fontsize=10)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# 主程序入口
if train_flag:
    train(model, train_loader, test_loader, optimizer, criterion, device, epochs)
    print("训练完成")
    torch.save(model.state_dict(), "LeNet.pth")
    print("模型保存完成")
else:
    model.load_state_dict(torch.load("LeNet.pth"))
    print("模型加载完成")

visualize_misclassified_samples(model, test_loader, device)
