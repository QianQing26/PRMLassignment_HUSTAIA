import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from LeNet import LeNet
from sklearn.metrics import accuracy_score
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

# 设置超参数
learning_rate = 0.01
epochs = 10
batch_size = 256
train_flag = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{device}")
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


def train(model, train_loader, optimizer, criterion, device, epochs):
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
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="best")

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Acc")
    plt.plot(test_accuracies, label="Test Acc")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")

    plt.tight_layout()
    plt.show()


# 可视化识别错误的样本
def visualize_misclassified_samples(model, test_loader, device, num_samples=10):
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

                # 如果已经找够了指定数量的样本，退出循环
                if len(misclassified_images) >= num_samples:
                    break
            if len(misclassified_images) >= num_samples:
                break

    # 可视化这些样本
    plot_misclassified_images(
        misclassified_images, true_labels, predicted_labels, num_samples
    )


def plot_misclassified_images(images, true_labels, predicted_labels, num_samples):
    plt.figure(figsize=(12, 6))
    for i in range(num_samples):
        plt.subplot(2, 5, i + 1)
        image = images[i].squeeze()  # 移除单通道维度
        plt.imshow(image, cmap="gray")
        plt.title(f"True: {true_labels[i]}\nPred: {predicted_labels[i]}", fontsize=10)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def visualize_samples(model, test_loader, device, num_samples=10):
    model.eval()
    correct_images = []
    correct_labels = []
    misclassified_images = []
    misclassified_true_labels = []
    misclassified_predicted_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            for idx in range(len(images)):
                if predicted[idx] == labels[idx]:
                    if len(correct_images) < num_samples:
                        correct_images.append(images[idx].cpu().numpy())
                        correct_labels.append(labels[idx].item())
                else:
                    if len(misclassified_images) < num_samples:
                        misclassified_images.append(images[idx].cpu().numpy())
                        misclassified_true_labels.append(labels[idx].item())
                        misclassified_predicted_labels.append(predicted[idx].item())

                if (
                    len(correct_images) >= num_samples
                    and len(misclassified_images) >= num_samples
                ):
                    break

            if (
                len(correct_images) >= num_samples
                and len(misclassified_images) >= num_samples
            ):
                break

    # 可视化这些样本
    plot_correct_samples(correct_images, correct_labels, num_samples)
    plot_misclassified_samples(
        misclassified_images,
        misclassified_true_labels,
        misclassified_predicted_labels,
        num_samples,
    )


def plot_correct_samples(correct_images, correct_labels, num_samples):
    plt.figure(figsize=(15, 8))

    # 绘制正确预测的样本
    for i in range(num_samples):
        plt.subplot(2, 5, i + 1)
        image = correct_images[i].squeeze()
        plt.imshow(image, cmap="gray")
        plt.title(f"Correct: {correct_labels[i]}", fontsize=12)
        plt.axis("off")
    plt.suptitle("Correctly Predicted", fontsize=15)
    plt.tight_layout()
    plt.show()


def plot_misclassified_samples(
    misclassified_images,
    misclassified_true_labels,
    misclassified_predicted_labels,
    num_samples,
):
    plt.figure(figsize=(15, 8))

    # 绘制错误预测的样本
    for i in range(num_samples):
        plt.subplot(2, 5, i + 1)
        image = misclassified_images[i].squeeze()
        plt.imshow(image, cmap="gray")
        plt.title(
            f"True: {misclassified_true_labels[i]}    Pred: {misclassified_predicted_labels[i]}",
            fontsize=12,
        )
        plt.axis("off")
    plt.suptitle("Misclassified", fontsize=15)
    plt.tight_layout()
    plt.show()


if train_flag:
    train(model, train_loader, optimizer, criterion, device, epochs)
    print("训练完成")
    # 保存模型
    torch.save(model.state_dict(), "LeNet.pth")
    print("模型保存完成")
else:
    # 加载模型
    model.load_state_dict(torch.load("LeNet.pth"))
    print("模型加载完成")


model.eval()
# 在测试集上预测
y_pred = []
y_true = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.tolist())
        y_true.extend(labels.tolist())

acc = accuracy_score(y_true, y_pred)
print(f"测试集准确率：{acc:.4f}")
# 绘制混淆矩阵
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)
print(cm)
plt.imshow(cm, cmap="gray")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
# 可视化识别错误的样本
# visualize_misclassified_samples(model, test_loader, device)

visualize_samples(model, test_loader, device, num_samples=10)
