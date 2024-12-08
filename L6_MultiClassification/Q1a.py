import numpy as np
from PLA import PLA
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
from OneVersusOne import OVOwithPLA

# 加载Iris数据集
iris = datasets.load_iris()
# print(iris)
X = iris.data
y = iris.target

# 随机打乱数据
idx = np.random.permutation(len(X))
X = X[idx]
y = y[idx]
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# 创建OVO分类器并训练
ovo = OVOwithPLA(X_train, y_train, learning_rate=1, max_iter=1500)
# 对测试集进行预测

# # 选择3个类别
# selected_classes = [0, 1, 2]  # 设定我们只使用setosa、versicolor和virginica
# mask = np.isin(y, selected_classes)
# X = X[mask]
# y = y[mask]

# # 随机选择30个样本作为训练集，20个样本作为测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# # 创建OVO分类器并训练
# ovo = OVO(X_train, y_train, learning_rate=1, max_iter=1500)

# # 对测试集进行预测
y_pred = ovo.predict(X_test)

# # 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"使用PLA的OVO分类器准确率：{accuracy:.4f}")
for classifier in ovo.classifiers.values():
    print(f"分类器{classifier}的权重：{classifier.w}")