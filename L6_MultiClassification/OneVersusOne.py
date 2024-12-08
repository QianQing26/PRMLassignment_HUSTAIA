import numpy as np
from PLA import PLA


class OVOwithPLA:
    def __init__(self, X_train, y_train, learning_rate=0.5, max_iter=1500):
        self.X_train = X_train
        self.y_train = y_train
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.classifiers = {}
        self.classes = np.unique(y_train)

        # 构建所有二分类器
        for i, class_1 in enumerate(self.classes):
            for class_2 in self.classes[i + 1 :]:
                idx = np.isin(y_train, [class_1, class_2])  # 筛选两类数据的索引
                X = X_train[idx]
                y = np.where(y_train[idx] == class_1, 1, -1)
                self.classifiers[(class_1, class_2)] = PLA(
                    X, y, self.learning_rate, self.max_iter
                )

    def predict(self, X):
        n_samples = X.shape[0]
        votes = np.zeros((n_samples, len(self.classes)))  # 每个样本对每个类的投票计数

        # 获取所有二分类器的预测结果
        for (class_1, class_2), classifier in self.classifiers.items():
            y_pred = classifier.predict(X)
            votes[:, class_1] += y_pred == 1
            votes[:, class_2] += y_pred == -1

        # 返回投票数最多的类别
        return self.classes[np.argmax(votes, axis=1)]

    # def plot_decision_boundary(self, X, y, ax=None):
    #     # 设置绘图风格
    #     sns.set_style("whitegrid")
    #     if ax is None:
    #         ax = plt.gca()

    #     # 设置特征边界
    #     x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    #     y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

    #     # 绘制数据点
    #     for cls in self.classes:
    #         sns.scatterplot(
    #             x=X[y == cls, 0],
    #             y=X[y == cls, 1],
    #             label=f"Class {cls}",
    #             ax=ax,
    #             s=60,
    #             alpha=0.9,
    #             edgecolor="k",
    #         )

    #     # 绘制每个二分类器的决策边界
    #     for (class_1, class_2), classifier in self.classifiers.items():
    #         x_vals = np.linspace(x_min, x_max, 500)
    #         y_vals = (-classifier.w[0] * x_vals - classifier.b) / classifier.w[1]
    #         ax.plot(
    #             x_vals,
    #             y_vals,
    #             label=f"{class_1} vs {class_2}",
    #             linewidth=2,
    #         )

    #     # 设置图形标签
    #     ax.set_xlabel("Feature 1", fontsize=12)
    #     ax.set_ylabel("Feature 2", fontsize=12)
    #     ax.set_title("Decision Boundary", fontsize=14)
    #     ax.legend(fontsize=10, loc="best", frameon=True)
    #     plt.tight_layout()
    #     plt.show()
