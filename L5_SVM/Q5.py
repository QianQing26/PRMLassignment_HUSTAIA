import SVM
import numpy as np
import matplotlib.pyplot as plt

# data
data = np.array(
    [
        # 中国城市 (+1)
        [31.2304, 121.4737, 1],  # 上海
        [39.9042, 116.4074, 1],  # 北京
        [22.5431, 114.0579, 1],  # 深圳
        [23.1291, 113.2644, 1],  # 广州
        [30.5728, 104.0668, 1],  # 成都
        [24.4798, 118.0895, 1],  # 厦门
        [29.5630, 106.5516, 1],  # 重庆
        [26.0722, 119.2965, 1],  # 福州
        # 日本城市 (-1)
        [35.6895, 139.6917, -1],  # 东京
        [34.6937, 135.5023, -1],  # 大阪
        [35.0116, 135.7681, -1],  # 京都
        [43.0621, 141.3544, -1],  # 札幌
        [33.5904, 130.4017, -1],  # 福冈
        [35.4437, 139.6380, -1],  # 横滨
        [34.6851, 135.8050, -1],  # 奈良
        [32.7503, 129.8777, -1],  # 长崎
        # 加入一些中国内陆城市
        [39.90, 116.40, 1],  # 北京
        [34.34, 10.94, 1],  # 西安
        [34.75, 113.62, 1],  # 郑州
        [30.59, 114.30, 1],  # 武汉
    ]
)

# data前两列为X
X = data[:, :2]
Y = data[:, 2]

model = SVM.KernelSVM()
model.fit(X, Y)
model.visualize(X, Y, show=True)
x = np.array([[25.446, 123.284]])
y = model.predict(x)
print("\n")
print(f"支撑向量为 {model.SupportVector}")
if y[0] == 1:
    print("钓鱼岛是中国的")
else:
    print("钓鱼岛是日本的")
