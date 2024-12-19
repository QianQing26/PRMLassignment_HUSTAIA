import numpy as np
import matplotlib.pyplot as plt
from optimizer import GradientDescent, Momentum, AdaGrad, RMSprop, Adam, Nesterov

np.random.seed(114514)


def f(x):
    return x * np.cos(0.25 * np.pi * x)


def df(x):
    return np.cos(0.25 * np.pi * x) - 0.25 * np.pi * x * np.sin(0.25 * np.pi * x)


x0 = -4
max_iter = 500
learning_rate = 0.4
optimizers = [
    GradientDescent(learning_rate=learning_rate),
    Momentum(learning_rate=learning_rate, momentum=0.9),
    Nesterov(learning_rate=learning_rate, momentum=0.9),
    AdaGrad(learning_rate=learning_rate, epsilon=1e-6),
    RMSprop(learning_rate=learning_rate, alpha=0.9),
    Adam(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-6),
]

# 配置子图
fig, axs = plt.subplots(3, 2, figsize=(12, 12))
axs = axs.flatten()

x_range = np.linspace(-5, 5, 500)
y_range = f(x_range)

for idx, opt in enumerate(optimizers):
    x = x0
    params = {"x": x}
    x_history = [x]
    y_history = [f(x)]

    for i in range(max_iter):
        opt.update(params, {"x": df(params["x"])})
        x_history.append(params["x"])
        y_history.append(f(params["x"]))

    ax = axs[idx]
    ax.plot(
        x_range,
        y_range,
        label="Original Function",
        color="gray",
        linestyle="--",
        linewidth=1.5,
    )

    ax.plot(
        x_history,
        y_history,
        marker="o",
        label=f"{opt.__class__.__name__}",
        color=f"C{idx}",
    )

    ax.set_title(
        f"{opt.__class__.__name__}   result: x={x_history[-1]:.3f}, f(x)={y_history[-1]:.3f}",
        fontsize=12,
    )
    ax.set_xlabel("x value", fontsize=12)
    ax.set_ylabel("f(x) value", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best", fontsize=10)
# 去掉多余的子图
for ax in axs[len(optimizers) :]:
    fig.delaxes(ax)

plt.tight_layout()
plt.show()
