import numpy as np
import matplotlib.pyplot as plt
from optimizer import GradientDescent, Momentum, AdaGrad, RMSprop, Adam

np.random.seed(114514)


def f(x):
    return x * np.cos(0.25 * np.pi * x)
    # return x**2


def df(x):
    return np.cos(0.25 * np.pi * x) - 0.25 * np.pi * x * np.sin(0.25 * np.pi * x)
    # return 2 * x


x0 = -4
max_iter = 10
optim = [
    GradientDescent(learning_rate=0.4),
    Momentum(learning_rate=0.4, momentum=0.9),
    AdaGrad(learning_rate=0.4, epsilon=1e-6),
    RMSprop(learning_rate=0.4, alpha=0.9),
    Adam(learning_rate=0.4, beta1=0.9, beta2=0.999, epsilon=1e-6),
]
for opt in optim:
    x = x0
    params = {"x": x}
    print(f"\nUsing optimizer: {opt.__class__.__name__}")
    x_history = [x]
    y_history = [f(x)]
    for i in range(max_iter):
        opt.update(params, {"x": df(params["x"])})
        x_history.append(params["x"])
        y_history.append(f(params["x"]))
    plt.plot(x_history, y_history, label=opt.__class__.__name__)
plt.legend()
plt.show()
