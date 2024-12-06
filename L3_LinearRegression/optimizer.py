import numpy as np


class GradientDescent:
    def __init__(self, learning_rate=0.01):
        """_summary_

        Args:
            learning_rate (float, optional): Learning rate. Defaults to 0.01.
        """
        self.learning_rate = learning_rate

    def update(self, params, grads):
        """_summary_

        Args:
            params (_type_): parameters to be updated
            grads (_type_): gradients of parameters to be updated
        """
        for key in params.keys():
            params[key] -= self.learning_rate * grads[key]


class Momentum:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        """_summary_

        Args:
            learning_rate (float, optional): _description_. Defaults to 0.01.
            momentum (float, optional): _description_. Defaults to 0.9.
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None

    def update(self, params, grads):
        """_summary_

        Args:
            params (_type_): _description_
            grads (_type_): _description_
        """
        if self.velocity is None:
            self.velocity = {}
            for key, val in params.items():
                self.velocity[key] = (
                    np.zeros_like(val) if isinstance(val, np.ndarray) else 0.0
                )

        for key in params.keys():
            self.velocity[key] = (
                self.momentum * self.velocity[key] - self.learning_rate * grads[key]
            )
            params[key] += self.velocity[key]


class AdaGrad:
    def __init__(self, learning_rate=0.01, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.espsilon = epsilon
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val) if isinstance(val, np.ndarray) else 0.0

        for key in params.keys():
            self.h[key] += grads[key] ** 2
            params[key] -= (
                self.learning_rate * grads[key] / (self.h[key] + self.espsilon) ** 0.5
            )


class RMSprop:
    def __init__(self, learning_rate=0.01, alpha=0.9, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.epsilon = epsilon
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val) if isinstance(val, np.ndarray) else 0.0

        for key in params.keys():
            self.h[key] = self.alpha * self.h[key] + (1 - self.alpha) * grads[key] ** 2
            params[key] -= (
                self.learning_rate * grads[key] / (self.h[key] ** 0.5 + self.epsilon)
            )


class Adam:
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, params, grads):
        if self.m is None or self.v is None or self.t == 0:
            self.m = {}
            self.v = {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val) if isinstance(val, np.ndarray) else 0.0
                self.v[key] = np.zeros_like(val) if isinstance(val, np.ndarray) else 0.0
        self.t += 1
        for key in params.keys():
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grads[key] ** 2
            m_hat = self.m[key] / (1 - self.beta1**self.t)
            v_hat = self.v[key] / (1 - self.beta2**self.t)
            params[key] -= self.learning_rate * m_hat / (v_hat + self.epsilon) ** 0.5
