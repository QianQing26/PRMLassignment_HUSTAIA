import torch
import torch.nn as nn
import torch.optim as optim


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation_fn):
        super(NeuralNetwork, self).__init__()
        layers = []
        prev_size = input_size
        for size in hidden_size:
            layers.append(nn.Linear(prev_size, size))
            if activation_fn == "relu":
                layers.append(nn.ReLU())
            elif activation_fn == "tanh":
                layers.append(nn.Tanh())
            elif activation_fn == "sigmoid":
                layers.append(nn.Sigmoid())
            elif activation_fn == "leaky_relu":
                layers.append(nn.LeakyReLU())
            prev_size = size
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def predict(self, X):
        with torch.no_grad():
            output = self.forward(X)
            _, predicted = torch.max(output.data, 1)
            return predicted


if __name__ == "__main__":
    model = NeuralNetwork(4, [3, 4], 3, "leaky_relu")
    print(model)
