"""Neural network architecture for option pricing (Section 3.1, Table 2 of the article).

Architecture: MLP with 4 hidden layers x 400 neurons, ReLU activation,
Glorot uniform initialization, no dropout, no batch normalization.
"""

import torch.nn as nn


class PricingMLP(nn.Module):
    """Multi-layer perceptron for option pricing / implied volatility.

    Default: 4 hidden layers x 400 neurons (Table 2 of the article).
    """

    def __init__(self, input_dim, hidden_dim=400, n_hidden=4, output_dim=1):
        super().__init__()

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(n_hidden - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.network(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
