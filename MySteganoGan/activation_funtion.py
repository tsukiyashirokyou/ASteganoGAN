import torch
from torch import nn


class ACON_C(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.p1 = nn.Parameter(torch.randn(1, width, 1, 1))
        self.p2 = nn.Parameter(torch.randn(1, width, 1, 1))
        self.beta = nn.Parameter(torch.ones(1, width, 1, 1))

    def forward(self, x):
        delta_p = self.p1 - self.p2
        sigmoid_input = self.beta * delta_p * x
        return delta_p * x * torch.sigmoid(sigmoid_input) + self.p2 * x
