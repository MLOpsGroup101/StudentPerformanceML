import torch
import torch.nn as nn
from torch import Tensor


class SimpleMLP(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MeanBaseModel(nn.Module):
    mean_value: Tensor

    def __init__(self, target_tensor: Tensor):
        super().__init__()
        self.register_buffer("mean_value", torch.mean(target_tensor.float()))

    def forward(self, x: Tensor) -> Tensor:
        return self.mean_value.expand(x.size(0), 1)


if __name__ == "__main__":
    x = torch.rand(1)
    model = SimpleMLP(input_size=x.shape[0])
    print(f"Output shape of model: {model(x).shape}")
