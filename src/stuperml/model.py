import torch
import torch.nn as nn
from torch import Tensor

from stuperml.logger import setup_logger
from luguru import logger

setup_logger


class SimpleMLP(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        logger.debug(f"Initializing SimpleMLP with input_size={input_size}")
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        logger.info("SimpleMLP initialized successfully.")

    def forward(self, x: Tensor) -> Tensor:
        logger.debug(f"Forward pass with input shape: {x.shape}")
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        if x.isnan().any():
            logger.warning("NaN values detected in model output.")
        return x


class MeanBaseModel(nn.Module):
    mean_value: Tensor

    def __init__(self, target_tensor: Tensor):
        super().__init__()
        # self.register_buffer("mean_value", torch.mean(target_tensor.float()))
        if target_tensor.numel() == 0:
            logger.error("Empty target tensor provided to MeanBaseModel.")
            raise ValueError("Target tensor cannot be empty.")
        mean_value = torch.mean(target_tensor.float())
        if mean_value.isnan():
            logger.critical("Computed mean is NaN - this indicates a serious data issue")
        self.register_buffer("mean_value", mean_value)
        logger.info(f"MeanBaseModel initialized with mean value: {mean_value.item():.4f}")

    def forward(self, x: Tensor) -> Tensor:
        return self.mean_value.expand(x.size(0), 1)


if __name__ == "__main__":
    x = torch.rand(1)
    model = SimpleMLP(input_size=x.shape[0])
    # print(f"Output shape of model: {model(x).shape}")
    output = model(x)
    logger.info(f"Output shape of model: {output.shape}")
