import torch
import typer
import matplotlib.pyplot as plt
from loguru import logger

from stuperml.data import MyDataset
from stuperml.model import SimpleMLP
from configs import data_config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def evaluate(model_checkpoint: str) -> None:
    print("Evaluating like my life depended on it")
    print(model_checkpoint)

    _, _, test_set = MyDataset(cfg=data_config).load_data()
    n_features = test_set.tensors[0].shape[1]

    model = SimpleMLP(input_size=n_features).to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))

    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)

    model.eval()
    total_mae, total_samples = 0, 0
    all_residuals = []

    with torch.no_grad():
        for features, target in test_dataloader:
            features, target = features.to(DEVICE), target.to(DEVICE)
            target = target.view(-1, 1).float()

            y_pred = model(features)

            total_mae += torch.abs(y_pred - target).sum().item()
            total_samples += target.size(0)

            residuals = (target - y_pred).cpu().numpy()
            all_residuals.extend(residuals.flatten())

    print(f"Test MAE: {total_mae / total_samples:.2f}")

    plt.figure(figsize=(10, 6))
    plt.hist(all_residuals, bins=30)
    plt.xlabel("Error Magnitude")
    plt.ylabel("Count")
    plt.savefig("src/stuperml/figures/residual_distribution.png")


if __name__ == "__main__":
    # evaluate('models/model.pth') # For testing
    typer.run(evaluate)
