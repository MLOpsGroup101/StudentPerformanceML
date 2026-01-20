import torch
import typer
import matplotlib.pyplot as plt
from loguru import logger

from stuperml.data import MyDataset
from stuperml.model import SimpleMLP
from stuperml.logger import setup_logger
from configs import data_config

setup_logger()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def evaluate(model_checkpoint: str) -> None:
    # print("Evaluating like my life depended on it")
    # print(model_checkpoint)

    logger.debug(f"Starting evaluation with model checkpoint: {model_checkpoint}")
    logger.info("Evaluating model...")

    _, _, test_set = MyDataset(cfg=data_config).load_data()
    n_features = test_set.tensors[0].shape[1]
    logger.debug(f"Test set has {len(test_set)} samples and {n_features} features.")

    model = SimpleMLP(input_size=n_features).to(DEVICE)

    try:
        model.load_state_dict(torch.load(model_checkpoint))
        logger.info("Model checkpoint loaded successfully")
    except FileNotFoundError:
        logger.error(f"Model checkpoint not found: {model_checkpoint}")
        raise
    except Exception as e:
        logger.critical(f"Critical error loading model: {e}")
        raise

    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)

    model.eval()
    total_mae, total_samples = 0, 0
    all_residuals = []

    logger.debug("Starting evaluation loop")
    with torch.no_grad():
        for batch_idx, (features, target) in enumerate(test_dataloader):
            features, target = features.to(DEVICE), target.to(DEVICE)
            target = target.view(-1, 1).float()

            y_pred = model(features)

            total_mae += torch.abs(y_pred - target).sum().item()
            total_samples += target.size(0)

            residuals = (target - y_pred).cpu().numpy()
            all_residuals.extend(residuals.flatten())

            if batch_idx == 0:
                logger.debug(f"First batch sample predictions: {target.size(0)} samples")

    mae = total_mae / total_samples
    logger.info(f"Test MAE: {mae:.2f}")

    if mae > 10.0:
        logger.warning(f"High MAE detected: {mae:.2f}")

    print(f"Test MAE: {total_mae / total_samples:.2f}")

    plt.figure(figsize=(10, 6))
    plt.hist(all_residuals, bins=30)
    plt.xlabel("Error Magnitude")
    plt.ylabel("Count")
    plt.savefig("src/stuperml/figures/residual_distribution.png")
    logger.info("Residual distribution plot saved to 'src/stuperml/figures/residual_distribution.png'")


if __name__ == "__main__":
    # evaluate('models/model.pth') # For testing
    typer.run(evaluate)
