import os
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import typer
import wandb
from google.cloud import storage

from configs import data_config
from stuperml.data import MyDataset
from stuperml.model import SimpleMLP

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
MODEL_DIR = Path("models")


def _parse_gcs_uri(uri: str) -> tuple[str, str]:
    """Parse a GCS URI into bucket name and object prefix."""
    bucket_path = uri[5:] if uri.startswith("gs://") else uri
    parts = bucket_path.split("/", 1)
    if len(parts) != 2 or not parts[1]:
        raise ValueError("Model GCS URI must include bucket and prefix, e.g. gs://bucket/models")
    return parts[0], parts[1].rstrip("/")


def _upload_model_artifacts(local_model_path: Path, gcs_models_uri: str, timestamp: str) -> None:
    """Upload the timestamped model artifact to GCS."""
    bucket_name, prefix = _parse_gcs_uri(gcs_models_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    stamped_blob = bucket.blob(f"{prefix}/model_{timestamp}.pth")
    stamped_blob.upload_from_filename(local_model_path.as_posix())


def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 30, verbose: bool = False) -> None:
    """Train the model and persist artifacts."""
    print("Training day and night")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    wandb_entity = os.getenv("WANDB_ENTITY", None)
    wandb.init(
        project="StuPerML",
        entity=wandb_entity,
        config={"learning_rate": lr, "batch_size": batch_size, "epochs": epochs, "device": str(DEVICE)},
        mode="online",
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Run ID: {timestamp}")

    print(f"{lr=}, {batch_size=}, {epochs=}")

    train_set, val_set, _ = MyDataset(cfg=data_config).load_data()
    n_features = train_set.tensors[0].shape[1]

    model = SimpleMLP(input_size=n_features).to(DEVICE)
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    statistics = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        for index, (features, target) in enumerate(train_dataloader):
            features, target = features.to(DEVICE), target.to(DEVICE)
            target = target.view(-1, 1).float()

            optimizer.zero_grad()
            y_pred = model(features)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            if verbose:
                if index % 10 == 0:
                    print(f"Epoch {epoch}, iter {index}, \t train_loss: {loss.item():.5f}")

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for features, target in val_dataloader:
                features, target = features.to(DEVICE), target.to(DEVICE)
                target = target.view(-1, 1).float()
                y_pred = model(features)
                v_loss = loss_fn(y_pred, target)
                epoch_val_loss += v_loss.item()

        avg_train = epoch_train_loss / len(train_dataloader)
        avg_val = epoch_val_loss / len(val_dataloader)
        statistics["train_loss"].append(avg_train)
        statistics["val_loss"].append(avg_val)

        print(f"Epoch {epoch} \t Summary: Train Loss: {avg_train:.5f}, \t Val Loss: {avg_val:.5f}")
        wandb.log({"train_loss": avg_train, "val_loss": avg_val, "epoch": epoch})

    print("Training complete")
    wandb.finish()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"model_{timestamp}.pth"
    torch.save(model.state_dict(), model_path)
    print("Saved locally.")

    gcs_models_uri = os.getenv("AIP_MODEL_DIR") or data_config.gcs_models_uri
    if gcs_models_uri:
        if (
            data_config.gcs_service_account_key
            and "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ
            and os.path.exists(data_config.gcs_service_account_key)
        ):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = data_config.gcs_service_account_key
        _upload_model_artifacts(model_path, gcs_models_uri, timestamp)

    plt.figure(figsize=(10, 5))
    plt.plot(statistics["train_loss"], label="Train Loss")
    plt.plot(statistics["val_loss"], label="Val Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("src/stuperml/figures/training_validation_epoch_error.png")


if __name__ == "__main__":
    typer.run(train)
