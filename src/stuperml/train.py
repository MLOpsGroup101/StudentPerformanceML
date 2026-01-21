import matplotlib.pyplot as plt
import torch
import typer
import os
from datetime import datetime
from google.cloud import storage

from configs import data_config
from stuperml.data import MyDataset
from stuperml.model import SimpleMLP

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model_dir = os.getenv("AIP_MODEL_DIR")

def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 30, verbose: bool = False) -> None:
    print("Training day and night")

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

    print("Training complete")

    model_filename = f"model_{timestamp}.pth"

    if model_dir:
        torch.save(model.state_dict(), model_filename)
        
        path_parts = model_dir.replace("gs://", "").split("/")
        bucket_name = path_parts[0]
        blob_path_prefix = "/".join(path_parts[1:])
        
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        # Upload Model
        model_blob_path = f"{blob_path_prefix}/{model_filename}"
        blob = bucket.blob(model_blob_path)
        blob.upload_from_filename(model_filename)
        print(f"Model saved to gs://{bucket_name}/{model_blob_path}")

        # 3. SAVE & UPLOAD FIGURE
        plt.figure(figsize=(10, 5))
        plt.plot(statistics["train_loss"], label="Train Loss")
        plt.plot(statistics["val_loss"], label="Val Loss")
        plt.title(f"Loss - Run {timestamp}")
        plt.legend()
        
        plot_filename = f"loss_plot_{timestamp}.png"
        plt.savefig(plot_filename)
        
        # Upload Plot
        plot_blob_path = f"{blob_path_prefix}/{plot_filename}"
        blob_plot = bucket.blob(plot_blob_path)
        blob_plot.upload_from_filename(plot_filename)
        print(f"Plot saved to gs://{bucket_name}/{plot_blob_path}")
    else:
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), f"models/model.pth")
        print("Saved locally.")

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
