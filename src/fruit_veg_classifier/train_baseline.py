from pathlib import Path
import csv
import time

import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader
from mlflow.tracking import MlflowClient
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
MLFLOW_ARTIFACT_DIR = PROJECT_ROOT / "mlartifacts"
IMAGE_SIZE = 100
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001
USE_DATA_AUGMENTATION = True
EXPERIMENT_NAME = "fruit-veg-classifier"
MODEL_NAME = "simple_cnn_baseline"
MODEL_VERSION = "simple_cnn_v1"
DATASET_VERSION = "fruits360_10class_v1"
DATASET_DVC_HASH = "368e305e991f8cbb42a39e501c5cf9d4.dir"
MODEL_DVC_HASH = "9986963301918306a9666808438a114c"


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def build_dataloaders() -> tuple[DataLoader, DataLoader, list[str]]:
    train_transforms = [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    ]

    if USE_DATA_AUGMENTATION:
        train_transforms.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
            ]
        )

    train_transforms.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    train_dataset = datasets.ImageFolder(
        DATA_DIR / "train", transform=transforms.Compose(train_transforms)
    )
    val_dataset = datasets.ImageFolder(DATA_DIR / "val", transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, train_dataset.classes


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def save_checkpoint(
    model: nn.Module,
    class_names: list[str],
    val_accuracy: float,
    epoch: int,
    model_path: Path,
) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "use_data_augmentation": USE_DATA_AUGMENTATION,
        "best_epoch": epoch,
        "val_accuracy": val_accuracy,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "dataset_version": DATASET_VERSION,
        "dataset_dvc_hash": DATASET_DVC_HASH,
        "model_dvc_hash": MODEL_DVC_HASH,
    }
    torch.save(checkpoint, model_path)


def save_training_history(history: list[dict[str, float]]) -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    history_path = METRICS_DIR / "baseline_training_history.csv"
    fieldnames = ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"]

    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def configure_mlflow() -> None:
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        client.create_experiment(
            EXPERIMENT_NAME,
            artifact_location=MLFLOW_ARTIFACT_DIR.as_uri(),
        )
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_mlflow_setup(
    class_names: list[str],
    train_image_count: int,
    val_image_count: int,
    device: torch.device,
) -> None:
    mlflow.log_params(
        {
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "dataset_version": DATASET_VERSION,
            "dataset_dvc_hash": DATASET_DVC_HASH,
            "model_dvc_hash": MODEL_DVC_HASH,
            "image_size": IMAGE_SIZE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "optimizer": "Adam",
            "loss_function": "CrossEntropyLoss",
            "use_data_augmentation": USE_DATA_AUGMENTATION,
            "num_classes": len(class_names),
            "classes": ",".join(class_names),
            "train_images": train_image_count,
            "val_images": val_image_count,
            "device": str(device),
        }
    )


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, class_names = build_dataloaders()
    configure_mlflow()

    model = SimpleCNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Device: {device}")
    print(f"Classes: {class_names}")
    print(f"Train images: {len(train_loader.dataset)}")
    print(f"Validation images: {len(val_loader.dataset)}")

    best_val_accuracy = 0.0
    history = []
    start_time = time.time()

    with mlflow.start_run(run_name="simple-cnn-augmentation"):
        mlflow.set_tags(
            {
                "model_family": "cnn",
                "model_version": MODEL_VERSION,
                "dataset_version": DATASET_VERSION,
                "artifact_location": str(MLFLOW_ARTIFACT_DIR),
            }
        )
        log_mlflow_setup(
            class_names,
            len(train_loader.dataset),
            len(val_loader.dataset),
            device,
        )

        for epoch in range(EPOCHS):
            train_loss, train_accuracy = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)

            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}"
            )
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                }
            )
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                },
                step=epoch + 1,
            )

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                save_checkpoint(
                    model,
                    class_names,
                    best_val_accuracy,
                    epoch + 1,
                    MODEL_DIR / "simple_cnn_baseline.pt",
                )

        elapsed_minutes = (time.time() - start_time) / 60
        save_training_history(history)
        mlflow.log_metric("best_val_accuracy", best_val_accuracy)
        mlflow.log_metric("training_time_minutes", elapsed_minutes)
        mlflow.log_artifact(METRICS_DIR / "baseline_training_history.csv")
        mlflow.log_artifact(MODEL_DIR / "simple_cnn_baseline.pt")

        print(f"Best validation accuracy: {best_val_accuracy:.4f}")
        print(f"Saved model: {MODEL_DIR / 'simple_cnn_baseline.pt'}")
        print(f"Saved history: {METRICS_DIR / 'baseline_training_history.csv'}")
        print(f"MLflow run: {mlflow.active_run().info.run_id}")
        print(f"Training time: {elapsed_minutes:.2f} minutes")


if __name__ == "__main__":
    main()
