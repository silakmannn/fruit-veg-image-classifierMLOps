from pathlib import Path
import csv

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.fruit_veg_classifier.train_baseline import SimpleCNN


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_PATH = PROJECT_ROOT / "models" / "simple_cnn_baseline.pt"
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
BATCH_SIZE = 32


def build_test_loader(image_size: int) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    test_dataset = datasets.ImageFolder(DATA_DIR / "test", transform=transform)
    return DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


def save_predictions(rows: list[dict[str, str | float]]) -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = METRICS_DIR / "baseline_test_predictions.csv"
    fieldnames = ["image_path", "true_label", "predicted_label", "confidence"]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_summary(test_accuracy: float, total_images: int) -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = METRICS_DIR / "baseline_test_summary.csv"

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerow({"metric": "test_accuracy", "value": test_accuracy})
        writer.writerow({"metric": "test_images", "value": total_images})


def save_confusion_matrix(
    true_labels: list[int],
    predicted_labels: list[int],
    class_names: list[str],
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(true_labels, predicted_labels)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_names)
    display.plot(xticks_rotation=45, cmap="Blues", values_format="d")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "baseline_confusion_matrix.png")
    plt.close()


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    class_names = checkpoint["class_names"]
    image_size = checkpoint["image_size"]

    test_loader = build_test_loader(image_size)
    model = SimpleCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_true_labels = []
    all_predicted_labels = []
    prediction_rows = []

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probabilities = nn.functional.softmax(outputs, dim=1)
            confidences, predictions = probabilities.max(dim=1)

            all_true_labels.extend(labels.cpu().tolist())
            all_predicted_labels.extend(predictions.cpu().tolist())

            start = batch_index * test_loader.batch_size
            batch_paths = [
                path for path, _ in test_loader.dataset.samples[start : start + len(labels)]
            ]
            for image_path, true_label, predicted_label, confidence in zip(
                batch_paths,
                labels.cpu().tolist(),
                predictions.cpu().tolist(),
                confidences.cpu().tolist(),
            ):
                prediction_rows.append(
                    {
                        "image_path": str(Path(image_path).relative_to(PROJECT_ROOT)),
                        "true_label": class_names[true_label],
                        "predicted_label": class_names[predicted_label],
                        "confidence": confidence,
                    }
                )

    test_accuracy = accuracy_score(all_true_labels, all_predicted_labels)
    save_predictions(prediction_rows)
    save_summary(test_accuracy, len(all_true_labels))
    save_confusion_matrix(all_true_labels, all_predicted_labels, class_names)

    print(f"Device: {device}")
    print(f"Test images: {len(all_true_labels)}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Saved predictions: {METRICS_DIR / 'baseline_test_predictions.csv'}")
    print(f"Saved summary: {METRICS_DIR / 'baseline_test_summary.csv'}")
    print(f"Saved confusion matrix: {FIGURES_DIR / 'baseline_confusion_matrix.png'}")


if __name__ == "__main__":
    main()
