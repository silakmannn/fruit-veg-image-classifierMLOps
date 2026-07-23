# Fruit & Vegetable Image Classifier

A beginner-friendly 10-class image classification project for learning AI and MLOps step by step.

## Goal

Build an image classifier that can recognize 10 fruit and vegetable classes, then gradually add practical MLOps components such as experiment tracking, model versioning, testing, API deployment, containerization, CI/CD, and monitoring.

## Current Status

Step 8 is complete: dataset and model artifacts are versioned with DVC.

## Project Structure

```text
imgclassifier/
|-- configs/        # Training and app configuration files
|-- data/           # Local datasets; raw image files are not committed
|-- docs/           # Learning notes and project documentation
|-- models/         # Trained model artifacts; large files are not committed
|-- notebooks/      # Exploration notebooks
|-- reports/        # Metrics, plots, and evaluation outputs
|-- requirements.txt
|-- src/            # Python package source code
|-- tests/          # Automated tests
|-- .gitignore
`-- README.md
```

## Environment Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Roadmap

1. Create the repository and project structure.
2. Set up the Python environment.
3. Define the 10-class dataset.
4. Build a baseline image classification model.
5. Evaluate the model.
6. Improve the model.
7. Add experiment tracking.
8. Add model versioning.
9. Create an inference script.
10. Build a simple FastAPI API.
11. Add basic testing.
12. Containerize the project.
13. Add CI/CD basics.
14. Deploy the model.
15. Add basic monitoring.

## Next Step

Step 9: Create an inference script for single-image predictions.

## Dataset

The first dataset version uses these 10 classes:

```text
apple
banana
orange
grape
strawberry
tomato
potato
carrot
cucumber
bell_pepper
```

Images should be placed under `data/raw/train`, `data/raw/val`, and `data/raw/test`. See `data/README.md` and `docs/dataset_plan.md` for details.

## Baseline Training

Run the baseline training script:

```powershell
.\.venv\Scripts\python.exe -m src.fruit_veg_classifier.train_baseline
```

The script trains a small CNN with PyTorch and saves the best checkpoint locally:

```text
models/simple_cnn_baseline.pt
```

Model files are ignored by Git.

During training, simple data augmentation is applied only to training images:

- random horizontal flip
- small random rotation

Validation and test images are not augmented.

Training history is saved locally:

```text
reports/metrics/baseline_training_history.csv
```

Training also logs an MLflow experiment to local SQLite storage in `mlflow.db`. MLflow tracks parameters, metrics, and artifacts for each run.

Open the MLflow UI:

```powershell
.\.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///mlflow.db
```

Then visit:

```text
http://127.0.0.1:5000
```

See `docs/mlflow_tracking.md` for details.

## Versioning

Dataset and model artifacts are tracked with DVC:

```text
dataset_version = fruits360_10class_v1
model_version = simple_cnn_v1
```

Git tracks DVC pointer files, while DVC tracks the actual image/model artifacts. See `docs/versioning.md` for details.

## Baseline Evaluation

Run the baseline evaluation script:

```powershell
.\.venv\Scripts\python.exe -m src.fruit_veg_classifier.evaluate_baseline
```

The script evaluates the saved model on `data/raw/test` and saves local reports:

```text
reports/metrics/baseline_test_summary.csv
reports/metrics/baseline_test_predictions.csv
reports/figures/baseline_confusion_matrix.png
```
