# MLflow Experiment Tracking

## Why MLflow

MLflow lets us save each training experiment in a structured way. Instead of only reading terminal output, we can compare runs later in a browser UI.

For this beginner project, MLflow stores local experiment metadata in:

```text
mlflow.db
```

Run artifacts are stored locally by MLflow. Local MLflow outputs are ignored by Git because they are generated experiment artifacts.

## What We Track

Each training run logs:

- parameters: image size, batch size, epochs, learning rate, optimizer, loss function, augmentation setting
- metrics per epoch: train loss, train accuracy, validation loss, validation accuracy
- final metrics: best validation accuracy and training time
- artifacts: training history CSV and saved PyTorch checkpoint

## Commands

Install the new dependency:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run training:

```powershell
.\.venv\Scripts\python.exe -m src.fruit_veg_classifier.train_baseline
```

Open the MLflow UI:

```powershell
.\.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///mlflow.db
```

Then open this address in your browser:

```text
http://127.0.0.1:5000
```
