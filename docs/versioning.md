# Data and Model Versioning

## Why Version Data and Models

In MLOps, a model is not meaningful by itself. We also need to know which dataset, code, parameters, and metrics produced it.

The relationship is:

```text
dataset version + code version + training parameters -> training run -> model version
```

This is called lineage.

## Current Versions

| Item | Version | DVC pointer | DVC hash |
| --- | --- | --- | --- |
| Dataset | fruits360_10class_v1 | data/raw.dvc | 368e305e991f8cbb42a39e501c5cf9d4.dir |
| Model | simple_cnn_v1 | models/simple_cnn_baseline.pt.dvc | 9986963301918306a9666808438a114c |

## What Git Tracks

Git tracks code and small metadata files:

```text
data/raw.dvc
models/simple_cnn_baseline.pt.dvc
.dvc/config
```

## What DVC Tracks

DVC tracks large artifacts:

```text
data/raw
models/simple_cnn_baseline.pt
```

The current DVC remote is local:

```text
C:\Users\HP\dvc-storage\imgclassifier
```

## Pulling Artifacts Later

After cloning the Git repository, restore the dataset and model artifacts with:

```powershell
dvc pull
```

## Future Improvement

A cloud DVC remote such as Google Drive, S3, or Azure Blob can be added later so the dataset and model artifacts are available from another machine.
