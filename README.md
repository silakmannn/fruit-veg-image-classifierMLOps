# Fruit & Vegetable Image Classifier

A beginner-friendly 10-class image classification project for learning AI and MLOps step by step.

## Goal

Build an image classifier that can recognize 10 fruit and vegetable classes, then gradually add practical MLOps components such as experiment tracking, model versioning, testing, API deployment, containerization, CI/CD, and monitoring.

## Current Status

Step 2 is complete: Python environment setup.

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

Step 3: Define the 10-class dataset.
