---
title: Student Performance Machine Learning
---

# Student Performance ML

Welcome to the documentation for **Student Performance ML (stuperml)**.

This project analyzes how AI-assisted learning influences student outcomes and provides a reproducible machine learning pipeline for predicting students' final scores. It is built with PyTorch and follows MLOps best practices such as structured configuration, reproducible environments, testing, and containerization.

## What this project does

- Downloads and preprocesses a public dataset on AI usage and student performance.
- Trains a neural network to predict final exam scores from demographic, study habit, and AI usage features.
- Evaluates the model against a simple baseline.
- Exposes a FastAPI-based HTTP service to serve predictions.

## Documentation structure

- **Home**: High-level overview of the project (this page).
- **Usage**: How to set up the environment, run preprocessing, train the model, evaluate it, and start the API.
- **Data**: Details about the dataset, preprocessing steps, and generated artifacts.
- **Models**: Description of the `SimpleMLP` model, the baseline model, and the training/evaluation procedures.
- **API Reference**: HTTP endpoints and auto-generated Python API documentation powered by mkdocstrings.

## Getting started quickly

From the project root:

1. Sync dependencies and download the dataset:

	```bash
	uv run invoke sync
	```

2. Run preprocessing and generate data artifacts:

	```bash
	uv run src/stuperml/data.py
	```

3. Train the model:

	```bash
	uv run src/stuperml/train.py
	```

4. Start the prediction API:

	```bash
	uv run uvicorn src.stuperml.api:app --reload
	```

For more details, see the **Usage** page.
