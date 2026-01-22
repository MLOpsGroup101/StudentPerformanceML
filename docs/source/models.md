# Models

This page describes the baseline and neural network models implemented in the *stuperml* project, as well as how they are trained and evaluated.

## Model module (`stuperml.model`)

The core model definitions live in the `stuperml.model` module.

### `SimpleMLP`

`SimpleMLP` is a feed-forward neural network implemented using `torch.nn.Module`. It is used to predict the final student score from the preprocessed feature vector.

Architecture:

- Input layer: size equals the number of preprocessed features.
- Hidden layer 1: `Linear(input_size, 64)` + `ReLU` activation.
- Hidden layer 2: `Linear(64, 32)` + `ReLU` activation.
- Output layer: `Linear(32, 1)` (scalar regression output).

Additional behaviors:

- Logs debug information upon initialization (input size, successful construction).
- Warns via the logger if NaN values appear in the output during the forward pass.

This model is used in both training (`stuperml.train`) and evaluation (`stuperml.evaluate`), as well as by the API (`stuperml.api`).

### `MeanBaseModel`

`MeanBaseModel` is a simple baseline model for regression tasks. It learns only the mean of the target variable on the training set and always predicts this constant value for any input.

Key properties:

- Takes a target tensor at initialization and computes its mean value.
- Registers the mean as a buffer so it moves correctly across devices.
- Validates that the target tensor is non-empty and logs errors or critical issues if the mean is NaN.
- The `forward` method returns a tensor of shape `(batch_size, 1)` where every element equals the stored mean.

This baseline allows you to compare the performance of `SimpleMLP` against a naive "predict-the-mean" strategy.

## Training (`stuperml.train`)

The training script defines a `train` function that orchestrates loading data, initializing the model, and running the optimization loop.

High-level steps:

1. Determine the device: CUDA, Apple MPS (if available), or CPU.
2. Load train and validation splits via `MyDataset(cfg=data_config).load_data()`.
3. Infer the number of input features from the training tensors.
4. Instantiate a `SimpleMLP` model with the computed input size.
5. Create PyTorch `DataLoader` instances for train and validation data.
6. Use Mean Squared Error (MSE) loss and the Adam optimizer.
7. For each epoch:
   - Train the model and accumulate training loss.
   - Evaluate on the validation set and accumulate validation loss.
   - Optionally print intermediate training loss if `verbose` is enabled.
8. Track epoch-wise training and validation losses in a `statistics` dictionary.
9. Save the final model weights to `models/model.pth`.
10. Generate and save a loss curve figure to `src/stuperml/figures/training_validation_epoch_error.png`.

The `train` function is exposed via `typer`, so you can call it from the command line with configurable hyperparameters. See the **Usage** page for examples.

## Evaluation (`stuperml.evaluate`)

The evaluation script defines an `evaluate` function, also exposed via `typer`, which loads a trained model and computes metrics on the test set.

Workflow:

1. Load the test set tensors from disk via `MyDataset(cfg=data_config).load_data()`.
2. Instantiate a `SimpleMLP` model with the correct input size.
3. Load the model checkpoint from the provided path.
4. Wrap the test split in a `DataLoader` with a default batch size of 32.
5. Set the model to evaluation mode and iterate over batches without gradient computation.
6. Compute the Mean Absolute Error (MAE) across all test samples.
7. Log the MAE and emit a warning if it exceeds a threshold (10.0 by default).
8. Collect residuals (target - prediction) and plot their distribution.
9. Save the residual histogram to `src/stuperml/figures/residual_distribution.png`.

This evaluation routine provides both a scalar performance metric and a visual diagnostic of model errors.

## Figures

Training and evaluation generate the following figures under `src/stuperml/figures/`:

- `training_validation_epoch_error.png`: Training and validation loss over epochs.
- `residual_distribution.png`: Histogram of residuals on the test set.

These plots can be included in reports or inspected manually to understand learning dynamics and error characteristics.
