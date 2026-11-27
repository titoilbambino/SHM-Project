"""Keras/Torch training and prediction for endpoint strain models."""

import os
from math import ceil, isfinite

os.environ["KERAS_BACKEND"] = "torch"

import keras
import torch
from keras import layers
from torch.utils.data import DataLoader


def require_device(device):
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable to this Python process.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable to this Python process.")
    if device not in {"mps", "cuda", "cpu"}:
        raise ValueError(f"Unknown training device: {device}")


def build_tcn(n_features, architecture, learning_rate):
    inputs = layers.Input(shape=(None, n_features))
    width = architecture["channels"]
    x = layers.Conv1D(width, 1, padding="causal")(inputs)
    for index, dilation in enumerate(architecture["dilations"]):
        residual = x
        x = layers.Conv1D(width, architecture["kernel_size"], padding="causal",
                          dilation_rate=dilation)(x)
        x = layers.Activation("relu")(layers.Add()([residual, x]))
        if index in architecture["pool_after"]:
            x = layers.MaxPooling1D(pool_size=architecture["pool_size"], padding="same")(x)
    model = keras.Model(inputs, layers.Dense(1)(x[:, -1, :]))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss="mse")
    return model


class WithinEpochValidation(keras.callbacks.Callback):
    """Select by full-recording MSE after at least one pass over the fit data."""

    def __init__(self, validation, device, steps, checks, patience):
        super().__init__()
        if checks < 1 or patience < 1:
            raise ValueError("Validation frequency and patience must be positive")
        checks = min(checks, steps)
        self.scheduled_steps = {ceil(i * steps / checks) for i in range(1, checks + 1)}
        self.steps = steps
        self.patience = patience
        self.count = len(validation)
        self.batches = [(x.to(device), y.to(device)) for x, y in DataLoader(validation, batch_size=256)]
        self.checks = []
        self.stale = 0
        self.best_loss = float("inf")
        self.best_weights = None

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_index = epoch

    def on_train_batch_end(self, batch, logs=None):
        step = batch + 1
        if step not in self.scheduled_steps:
            return
        self.model.eval()
        try:
            with torch.no_grad():
                error = sum((self.model(x, training=False).reshape_as(y) - y).square().sum()
                            for x, y in self.batches)
                loss = error.item() / self.count
        finally:
            self.model.train()
        if not isfinite(loss):
            raise FloatingPointError(f"Non-finite validation MSE: {loss}")
        update = self.epoch_index * self.steps + step
        progress = update / self.steps
        self.checks.append({"epoch": progress, "update": update, "val_loss": loss})
        print(f"Validation at pass {progress:.3f}, update {update}: MSE {loss:.8f}", flush=True)
        if update < self.steps:
            return
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_weights = [w.copy() for w in self.model.get_weights()]
            self.best_update = update
            self.stale = 0
        else:
            self.stale += 1
            if self.stale >= self.patience:
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        self.model.set_weights(self.best_weights)


def fit_tcn(dataset, *, n_features, architecture, training, device, epochs, validation=None,
            build_model=build_tcn):
    keras.backend.clear_session()
    keras.utils.set_random_seed(training["seed"])
    batch_size = training["batch_size"]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    monitor = None
    if validation is not None:
        monitor = WithinEpochValidation(validation, device, len(loader),
                                        training["validation_checks_per_epoch"], training["patience_checks"])
    with keras.device(device):
        model = build_model(n_features, architecture, training["learning_rate"])
        history = model.fit(loader, epochs=epochs, callbacks=[monitor] if monitor else [],
                            shuffle=False, verbose=2).history
    if monitor is not None:
        history["validation_checks"] = monitor.checks
        return model, monitor.best_update, history
    return model, epochs * len(loader), history


def predict(model, dataset, device):
    with keras.device(device):
        return model.predict(DataLoader(dataset, batch_size=256), verbose=0).ravel()
