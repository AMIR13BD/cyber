"""PyTorch feed-forward autoencoder for normal-only intrusion anomaly detection."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.preprocessing import RANDOM_STATE

DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 256
DEFAULT_LR = 1e-3
DEFAULT_ENCODING_DIM = 8
DEFAULT_THRESHOLD_PERCENTILE = 95
EARLY_STOPPING_PATIENCE = 3


def set_seed(seed: int = RANDOM_STATE) -> None:
    """Fix randomness for reproducible training."""
    torch.manual_seed(seed)
    np.random.seed(seed)


class TabularAutoencoder(nn.Module):
    """Encoder-decoder network matching the source tutorial architecture."""

    def __init__(self, input_dim: int, encoding_dim: int = DEFAULT_ENCODING_DIM) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, encoding_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample mean squared reconstruction error."""
        reconstructed = self.forward(x)
        return torch.mean((x - reconstructed) ** 2, dim=1)


def train_autoencoder(
    x_normal: np.ndarray,
    input_dim: int,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    encoding_dim: int = DEFAULT_ENCODING_DIM,
    x_val_normal: np.ndarray | None = None,
    patience: int = EARLY_STOPPING_PATIENCE,
) -> TabularAutoencoder:
    """Train an autoencoder on benign traffic only with optional early stopping."""
    set_seed()
    device = torch.device("cpu")
    x_tensor = torch.tensor(x_normal, dtype=torch.float32)
    loader = DataLoader(TensorDataset(x_tensor), batch_size=batch_size, shuffle=True)

    model = TabularAutoencoder(input_dim, encoding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    stale_epochs = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)
        train_loss = epoch_loss / len(x_normal)

        if x_val_normal is not None:
            model.eval()
            with torch.no_grad():
                val_tensor = torch.tensor(x_val_normal, dtype=torch.float32, device=device)
                val_loss = criterion(model(val_tensor), val_tensor).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                stale_epochs = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale_epochs += 1
            if (epoch + 1) % 5 == 0:
                print(
                    f"  Autoencoder epoch {epoch + 1}/{epochs}, "
                    f"train MSE={train_loss:.6f}, val MSE={val_loss:.6f}"
                )
            if stale_epochs >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break
        elif (epoch + 1) % 5 == 0:
            print(f"  Autoencoder epoch {epoch + 1}/{epochs}, train MSE={train_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def reconstruction_errors(model: TabularAutoencoder, x: np.ndarray) -> np.ndarray:
    """Compute reconstruction error for each sample."""
    device = torch.device("cpu")
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
        return model.reconstruction_error(x_tensor).cpu().numpy()


def fit_threshold(errors_normal: np.ndarray, percentile: float = DEFAULT_THRESHOLD_PERCENTILE) -> float:
    """Calibrate anomaly threshold from normal reconstruction errors."""
    return float(np.percentile(errors_normal, percentile))


def predict_from_errors(
    errors: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert reconstruction errors to binary predictions and normalized scores."""
    y_pred = (errors >= threshold).astype(int)
    score_min, score_max = errors.min(), errors.max()
    y_score = (errors - score_min) / (score_max - score_min + 1e-9)
    return y_pred, y_score


def train_and_evaluate_autoencoder(
    x_train_normal: np.ndarray,
    x_val_normal: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    input_dim: int,
) -> tuple[TabularAutoencoder, np.ndarray, np.ndarray, float, np.ndarray]:
    """Train on normal data, threshold on validation normal, score test set."""
    model = train_autoencoder(
        x_train_normal,
        input_dim=input_dim,
        x_val_normal=x_val_normal,
    )
    val_errors = reconstruction_errors(model, x_val_normal)
    threshold = fit_threshold(val_errors)
    test_errors = reconstruction_errors(model, x_test)
    y_pred, y_score = predict_from_errors(test_errors, threshold)
    return model, y_pred, y_score, threshold, test_errors
