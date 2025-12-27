import json
import time
from pathlib import Path
from typing import Tuple, Optional, List
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainingConfig:
    model_type: str = "lstm"
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = False
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    epochs: int = 50
    patience: int = 10
    seq_length: int = 50
    pred_length: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainingMetrics:
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_epoch: int = 0
    training_time: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


class Trainer:
    def __init__(
        self, model: nn.Module, config: TrainingConfig, save_dir: Optional[str] = None
    ):
        self.model = model.to(config.device)
        self.config = config
        self.device = config.device
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )
        self.early_stopping = EarlyStopping(patience=config.patience)
        self.metrics = TrainingMetrics()
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def create_dataloaders(
        self, X_train, y_train, X_val, y_val
    ) -> Tuple[DataLoader, DataLoader]:
        train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        train_loader = DataLoader(
            train_ds, batch_size=self.config.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.config.batch_size, shuffle=False
        )
        return train_loader, val_loader

    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(batch_x), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def validate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                loss = self.criterion(self.model(batch_x), batch_y)
                total_loss += loss.item()
        return total_loss / len(val_loader)

    def train(self, X_train, y_train, X_val, y_val) -> TrainingMetrics:
        train_loader, val_loader = self.create_dataloaders(
            X_train, y_train, X_val, y_val
        )

        print(f"\nTraining {self.config.model_type.upper()} model")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        print("-" * 50)

        start_time = time.time()
        best_model_state = None

        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            self.metrics.train_losses.append(train_loss)
            self.metrics.val_losses.append(val_loss)
            self.scheduler.step(val_loss)

            if val_loss < self.metrics.best_val_loss:
                self.metrics.best_val_loss = val_loss
                self.metrics.best_epoch = epoch
                best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }

            print(
                f"Epoch {epoch + 1}/{self.config.epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}"
            )

            if self.early_stopping(val_loss):
                print(f"Early stopping at epoch {epoch + 1}")
                break

        self.metrics.training_time = time.time() - start_time

        if best_model_state:
            self.model.load_state_dict(best_model_state)

        print(f"\nCompleted in {self.metrics.training_time:.2f}s")
        print(
            f"Best val loss: {self.metrics.best_val_loss:.6f} at epoch {self.metrics.best_epoch + 1}"
        )

        if self.save_dir:
            self.save_checkpoint()

        return self.metrics

    def save_checkpoint(self, name: Optional[str] = None):
        name = name or f"{self.config.model_type}_checkpoint"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config.to_dict(),
                "metrics": self.metrics.to_dict(),
            },
            self.save_dir / f"{name}.pt",
        )

        with open(self.save_dir / f"{name}_metrics.json", "w") as f:
            json.dump(
                {"config": self.config.to_dict(), "metrics": self.metrics.to_dict()},
                f,
                indent=2,
            )

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.metrics = TrainingMetrics(**checkpoint["metrics"])


def train_test_split(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2, val_size: float = 0.1
) -> Tuple[np.ndarray, ...]:
    """Temporal split to avoid data leakage."""
    n = len(X)
    train_end = int(n * (1 - test_size - val_size))
    val_end = int(n * (1 - test_size))
    return (
        X[:train_end],
        X[train_end:val_end],
        X[val_end:],
        y[:train_end],
        y[train_end:val_end],
        y[val_end:],
    )
