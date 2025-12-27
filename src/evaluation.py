import numpy as np
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(mse(y_true, y_pred))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs(y_true - y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100


def correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_flat, y_pred_flat = y_true.flatten(), y_pred.flatten()
    if np.std(y_true_flat) < 1e-8 or np.std(y_pred_flat) < 1e-8:
        return 0.0
    return np.corrcoef(y_true_flat, y_pred_flat)[0, 1]


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot > 1e-8 else 0.0


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mse": mse(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "correlation": correlation(y_true, y_pred),
        "r2_score": r2_score(y_true, y_pred),
    }


def per_channel_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, channel_names: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    n_channels = y_true.shape[-1]
    if channel_names is None:
        channel_names = [f"ch_{i}" for i in range(n_channels)]
    return {
        name: compute_all_metrics(y_true[:, :, i], y_pred[:, :, i])
        for i, name in enumerate(channel_names)
    }


def per_timestep_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[int, Dict[str, float]]:
    """Compute metrics per prediction timestep (shows error growth with horizon)."""
    return {
        t: compute_all_metrics(y_true[:, t, :], y_pred[:, t, :])
        for t in range(y_true.shape[1])
    }


def frequency_band_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, fs: int = 250
) -> Dict[str, Dict[str, float]]:
    """Compute spectral error metrics across EEG frequency bands using FFT."""
    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, min(100, fs / 2 - 1)),
    }

    n_samples, n_timesteps, n_channels = y_true.shape
    y_true_flat = y_true.transpose(0, 2, 1).reshape(-1, n_timesteps)
    y_pred_flat = y_pred.transpose(0, 2, 1).reshape(-1, n_timesteps)

    freqs = np.fft.rfftfreq(n_timesteps, 1 / fs)
    fft_true = np.fft.rfft(y_true_flat, axis=1)
    fft_pred = np.fft.rfft(y_pred_flat, axis=1)

    results = {}
    for band_name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs <= high)
        if not np.any(mask):
            results[band_name] = {
                "rmse": np.nan,
                "mae": np.nan,
                "note": f"Not representable with {n_timesteps} samples",
            }
            continue

        true_band = np.abs(fft_true[:, mask])
        pred_band = np.abs(fft_pred[:, mask])
        band_mse = np.mean((true_band - pred_band) ** 2)
        true_power = np.mean(true_band**2)
        pred_power = np.mean(pred_band**2)

        results[band_name] = {
            "mse": float(band_mse),
            "rmse": float(np.sqrt(band_mse)),
            "mae": float(np.mean(np.abs(true_band - pred_band))),
            "true_power": float(true_power),
            "pred_power": float(pred_power),
            "power_ratio": float(pred_power / (true_power + 1e-8)),
        }
    return results


class Evaluator:
    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        channel_names: Optional[List[str]] = None,
        sampling_rate: int = 250,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.channel_names = channel_names
        self.sampling_rate = sampling_rate

    def predict(self, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = torch.FloatTensor(X[i : i + batch_size]).to(self.device)
                predictions.append(self.model(batch).cpu().numpy())
        return np.concatenate(predictions, axis=0)

    def evaluate(
        self, X: np.ndarray, y: np.ndarray, include_frequency_bands: bool = True
    ) -> Dict:
        y_pred = self.predict(X)
        results = {
            "overall": compute_all_metrics(y, y_pred),
            "per_timestep": per_timestep_metrics(y, y_pred),
            "per_channel": per_channel_metrics(y, y_pred, self.channel_names),
        }
        if include_frequency_bands:
            results["frequency_bands"] = frequency_band_metrics(
                y, y_pred, self.sampling_rate
            )
        return results


def format_results(results: Dict, indent: int = 0) -> str:
    lines = []
    prefix = "  " * indent
    for key, value in results.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.append(format_results(value, indent + 1))
        elif isinstance(value, float):
            lines.append(f"{prefix}{key}: {value:.6f}")
        else:
            lines.append(f"{prefix}{key}: {value}")
    return "\n".join(lines)


def compare_models(results: Dict[str, Dict]) -> str:
    lines = ["=" * 60, "MODEL COMPARISON", "=" * 60, "\nOverall Metrics:", "-" * 40]

    metrics = ["mse", "rmse", "mae", "correlation", "r2_score"]
    header = f"{'Model':<15} " + " ".join([f"{m:<12}" for m in metrics])
    lines.extend([header, "-" * len(header)])

    for model_name, model_results in results.items():
        overall = model_results.get("overall", {})
        row = f"{model_name:<15} " + " ".join(
            [f"{overall.get(m, np.nan):<12.6f}" for m in metrics]
        )
        lines.append(row)

    lines.extend(["\nFrequency Band RMSE:", "-" * 40])
    bands = ["delta", "theta", "alpha", "beta", "gamma"]
    header = f"{'Model':<15} " + " ".join([f"{b:<10}" for b in bands])
    lines.extend([header, "-" * len(header)])

    for model_name, model_results in results.items():
        freq = model_results.get("frequency_bands", {})
        row = f"{model_name:<15} " + " ".join(
            [f"{freq.get(b, {}).get('rmse', np.nan):<10.6f}" for b in bands]
        )
        lines.append(row)

    return "\n".join(lines)
