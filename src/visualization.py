import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def set_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"figure.figsize": (12, 8), "font.size": 12})


def plot_training_history(
    train_losses: List[float],
    val_losses: List[float],
    title: str = "Training History",
    save_path: Optional[str] = None,
) -> plt.Figure:
    set_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, "b-", label="Training Loss", linewidth=2)
    ax.plot(epochs, val_losses, "r-", label="Validation Loss", linewidth=2)

    best_epoch = np.argmin(val_losses) + 1
    ax.axvline(
        x=best_epoch, color="g", linestyle="--", alpha=0.7, label=f"Best ({best_epoch})"
    )
    ax.scatter([best_epoch], [min(val_losses)], color="g", s=100, zorder=5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_idx: int = 0,
    channels: List[int] = None,
    channel_names: List[str] = None,
    title: str = "Prediction vs Ground Truth",
    save_path: Optional[str] = None,
) -> plt.Figure:
    set_style()
    channels = channels or list(range(min(4, y_true.shape[-1])))
    channel_names = channel_names or [f"Channel {i}" for i in channels]

    fig, axes = plt.subplots(len(channels), 1, figsize=(12, 3 * len(channels)))
    if len(channels) == 1:
        axes = [axes]

    timesteps = range(y_true.shape[1])
    for ax, ch, name in zip(axes, channels, channel_names):
        ax.plot(
            timesteps,
            y_true[sample_idx, :, ch],
            "b-",
            label="Ground Truth",
            linewidth=2,
        )
        ax.plot(
            timesteps, y_pred[sample_idx, :, ch], "r--", label="Prediction", linewidth=2
        )
        ax.fill_between(
            timesteps,
            y_true[sample_idx, :, ch],
            y_pred[sample_idx, :, ch],
            alpha=0.3,
            color="gray",
        )
        ax.set_ylabel(name)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Timestep")
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_error_by_timestep(
    per_timestep_metrics: Dict[int, Dict[str, float]],
    metric: str = "rmse",
    title: str = "Error by Prediction Horizon",
    save_path: Optional[str] = None,
) -> plt.Figure:
    set_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    timesteps = sorted(per_timestep_metrics.keys())
    values = [per_timestep_metrics[t][metric] for t in timesteps]

    ax.bar(timesteps, values, color="steelblue", alpha=0.7, edgecolor="navy")
    ax.plot(timesteps, values, "r-", linewidth=2, marker="o", markersize=8)
    ax.set_xlabel("Prediction Timestep")
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_frequency_band_comparison(
    model_results: Dict[str, Dict],
    metric: str = "rmse",
    title: str = "Frequency Band Performance",
    save_path: Optional[str] = None,
) -> plt.Figure:
    set_style()
    bands = ["delta", "theta", "alpha", "beta", "gamma"]
    band_labels = [
        "Delta\n(0.5-4)",
        "Theta\n(4-8)",
        "Alpha\n(8-13)",
        "Beta\n(13-30)",
        "Gamma\n(30+)",
    ]

    n_models = len(model_results)
    x = np.arange(len(bands))
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, n_models))

    for i, (model_name, results) in enumerate(model_results.items()):
        freq_results = results.get("frequency_bands", {})
        values = [freq_results.get(b, {}).get(metric, np.nan) for b in bands]
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(
            x + offset,
            values,
            width,
            label=model_name.upper(),
            color=colors[i],
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xlabel("Frequency Band")
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(band_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_model_comparison(
    model_results: Dict[str, Dict],
    metrics: List[str] = None,
    title: str = "Model Comparison",
    save_path: Optional[str] = None,
) -> plt.Figure:
    set_style()
    metrics = metrics or ["mse", "rmse", "mae", "correlation"]
    n_models = len(model_results)

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 6))
    if len(metrics) == 1:
        axes = [axes]

    colors = plt.cm.Set2(np.linspace(0, 1, n_models))

    for ax, metric in zip(axes, metrics):
        model_names = [m.upper() for m in model_results.keys()]
        values = [r.get("overall", {}).get(metric, 0) for r in model_results.values()]

        bars = ax.bar(
            model_names,
            values,
            color=colors[:n_models],
            edgecolor="black",
            linewidth=0.5,
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        ax.set_ylabel(metric.upper())
        ax.set_title(metric.upper())
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def create_summary_report(
    model_results: Dict[str, Dict],
    training_histories: Dict[str, Tuple[List, List]],
    save_dir: str = "reports",
) -> None:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    print(f"Creating summary report in {save_dir}/")

    for model_name, (train_loss, val_loss) in training_histories.items():
        plot_training_history(
            train_loss,
            val_loss,
            f"{model_name.upper()} Training",
            save_path / f"{model_name}_training.png",
        )
        plt.close()

    plot_model_comparison(
        model_results,
        title="Model Comparison",
        save_path=save_path / "model_comparison.png",
    )
    plt.close()

    plot_frequency_band_comparison(
        model_results,
        title="Frequency Band Performance",
        save_path=save_path / "frequency_bands.png",
    )
    plt.close()

    for model_name, results in model_results.items():
        per_timestep = results.get("per_timestep", {})
        if per_timestep:
            plot_error_by_timestep(
                per_timestep,
                title=f"{model_name.upper()} Error by Horizon",
                save_path=save_path / f"{model_name}_timestep_error.png",
            )
            plt.close()

    print(f"Report saved to {save_dir}/")
