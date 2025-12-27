import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data_loader import BCIDataLoader, create_dummy_data
from src.models import get_model
from src.trainer import Trainer, TrainingConfig, train_test_split
from src.evaluation import Evaluator, compare_models, format_results
from src.visualization import (
    plot_training_history,
    plot_predictions,
    plot_model_comparison,
    plot_frequency_band_comparison,
    plot_error_by_timestep,
    create_summary_report,
)


def setup_directories():
    for d in ["data", "models", "results", "figures"]:
        Path(d).mkdir(exist_ok=True)


def load_data(args):
    if args.use_dummy_data:
        print("Using synthetic EEG-like data...")
        X, y = create_dummy_data(args.n_samples, 22, args.seq_length, args.pred_length)
    else:
        print(f"Loading BCI data for subject {args.subject}...")
        loader = BCIDataLoader(data_dir=args.data_dir)
        try:
            X, y, _ = loader.load_and_preprocess(
                args.subject, "T", args.seq_length, args.pred_length, args.stride
            )
        except Exception as e:
            print(f"Error: {e}. Falling back to synthetic data...")
            X, y = create_dummy_data(
                args.n_samples, 22, args.seq_length, args.pred_length
            )

    X_train, X_val, X_test, y_train, y_val, y_test = train_test_split(X, y, 0.15, 0.15)

    print(f"\nData: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(model_type, X_train, y_train, X_val, y_val, args):
    model = get_model(
        model_type=model_type,
        input_size=X_train.shape[-1],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        pred_length=args.pred_length,
        dropout=args.dropout,
    )
    config = TrainingConfig(
        model_type=model_type,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        patience=args.patience,
        seq_length=args.seq_length,
        pred_length=args.pred_length,
        device=args.device,
    )
    trainer = Trainer(model, config, save_dir=f"models/{model_type}")
    metrics = trainer.train(X_train, y_train, X_val, y_val)
    return model, metrics


def evaluate_model(model, X_test, y_test, args):
    evaluator = Evaluator(model=model, device=args.device, sampling_rate=250)
    return evaluator.evaluate(X_test, y_test, include_frequency_bands=True)


def main():
    parser = argparse.ArgumentParser(description="EEG Signal Forecasting")

    # Data
    parser.add_argument("--use-dummy-data", action="store_true", default=True)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--subject", type=str, default="A01")
    parser.add_argument("--n-samples", type=int, default=20000)

    # Sequences
    parser.add_argument("--seq-length", type=int, default=50)
    parser.add_argument("--pred-length", type=int, default=10)
    parser.add_argument("--stride", type=int, default=25)

    # Model
    parser.add_argument("--models", type=str, nargs="+", default=["lstm", "gru", "tcn"])
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)

    # Training
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Output
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--no-plots", action="store_true")

    args = parser.parse_args()

    print("=" * 60)
    print("EEG SIGNAL FORECASTING")
    print("=" * 60)
    print(f"Device: {args.device}, Models: {args.models}")
    print(f"Sequence: {args.seq_length} -> Predict: {args.pred_length}")

    setup_directories()
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(args)

    all_results, all_histories, trained_models = {}, {}, {}

    for model_type in args.models:
        print(f"\n{'=' * 60}\nTRAINING {model_type.upper()}\n{'=' * 60}")

        model, metrics = train_model(model_type, X_train, y_train, X_val, y_val, args)
        trained_models[model_type] = model
        all_histories[model_type] = (metrics.train_losses, metrics.val_losses)

        print(f"\nEvaluating {model_type.upper()}...")
        results = evaluate_model(model, X_test, y_test, args)
        all_results[model_type] = results
        print(f"\n{model_type.upper()} Results:\n{format_results(results['overall'])}")

    print(f"\n{'=' * 60}\nFINAL COMPARISON\n{'=' * 60}")
    print(compare_models(all_results))

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(args.output_dir) / f"results_{timestamp}.json"
    results_file.parent.mkdir(exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        return obj

    with open(results_file, "w") as f:
        json.dump(convert(all_results), f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Visualizations
    if not args.no_plots:
        print("\nGenerating visualizations...")
        figures_path = Path("figures") / timestamp
        figures_path.mkdir(parents=True, exist_ok=True)

        for model_type, (train_loss, val_loss) in all_histories.items():
            plot_training_history(
                train_loss,
                val_loss,
                f"{model_type.upper()} Training",
                str(figures_path / f"{model_type}_training.png"),
            )

        plot_model_comparison(
            all_results, save_path=str(figures_path / "model_comparison.png")
        )
        plot_frequency_band_comparison(
            all_results, save_path=str(figures_path / "frequency_bands.png")
        )

        for model_type, model in trained_models.items():
            y_pred = Evaluator(model, device=args.device).predict(X_test[:100])
            plot_predictions(
                y_test[:100],
                y_pred,
                0,
                title=f"{model_type.upper()} Predictions",
                save_path=str(figures_path / f"{model_type}_predictions.png"),
            )

        for model_type, results in all_results.items():
            if results.get("per_timestep"):
                plot_error_by_timestep(
                    results["per_timestep"],
                    title=f"{model_type.upper()} Error by Horizon",
                    save_path=str(figures_path / f"{model_type}_timestep_error.png"),
                )

        create_summary_report(
            all_results, all_histories, str(figures_path / "summary_report")
        )
        print(f"Figures saved to {figures_path}/")

    print(f"\n{'=' * 60}\nTRAINING COMPLETE\n{'=' * 60}")
    return all_results


if __name__ == "__main__":
    main()
