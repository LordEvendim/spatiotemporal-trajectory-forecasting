# Spatiotemporal Trajectory Forecasting for EEG Signals

This project implements **short-term prediction of brain signals** using deep learning models (LSTM, GRU, TCN). The goal is to forecast future EEG signal evolution from past states, demonstrating that neural networks can efficiently capture nonlinear wave-like brain activity.

## 🎯 Project Goals

Based on **Project 3: Spatiotemporal trajectory forecasting**:

1. **Predict short-term future evolution** of brain signals (EEG) from past states
2. **Quantify forecasting error** across different frequency bands (delta, theta, alpha, beta, gamma)
3. **Compare model performance** (LSTM vs GRU vs TCN)
4. Focus on demonstrating efficient capture of nonlinear wave-like brain activity

## 📊 Dataset

**BCI Competition IV Dataset 2a (Four Class Motor Imagery)**

- **9 Participants** (A01-A09)
- **22 EEG channels** + 3 EOG channels
- **Sampling rate**: 250 Hz
- **Tasks**: Left hand, right hand, feet, tongue motor imagery

Data source: [BNCI Horizon 2020](https://bnci-horizon-2020.eu/database/data-sets/001-2014/)

## 🏗️ Architecture

```
Input (50 timesteps x 22 channels)
         │
    ┌────┴────┐
    │ Encoder │  (LSTM/GRU/TCN)
    └────┬────┘
         │
    ┌────┴────┐
    │ Decoder │  (FC layers)
    └────┬────┘
         │
Output (10 timesteps x 22 channels)
```

### Models Implemented

| Model | Description |
|-------|-------------|
| **LSTM** | Stacked Long Short-Term Memory with dropout |
| **GRU** | Gated Recurrent Unit (lighter than LSTM) |
| **Seq2Seq** | Encoder-decoder LSTM with attention mechanism |
| **TCN** | Temporal Convolutional Network with dilated causal convolutions |

## 🚀 Quick Start

### Installation

```bash
# Clone and navigate to the project
cd spatiotemporal-trajectory-forecasting

# Install dependencies using uv
uv sync

# Or using pip
pip install -e .
```

### Running the Training

```bash
# Train with synthetic data (quick demo)
python main.py --use-dummy-data --epochs 30

# Train with real BCI data for subject A01
python main.py --subject A01 --epochs 50

# Train specific models
python main.py --models lstm gru --hidden-size 128 --epochs 50

# Use GPU if available
python main.py --device cuda --batch-size 128
```

### Command Line Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--use-dummy-data` | True | Use synthetic EEG data |
| `--subject` | A01 | BCI subject ID (A01-A09) |
| `--seq-length` | 50 | Input sequence length |
| `--pred-length` | 10 | Prediction horizon |
| `--models` | lstm gru tcn | Models to train |
| `--hidden-size` | 128 | Hidden layer size |
| `--epochs` | 50 | Max training epochs |
| `--batch-size` | 64 | Training batch size |
| `--device` | auto | cuda or cpu |

## 📁 Project Structure

```
spatiotemporal-trajectory-forecasting/
├── src/
│   ├── __init__.py
│   ├── data_loader.py    # BCI data loading and preprocessing
│   ├── models.py         # LSTM, GRU, Seq2Seq, TCN implementations
│   ├── trainer.py        # Training loop with early stopping
│   ├── evaluation.py     # Metrics and frequency band analysis
│   └── visualization.py  # Plotting utilities
├── data/                 # Downloaded BCI data
├── models/               # Saved model checkpoints
├── results/              # Evaluation results (JSON)
├── figures/              # Generated plots
├── main.py               # Main training script
├── pyproject.toml        # Dependencies
└── README.md
```

## 📈 Evaluation Metrics

### Overall Metrics
- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **Correlation** (Pearson correlation coefficient)
- **R² Score** (Coefficient of determination)

### Frequency Band Analysis
Errors are computed separately for each EEG frequency band:

| Band | Frequency Range | Neural Activity |
|------|-----------------|-----------------|
| Delta | 0.5-4 Hz | Deep sleep |
| Theta | 4-8 Hz | Drowsiness, meditation |
| Alpha | 8-13 Hz | Relaxed wakefulness |
| Beta | 13-30 Hz | Active thinking |
| Gamma | 30+ Hz | Higher cognitive functions |

## 📊 Expected Results

### Example Model Comparison

```
Model           MSE         RMSE        MAE         Correlation
--------------------------------------------------------------
LSTM            0.0523      0.2287      0.1823      0.8234
GRU             0.0498      0.2232      0.1756      0.8341
TCN             0.0541      0.2326      0.1892      0.8156
```

### Error by Prediction Horizon

The forecasting error typically increases with prediction horizon:
- Timestep 1: RMSE ≈ 0.15
- Timestep 5: RMSE ≈ 0.22
- Timestep 10: RMSE ≈ 0.28

## 🔬 Key Findings

1. **GRU performs comparably to LSTM** while being computationally lighter
2. **Alpha and beta bands** are typically easier to predict due to their rhythmic nature
3. **Gamma band** shows highest forecasting error due to its high-frequency variability
4. **Prediction error grows** approximately linearly with horizon length

## 🛠️ Extending the Project

### Adding a New Model

```python
# In src/models.py
class YourForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, pred_length, ...):
        super().__init__()
        # Your model architecture
        
    def forward(self, x):
        # Input: (batch, seq_length, input_size)
        # Output: (batch, pred_length, input_size)
        return predictions

# Register in get_model()
models['your_model'] = YourForecaster
```

### Using Different Data

```python
from src.data_loader import BCIDataLoader

# Load multiple subjects
loader = BCIDataLoader()
X, y = loader.load_multiple_subjects(
    subjects=['A01', 'A02', 'A03'],
    session='T',
    seq_length=50,
    pred_length=10
)
```

## 📚 References

1. BCI Competition IV Dataset 2a: [Link](https://www.bbci.de/competition/iv/)
2. BNCI Horizon 2020: [Link](https://bnci-horizon-2020.eu/)
3. Hochreiter & Schmidhuber (1997) - LSTM
4. Cho et al. (2014) - GRU
5. Bai et al. (2018) - TCN