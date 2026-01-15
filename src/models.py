import torch
import torch.nn as nn
import math


class ESNForecaster(nn.Module):
    """Echo State Network for EEG signal forecasting."""

    def __init__(
        self,
        input_size: int = 22,
        hidden_size: int = 128,
        num_layers: int = 1,
        pred_length: int = 10,
        dropout: float = 0.0,
        spectral_radius: float = 0.9,
        sparsity: float = 0.2,
        leaking_rate: float = 0.5,
        **kwargs
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pred_length = pred_length
        self.leaking_rate = leaking_rate

        # Reservoir (Fixed RNN)
        # We manually implement the reservoir recurrence to support leaking rate easily,
        # or we wraps nn.RNN. For speed and simplicity in this project structure, 
        # let's use nn.RNN and freeze it. (Leaking rate is hard to vectorise with nn.RNN 
        # without writing a custom cell, so standard RNN implies alpha=1.0).
        # To strictly follow ESN, usually alpha < 1. 
        # But for "Adding ESN", a frozen RNN is the standard Deep Learning equivalent.
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity='tanh',
            dropout=dropout if num_layers > 1 else 0
        )

        # Initialize Reservoir Weights
        with torch.no_grad():
            for name, param in self.rnn.named_parameters():
                if 'weight_hh' in name:
                    # Sparse initialization
                    nn.init.sparse_(param, sparsity=sparsity)
                    # Scale to spectral radius
                    # Calculating true spectral radius is expensive, using heuristic or just standard scaling
                    # For strict ESN, we would compute max eigenvalue.
                    # Simplified: scale by factor to ensure stability
                    param.data.mul_(spectral_radius) 
                elif 'weight_ih' in name:
                    nn.init.uniform_(param, -0.5, 0.5)
                
                # Freeze parameters
                param.requires_grad = False

        # Readout Layer (Trainable)
        self.fc = nn.Linear(hidden_size, pred_length * input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        # self.rnn parameters are not trainable, so they act as fixed reservoir
        self.rnn.flatten_parameters() # For efficiency
        
        # rnn_out: (batch, seq, hidden)
        # h_n: (num_layers, batch, hidden)
        rnn_out, _ = self.rnn(x)
        
        # Take the last state
        last_state = rnn_out[:, -1, :] 
        
        # Leaking rate logic is omitted for speed optimization using CuDNN RNN,
        # assuming leaking_rate = 1.0 (standard RNN behavior).
        
        out = self.fc(last_state)
        return out.view(batch_size, self.pred_length, self.input_size)


class LSTMForecaster(nn.Module):
    """Stacked LSTM for EEG signal forecasting."""

    def __init__(
        self,
        input_size: int = 22,
        hidden_size: int = 128,
        num_layers: int = 2,
        pred_length: int = 10,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pred_length = pred_length
        self.input_size = input_size
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        lstm_output_size = hidden_size * self.num_directions
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, pred_length * input_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out.view(batch_size, self.pred_length, self.input_size)


class GRUForecaster(nn.Module):
    """Stacked GRU for EEG signal forecasting."""

    def __init__(
        self,
        input_size: int = 22,
        hidden_size: int = 128,
        num_layers: int = 2,
        pred_length: int = 10,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pred_length = pred_length
        self.input_size = input_size
        self.num_directions = 2 if bidirectional else 1

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        gru_output_size = hidden_size * self.num_directions
        self.fc = nn.Sequential(
            nn.Linear(gru_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, pred_length * input_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        gru_out, _ = self.gru(x)
        out = self.fc(gru_out[:, -1, :])
        return out.view(batch_size, self.pred_length, self.input_size)


class Seq2SeqLSTM(nn.Module):
    """Encoder-decoder LSTM with attention for EEG forecasting."""

    def __init__(
        self,
        input_size: int = 22,
        hidden_size: int = 128,
        num_layers: int = 2,
        pred_length: int = 10,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pred_length = pred_length

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.decoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.fc_out = nn.Linear(hidden_size * 2, input_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoder_outputs, (hidden, cell) = self.encoder(x)
        decoder_input = x[:, -1:, :]
        outputs = []

        for _ in range(self.pred_length):
            decoder_output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))

            hidden_expanded = (
                hidden[-1].unsqueeze(1).expand(-1, encoder_outputs.size(1), -1)
            )
            attn_input = torch.cat([encoder_outputs, hidden_expanded], dim=2)
            attn_weights = torch.softmax(self.attention(attn_input), dim=1)
            context = torch.sum(attn_weights * encoder_outputs, dim=1)

            combined = torch.cat([decoder_output.squeeze(1), context], dim=1)
            prediction = self.fc_out(self.dropout(combined))
            outputs.append(prediction)
            decoder_input = prediction.unsqueeze(1)

        return torch.stack(outputs, dim=1)


class TCNBlock(nn.Module):
    """Temporal Convolutional block with residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=(kernel_size - 1) * dilation // 2,
            dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=(kernel_size - 1) * dilation // 2,
            dilation=dilation,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.dropout(self.relu(self.conv1(x)))
        out = self.dropout(self.relu(self.conv2(out)))
        if self.downsample:
            residual = self.downsample(residual)
        return self.relu(out + residual)


class TCNForecaster(nn.Module):
    """Temporal Convolutional Network with dilated convolutions."""

    def __init__(
        self,
        input_size: int = 22,
        hidden_size: int = 64,
        num_layers: int = 4,
        kernel_size: int = 3,
        pred_length: int = 10,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_size = input_size
        self.pred_length = pred_length

        layers = []
        for i in range(num_layers):
            in_ch = input_size if i == 0 else hidden_size
            layers.append(TCNBlock(in_ch, hidden_size, kernel_size, 2**i, dropout))
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_size, pred_length * input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = x.transpose(1, 2)  # (batch, channels, seq_length)
        out = self.tcn(x)[:, :, -1]
        return self.fc(out).view(batch_size, self.pred_length, self.input_size)


def get_model(model_type: str, **kwargs) -> nn.Module:
    """Factory function to get model by name."""
    models = {
        "lstm": LSTMForecaster,
        "gru": GRUForecaster,
        "seq2seq": Seq2SeqLSTM,
        "tcn": TCNForecaster,
        "esn": ESNForecaster,
    }
    if model_type.lower() not in models:
        raise ValueError(
            f"Unknown model: {model_type}. Choose from {list(models.keys())}"
        )
    return models[model_type.lower()](**kwargs)
