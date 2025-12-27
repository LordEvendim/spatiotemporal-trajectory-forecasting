import torch
import torch.nn as nn


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
    }
    if model_type.lower() not in models:
        raise ValueError(
            f"Unknown model: {model_type}. Choose from {list(models.keys())}"
        )
    return models[model_type.lower()](**kwargs)
