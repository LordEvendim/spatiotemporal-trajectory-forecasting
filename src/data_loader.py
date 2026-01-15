import urllib.request
import ssl
from pathlib import Path
from typing import Tuple, Optional, List
import numpy as np
import scipy.io
from scipy.signal import butter, filtfilt
import mne
from tqdm import tqdm


class BCIDataLoader:
    """Loader for BCI Competition IV Dataset 2a (22 EEG channels, 9 subjects)."""

    BASE_URL = "https://bnci-horizon-2020.eu/database/data-sets/001-2014/"
    DATA_DIR = Path("data")
    SUBJECTS = ["A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08", "A09"]
    SESSIONS = ["T", "E"]
    EEG_CHANNELS = [
        "Fz",
        "FC3",
        "FC1",
        "FCz",
        "FC2",
        "FC4",
        "C5",
        "C3",
        "C1",
        "Cz",
        "C2",
        "C4",
        "C6",
        "CP3",
        "CP1",
        "CPz",
        "CP2",
        "CP4",
        "P1",
        "Pz",
        "P2",
        "POz",
    ]
    SAMPLING_RATE = 250

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir) if data_dir else self.DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_dataset(self, subjects: Optional[List[str]] = None) -> None:
        if subjects is None:
            subjects = self.SUBJECTS

        # Create unverified SSL context to bypass certificate verification issues
        ssl_context = ssl._create_unverified_context()

        print(f"Downloading BCI Competition IV Dataset 2a to {self.data_dir}")
        for subject in tqdm(subjects, desc="Downloading subjects"):
            for session in self.SESSIONS:
                filename = f"{subject}{session}.mat"
                filepath = self.data_dir / filename
                if filepath.exists():
                    continue
                url = f"{self.BASE_URL}{filename}"
                try:
                    with urllib.request.urlopen(url, context=ssl_context) as response, \
                         open(filepath, "wb") as out_file:
                        out_file.write(response.read())
                except Exception as e:
                    print(f"Error downloading {filename}: {e}")

    def load_mat_file(self, filepath: str) -> Tuple[np.ndarray, np.ndarray, int]:
        """Load data from .mat file, extracting raw EEG and events."""
        mat = scipy.io.loadmat(filepath, struct_as_record=True)
        data_struct = mat['data']
        
        X_list = []
        events_list = []
        offset = 0
        
        # Iterate over all runs (usually 9 runs)
        for i in range(data_struct.shape[1]):
            run = data_struct[0, i]
            
            # Extract raw data (Timepoints x Channels)
            # Original GDF has 22 EEG + 3 EOG. We keep all for now, slice later if needed
            # or slice strictly to 22 here as in original code (data[:22])
            run_X = run['X'][0, 0]
            
            # Extract events if available
            if 'trial' in run.dtype.names and 'y' in run.dtype.names:
                run_trial = run['trial'][0, 0].flatten()
                run_y = run['y'][0, 0].flatten()
                
                if run_trial.size > 0:
                    # Create events array: [sample_index, 0, label]
                    # Note: MATLAB is 1-based, Python is 0-based. Subtract 1 from index.
                    current_events = np.column_stack((
                        run_trial - 1 + offset, 
                        np.zeros_like(run_trial, dtype=int), 
                        run_y
                    ))
                    events_list.append(current_events)
            
            X_list.append(run_X)
            offset += run_X.shape[0]
            
        # Concatenate all runs
        # X shape became (Total_Timepoints, 25).
        # We need (22, Total_Timepoints) to match original mne.get_data()[:22] behavior
        data_full = np.concatenate(X_list, axis=0)
        data = data_full[:, :22].T  # Transpose to (Channels, Timepoints)
        
        if events_list:
            events = np.concatenate(events_list, axis=0)
        else:
            events = np.array([])
            
        # Sampling rate is usually 250Hz for this dataset
        sfreq = 250
            
        return data, events, sfreq

    def bandpass_filter(
        self,
        data: np.ndarray,
        lowcut: float = 0.5,
        highcut: float = 100.0,
        fs: int = 250,
        order: int = 4,
    ) -> np.ndarray:
        nyq = 0.5 * fs
        b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
        filtered = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered[i] = filtfilt(b, a, data[i])
        return filtered

    def extract_frequency_bands(self, data: np.ndarray, fs: int = 250) -> dict:
        bands = {
            "delta": (0.5, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30),
            "gamma": (30, 100),
        }
        return {
            name: self.bandpass_filter(data, low, high, fs)
            for name, (low, high) in bands.items()
        }

    def create_sequences(
        self,
        data: np.ndarray,
        seq_length: int = 50,
        pred_length: int = 10,
        stride: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_channels, n_samples = data.shape
        data = data.T
        n_sequences = (n_samples - seq_length - pred_length) // stride + 1

        X = np.zeros((n_sequences, seq_length, n_channels))
        y = np.zeros((n_sequences, pred_length, n_channels))

        for i in range(n_sequences):
            start = i * stride
            X[i] = data[start : start + seq_length]
            y[i] = data[start + seq_length : start + seq_length + pred_length]
        return X, y

    def load_and_preprocess(
        self,
        subject: str = "A01",
        session: str = "T",
        seq_length: int = 50,
        pred_length: int = 10,
        stride: int = 25,
        lowcut: float = 0.5,
        highcut: float = 100.0,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        filepath = self.data_dir / f"{subject}{session}.mat"
        if not filepath.exists():
            self.download_dataset([subject])

        data, events, sfreq = self.load_mat_file(str(filepath))
        data_filtered = self.bandpass_filter(data, lowcut, highcut, sfreq)

        # Z-score normalization per channel
        data_normalized = (
            data_filtered - data_filtered.mean(axis=1, keepdims=True)
        ) / (data_filtered.std(axis=1, keepdims=True) + 1e-8)

        X, y = self.create_sequences(data_normalized, seq_length, pred_length, stride)

        band_data = {}
        for band_name, band_signal in self.extract_frequency_bands(data, sfreq).items():
            band_norm = (band_signal - band_signal.mean(axis=1, keepdims=True)) / (
                band_signal.std(axis=1, keepdims=True) + 1e-8
            )
            X_band, y_band = self.create_sequences(
                band_norm, seq_length, pred_length, stride
            )
            band_data[band_name] = {"X": X_band, "y": y_band}

        return X, y, band_data

    def load_multiple_subjects(
        self, subjects: Optional[List[str]] = None, sessions: Optional[List[str]] = None, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        if subjects is None:
            subjects = self.SUBJECTS
        if sessions is None:
            sessions = self.SESSIONS

        all_X, all_y = [], []
        for subject in tqdm(subjects, desc="Loading subjects"):
            for session in sessions:
                try:
                    X, y, _ = self.load_and_preprocess(subject, session, **kwargs)
                    all_X.append(X)
                    all_y.append(y)
                except Exception as e:
                    print(f"Error loading {subject} session {session}: {e}")

        if not all_X:
            raise RuntimeError("No data loaded successfully!")

        return np.concatenate(all_X, axis=0), np.concatenate(all_y, axis=0)


def create_dummy_data(
    n_samples: int = 10000,
    n_channels: int = 22,
    seq_length: int = 50,
    pred_length: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create synthetic EEG-like data with alpha, beta, gamma rhythms."""
    t = np.linspace(0, n_samples / 250, n_samples)

    data = np.zeros((n_channels, n_samples))
    for ch in range(n_channels):
        alpha = np.sin(2 * np.pi * (8 + ch * 0.2) * t)
        beta = 0.5 * np.sin(2 * np.pi * (18 + ch * 0.3) * t)
        gamma = 0.3 * np.sin(2 * np.pi * (40 + ch * 0.5) * t)
        noise = 0.1 * np.random.randn(n_samples)
        data[ch] = alpha + beta + gamma + noise

    data = (data - data.mean(axis=1, keepdims=True)) / (
        data.std(axis=1, keepdims=True) + 1e-8
    )
    data = data.T

    stride = 25
    n_sequences = (n_samples - seq_length - pred_length) // stride + 1
    X = np.zeros((n_sequences, seq_length, n_channels))
    y = np.zeros((n_sequences, pred_length, n_channels))

    for i in range(n_sequences):
        start = i * stride
        X[i] = data[start : start + seq_length]
        y[i] = data[start + seq_length : start + seq_length + pred_length]

    return X, y
