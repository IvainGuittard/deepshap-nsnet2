import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torchaudio
from config.parameters import sample_rate, n_fft
from utils.data_utils import load_and_resample
import numpy as np


def compute_log_mel_spectrogram(
    file_path, sample_rate=sample_rate, n_fft=n_fft, hop_length=256, n_mels=62
):
    """
    Compute the Mel spectrogram of an audio file.
    """
    waveform, _ = load_and_resample(file_path, sample_rate)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    mel_spectrogram = mel_transform(waveform)
    mel_spectrogram = torchaudio.functional.amplitude_to_DB(
        mel_spectrogram, multiplier=20.0, amin=1e-10, db_multiplier=0.0, top_db=80.0
    )
    # Compute Mel frequencies manually
    mel_min = 0
    mel_max = 2595 * np.log10(1 + (sample_rate // 2) / 700)  # Max Mel value
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)  # Mel scale points
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)  # Convert Mel to Hz
    mel_frequencies = hz_points[
        1:-1
    ]  # Exclude the first and last points (f_min, f_max)

    return mel_spectrogram, mel_frequencies
