import os
import sys
import torch
import torchaudio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from DeepShap.config.parameters import sample_rate, n_fft
from DeepShap.utils.common_utils import load_and_resample
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


def compute_snr_map(clean_path, noisy_path, eps=1e-8):
    """
    Compute a continuous SNR map SNR(f,t) in dB from clean and noisy audio.

    Args:
        clean_path (str): path to clean audio file, e.g. 'data/clean_input_tests/clean_p226_016.wav'
        eps (float): small number to avoid log(0)

    Returns:
        snr_map_np (np.ndarray): continuous SNR map in dB, shape (F, T)
    """
    # Deduce noisy path by removing 'clean_' prefix and changing folder
    filename = os.path.basename(clean_path).replace("clean_", "")

    clean, _ = load_and_resample(clean_path, sample_rate)
    noisy, _ = load_and_resample(noisy_path, sample_rate)
    S = torchaudio.transforms.Spectrogram(n_fft=n_fft, power=None)(clean).squeeze(0)
    X = torchaudio.transforms.Spectrogram(n_fft=n_fft, power=None)(noisy).squeeze(0)
    N = X - S  # Estimated noise STFT
    P_s = S.abs() ** 2
    P_n = N.abs() ** 2
    P_n = torch.clamp(P_n, min=1e-12)

    snr_map_db = 10 * torch.log10(P_s / (P_n))  # SNR in dB
    return snr_map_db.numpy()


def compute_binary_speech_mask(clean_path, eps=1e-8):
    """
    Compute binary mask M(f,t) where 1 = TF bin dominated by speech (SNR > 0 dB).

    Args:
        clean_path (str): path to clean audio file, e.g. 'data/clean_input_tests/clean_p226_016.wav'
        n_fft (int): FFT size for STFT
        hop_length (int): hop length for STFT
        eps (float): small number to avoid log(0)

    Returns:
        M (np.ndarray): binary mask (F, T) with 1 = speech dominant bin, 0 = noise dominant bin
    """
    snr_map = compute_snr_map(clean_path, eps=eps)

    # Binary mask: 1 if SNR > 0, else 0
    M = (snr_map > 0).astype(np.float32)

    return np.array(M, dtype=np.float32)
