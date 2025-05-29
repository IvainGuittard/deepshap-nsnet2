import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torchaudio
from config.parameters import sample_rate, n_fft
from utils.data_utils import load_and_resample


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

    return mel_spectrogram
