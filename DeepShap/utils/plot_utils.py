import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from utils.audio_features import compute_log_mel_spectrogram
from config.parameters import sample_rate


def save_plot(
    file_basename,
    division,
    noise_type,
    baseline_type,
    freq_range=None,
    rms_amplitude=None,
):
    folder_name = file_basename.replace(".wav", "")
    file_name = file_basename.replace(".wav", f"_div{division}_attribution_plot.png")
    if freq_range is not None and rms_amplitude is not None:
        file_name = file_basename.replace(
            ".wav",
            f"_div{division}_freq_{freq_range[0]}_{freq_range[1]}_rms_{rms_amplitude}_attribution_plot.png",
        )
    save_path = f"DeepShap/attributions/{noise_type}_noise_{baseline_type}_baseline/plots/{folder_name}/{file_name}"
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    plt.savefig(save_path)
    print(f"Subplot saved to {save_path}")


def plot_spectrogram_and_attributions(
    input_path,
    file_basename,
    converted_attr_map,
    division,
    noise_type,
    baseline_type,
    freq_range=None,
    rms_amplitude=None,
):
    """
    Plots the Mel spectrogram and the DeepLIFTShap attribution map side by side.
    Saves the plot as a PNG file.
    Args:
        input_path (str): Path to the input audio file.
        converted_attr_map (np.ndarray): Attribution map with frequency bands on the y-axis and time bands on the x-axis.
    """
    mel_spectrogram, mel_frequencies = compute_log_mel_spectrogram(
        input_path, sample_rate=sample_rate
    )
    mel_spectrogram = mel_spectrogram.squeeze(0).numpy()
    mel_spectrogram_resized = zoom(
        mel_spectrogram,
        (
            converted_attr_map.shape[0] / mel_spectrogram.shape[0],
            converted_attr_map.shape[1] / mel_spectrogram.shape[1],
        ),
        order=1,
    )

    num_frames = converted_attr_map.shape[1]
    duration = num_frames / sample_rate

    # Create a subplot with two graphs side by side
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    title = f"Mel Spectrogram and DeepLIFTShap Attributions for {file_basename}, {noise_type} noise, {baseline_type} baseline"
    if freq_range is not None and rms_amplitude is not None:
        title += f"\nNoise Frequency Range: {freq_range[0]}-{freq_range[1]} Hz, RMS Amplitude: {rms_amplitude}"
    fig.suptitle(title)
    # Plot the Mel spectrogram on the first subplot
    axes[0].set_title("Mel Spectrogram")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].set_xticks(np.linspace(0, mel_spectrogram_resized.shape[1] - 1, 10))
    axes[0].set_xticklabels([f"{t:.2f}s" for t in np.linspace(0, duration, 10)])

    # Use Mel frequencies for y-axis labels
    y_ticks = np.linspace(0, mel_spectrogram_resized.shape[0] - 1, 10)
    selected_frequencies = np.linspace(0, len(mel_frequencies) - 1, 10).astype(int)
    axes[0].set_yticks(y_ticks)
    axes[0].set_yticklabels(
        [f"{mel_frequencies[idx]:.0f} Hz" for idx in selected_frequencies]
    )
    im = axes[0].imshow(
        mel_spectrogram_resized,
        aspect="auto",
        origin="lower",
        cmap="magma",
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=axes[0], shrink=0.8)
    cbar.set_label("Amplitude (dB)")

    # Plot the attribution map on the second subplot
    axes[1].imshow(
        converted_attr_map,
        aspect="auto",
        origin="lower",
        cmap="seismic",
        interpolation="none",
    )
    axes[1].set_title(f"DeepLIFTShap Attributions, Temporal Slicing = 1/{division}")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Frequency (Hz)")
    axes[1].set_xticks(np.linspace(0, converted_attr_map.shape[1] - 1, 10))
    axes[1].set_xticklabels([f"{t:.2f}s" for t in np.linspace(0, duration, 10)])
    y_ticks = np.linspace(0, converted_attr_map.shape[0] - 1, 10)
    selected_frequencies = np.linspace(0, len(mel_frequencies) - 1, 10).astype(int)
    axes[1].set_yticks(y_ticks)
    axes[1].set_yticklabels(
        [f"{mel_frequencies[idx]:.0f} Hz" for idx in selected_frequencies]
    )
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap="seismic"), ax=axes[1], shrink=0.8)
    cbar.set_label("Attribution Value")
    plt.tight_layout()
    save_plot(
        file_basename, division, noise_type, baseline_type, freq_range, rms_amplitude
    )
    plt.close(fig)
