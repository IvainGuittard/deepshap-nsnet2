import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from utils.audio_features import compute_log_mel_spectrogram
from config.parameters import sample_rate, freqs, hop_length


def plot_spectrogram_and_attributions(input_path, converted_attr_map, division):
    """
    Plots the Mel spectrogram and the DeepLIFTShap attribution map side by side.
    Saves the plot as a PNG file.
    Args:
        input_path (str): Path to the input audio file.
        converted_attr_map (np.ndarray): Attribution map with frequency bands on the y-axis and time bands on the x-axis.
    """
    mel_spectrogram = compute_log_mel_spectrogram(input_path, sample_rate=sample_rate)
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

    # Plot the Mel spectrogram on the first subplot
    axes[0].imshow(
        mel_spectrogram_resized,
        aspect="auto",
        origin="lower",
        cmap="magma",
        interpolation="nearest",
    )
    axes[0].set_title("Mel Spectrogram")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].set_xticks(np.linspace(0, mel_spectrogram_resized.shape[1] - 1, 10))
    axes[0].set_xticklabels([f"{t:.2f}s" for t in np.linspace(0, duration, 10)])

    y_ticks = np.linspace(0, mel_spectrogram_resized.shape[0] - 1, 10)
    y_tick_labels = [f"{freq:.0f} Hz" for freq in np.linspace(0, sample_rate / 2, 10)]
    axes[0].set_yticks(y_ticks)
    axes[0].set_yticklabels(y_tick_labels)
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
        interpolation="nearest",
    )
    axes[1].set_title(f"DeepLIFTShap Attributions, Temporal Slicing = 1/{division}")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Frequency (Hz)")
    axes[1].set_xticks(np.linspace(0, converted_attr_map.shape[1] - 1, 10))
    axes[1].set_xticklabels([f"{t:.2f}s" for t in np.linspace(0, duration, 10)])
    y_ticks = np.linspace(0, converted_attr_map.shape[0] - 1, 10)
    y_tick_labels = [f"{freq:.0f} Hz" for freq in np.linspace(freqs[0], freqs[-1], 10)]
    axes[1].set_yticks(y_ticks)
    axes[1].set_yticklabels(y_tick_labels)
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap="seismic"), ax=axes[1], shrink=0.8)
    cbar.set_label("Attribution Value")

    # Adjust layout and save the figure
    plt.tight_layout()
    folder_name = input_path.split("/")[-1].replace(".wav", "")
    file_name = input_path.split("/")[-1].replace(
        ".wav", f"_div{division}_attribution_plot.png"
    )
    print(f"Saving plot to {file_name} in folder {folder_name}")
    save_path = f"DeepShap/attributions/plots/{folder_name}/{file_name}"
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    plt.savefig(save_path)
    print(f"Subplot saved to {save_path}")


def plot_mel_spectrogram(file_path):
    """
    Plot the Mel spectrogram of an audio file.
    """
    mel_spectrogram = compute_log_mel_spectrogram(file_path)
    mel_spectrogram = mel_spectrogram.squeeze(0).numpy()
    num_frames = mel_spectrogram.shape[1]
    duration = num_frames * hop_length / sample_rate
    freqs = np.linspace(0, sample_rate / 2, mel_spectrogram.shape[0])

    plt.figure(figsize=(10, 4))
    plt.imshow(
        mel_spectrogram,
        aspect="auto",
        origin="lower",
        cmap="magma",
        interpolation="nearest",
    )
    plt.colorbar(format="%+2.0f dB", shrink=0.8)
    plt.title("Mel Spectrogram" + (" (Log)" if log_spectrogram else ""))
    plt.xlabel("Time (s)")
    plt.ylabel("Mel Frequency Bins")
    plt.xticks(
        ticks=np.linspace(0, num_frames - 1, 10),
        labels=[f"{t:.2f}s" for t in np.linspace(0, duration, 10)],
    )
    y_ticks = np.arange(freqs[0], freqs[-1] + 1, 1000)  # Espacement de 1000 Hz
    y_tick_labels = [f"{freq:.0f} Hz" for freq in y_ticks]
    plt.yticks(
        ticks=np.linspace(0, mel_spectrogram.shape[0] - 1, len(y_ticks)),
        labels=y_tick_labels,
    )
    plt.tight_layout()
    save_path = file_path.replace(".wav", "_mel_spectrogram.png")
    plt.savefig(save_path)
    print(f"Mel spectrogram saved to {save_path}")
    plt.show()
