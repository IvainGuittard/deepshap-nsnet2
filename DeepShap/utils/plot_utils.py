import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.ndimage import zoom
from utils.audio_features import compute_log_mel_spectrogram
from config.parameters import sample_rate, hop_length, n_fft
import seaborn as sns
from scipy.cluster.hierarchy import linkage


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


def plot_global_influence(h5_filename, input_basename, F_bins, T_frames):
    # A_in2mask[f_in, t_in] = Σ_{f0, t0} |all_attr[f0, t0, f_in, t_in]|
    print(f"Plotting global influence from {h5_filename}...")
    h5f = h5py.File(h5_filename, "r")
    A_in2mask = np.zeros((F_bins, T_frames), dtype=np.float32)
    for key in h5f:
        if key.startswith("time_division"):
            continue
        attr_map = h5f[key][:]
        A_in2mask += np.abs(attr_map)
    # Min–max normalize entire map so it’s in [0,1]
    A_in2mask_norm = (A_in2mask - A_in2mask.min()) / (
        A_in2mask.max() - A_in2mask.min() + 1e-12
    )

    plt.figure(figsize=(5, 4))
    plt.title("Global influence of each input TF‐bin on the entire mask (min–max norm)")
    plt.imshow(A_in2mask_norm, origin="lower", aspect="auto", cmap="magma")
    plt.xlabel("input time t_in")
    plt.ylabel("input freq f_in")

    plt.xticks(
        np.arange(0, T_frames, T_frames // 5),
        [
            f"{(t * hop_length) / sample_rate:.2f} s"
            for t in range(0, T_frames, T_frames // 5)
        ],
    )
    plt.yticks(
        np.arange(0, F_bins, F_bins // 10),
        [
            f"{f * sample_rate / (2 * F_bins):.0f} Hz"
            for f in range(0, F_bins, F_bins // 10)
        ],
    )

    plt.colorbar(label="normalized attribution")
    plt.tight_layout()

    os.makedirs(
        f"DeepShap/attributions/tf_attributions_collapsed/{input_basename}",
        exist_ok=True,
    )
    plt.savefig(
        f"DeepShap/attributions/tf_attributions_collapsed/{input_basename}/{input_basename}_global_influence.png",
        bbox_inches="tight",
    )
    plt.show()
    plt.close()
    print("Global influence plot saved.")
    h5f.close()


def plot_input_time_influence(h5_filename, input_basename, T_frames):
    # A_time[t0, t_in] = Σ_{f0, f_in} |all_attr[f0, t0, f_in, t_in]|
    print(f"Plotting input time influence from {h5_filename}...")
    save_path = f"DeepShap/attributions/tf_attributions_collapsed/{input_basename}/{input_basename}_time_influence.png"
    if os.path.exists(save_path):
        print(f"Input time influence plot already exists at {save_path}. Skipping.")
        return
    h5f = h5py.File(h5_filename, "r")
    A_time = np.zeros((T_frames, T_frames), dtype=np.float32)
    for key in h5f:
        if key.startswith("time_division"):
            continue
        f0, t0 = map(int, [key.split("_")[0][1:], key.split("_")[1][1:]])
        attr_map = h5f[key][:]
        A_time[t0] += np.abs(attr_map).sum(axis=0)
    A_time_norm = A_time / (A_time.sum(axis=1, keepdims=True) + 1e-12)

    plt.figure(figsize=(5, 4))
    plt.title("Normalized: How output‐time t0 depends on input‐time t_in")
    plt.imshow(A_time_norm, origin="lower", aspect="auto", cmap="viridis")
    plt.xlabel("input time t_in")
    plt.ylabel("output time t0")

    plt.xticks(
        np.arange(0, T_frames, T_frames // 5),
        [
            f"{(t * hop_length) / sample_rate:.2f} s"
            for t in range(0, T_frames, T_frames // 5)
        ],
    )
    plt.yticks(
        np.arange(0, T_frames, T_frames // 5),
        [
            f"{(t * hop_length) / sample_rate:.2f} s"
            for t in range(0, T_frames, T_frames // 5)
        ],
    )

    plt.colorbar(label="row‐normalized attribution")
    plt.tight_layout()
    os.makedirs(
        os.path.dirname(save_path),
        exist_ok=True,
    )
    plt.savefig(
        save_path,
        bbox_inches="tight",
    )
    plt.show()
    plt.close()
    print("Input time influence plot saved.")
    h5f.close()


def plot_input_freq_influence(h5_filename, input_basename, F_bins):
    # A_freq[f0, f_in] = Σ_{t0, t_in} |all_attr[f0, t0, f_in, t_in]|
    print(f"Plotting input frequency influence from {h5_filename}...")
    save_path = f"DeepShap/attributions/tf_attributions_collapsed/{input_basename}/{input_basename}_freq_influence.png"
    if os.path.exists(save_path):
        print(
            f"Input frequency influence plot already exists at {save_path}. Skipping."
        )
        return
    h5f = h5py.File(h5_filename, "r")
    A_freq = np.zeros((F_bins, F_bins), dtype=np.float32)
    for key in h5f:
        if key.startswith("time_division"):
            continue
        f0, t0 = map(int, [key.split("_")[0][1:], key.split("_")[1][1:]])
        attr_map = h5f[key][:]
        A_freq[f0] += np.abs(attr_map).sum(axis=1)
    # Normalize each row so sum over f_in = 1
    A_freq_norm = A_freq / (A_freq.sum(axis=1, keepdims=True) + 1e-12)

    plt.figure(figsize=(5, 4))
    plt.title("Normalized: How output‐freq f0 depends on input‐freq f_in")
    plt.imshow(A_freq_norm, origin="lower", aspect="auto", cmap="plasma")
    plt.xlabel("input freq f_in")
    plt.ylabel("output freq f0")

    plt.xticks(
        np.arange(0, F_bins, F_bins // 10),
        [
            f"{f * sample_rate / (2 * F_bins):.0f} Hz"
            for f in range(0, F_bins, F_bins // 10)
        ],
    )
    plt.yticks(
        np.arange(0, F_bins, F_bins // 10),
        [
            f"{f * sample_rate / (2 * F_bins):.0f} Hz"
            for f in range(0, F_bins, F_bins // 10)
        ],
    )

    plt.colorbar(label="row‐normalized attribution")
    plt.tight_layout()
    os.makedirs(
        os.path.dirname(save_path),
        exist_ok=True,
    )
    plt.savefig(
        save_path,
        bbox_inches="tight",
    )
    plt.show()
    plt.close()
    print("Input frequency influence plot saved.")
    h5f.close()


def plot_input_time_correlation(h5_filename, input_basename, T_frames):
    """
    Compute Pearson correlation between each pair of input time steps (t_in).
    Correlation is computed across attribution contexts summed over (f0, f_in).
    """
    print(f"Plotting input time correlation from {h5_filename}...")
    save_path = f"DeepShap/attributions/tf_attributions_collapsed/{input_basename}/{input_basename}_t_in_corr.png"
    if os.path.exists(save_path):
        print(f"Input time correlation plot already exists at {save_path}. Skipping.")
        return
    h5f = h5py.File(h5_filename, "r")
    time_vectors = np.zeros((T_frames, 0), dtype=np.float32)

    for key in h5f:
        if key.startswith("time_division"):
            continue
        attr = np.abs(h5f[key][:])  # shape: [f_in, t_in]
        attr_summed = attr.sum(axis=0)  # sum over f_in → [t_in]
        time_vectors = np.column_stack((time_vectors, attr_summed))  # shape: [t_in, N]

    corr_matrix = np.corrcoef(time_vectors)
    corr_matrix = np.nan_to_num(corr_matrix)

    plt.figure(figsize=(5, 4))
    plt.title("Input time–time correlation (Pearson)")
    plt.imshow(
        corr_matrix, origin="lower", aspect="auto", cmap="coolwarm", vmin=-1, vmax=1
    )
    plt.xlabel("Input time t_in")
    plt.ylabel("Input time t_in")

    plt.xticks(
        np.arange(0, T_frames, T_frames // 5),
        [
            f"{(t * hop_length) / sample_rate:.2f} s"
            for t in range(0, T_frames, T_frames // 5)
        ],
    )
    plt.yticks(
        np.arange(0, T_frames, T_frames // 5),
        [
            f"{(t * hop_length) / sample_rate:.2f} s"
            for t in range(0, T_frames, T_frames // 5)
        ],
    )

    plt.colorbar(label="Pearson correlation")
    plt.tight_layout()
    os.makedirs(
        os.path.dirname(save_path),
        exist_ok=True,
    )
    plt.savefig(
        save_path,
        bbox_inches="tight",
    )
    plt.close()
    print("Input time correlation plot saved.")
    h5f.close()
    return corr_matrix


def plot_input_freq_correlation(h5_filename, input_basename, F_bins):
    """
    Compute Pearson correlation between each pair of input frequencies (f_in).
    Correlation is computed across attribution contexts summed over (f0, t0, t_in).
    """
    print(f"Plotting input frequency correlation from {h5_filename}...")
    save_path = f"DeepShap/attributions/tf_attributions_collapsed/{input_basename}/{input_basename}_f_in_corr.png"
    if os.path.exists(save_path):
        print(
            f"Input frequency correlation plot already exists at {save_path}. Skipping."
        )
        return
    h5f = h5py.File(h5_filename, "r")
    corr_matrix_sum = np.zeros((F_bins, F_bins), dtype=np.float32)
    for key in h5f:
        if key.startswith("time_division"):
            continue
        attr = np.abs(h5f[key][:])  # shape: [f_in, t_in]
        attr_summed = attr.sum(axis=1)  # sum over t_in → [f_in]
        current_corr_matrix = np.corrcoef(attr_summed)
        current_corr_matrix = np.nan_to_num(current_corr_matrix)
        corr_matrix_sum += current_corr_matrix
    corr_matrix = corr_matrix_sum / len(h5f)
    corr_matrix = np.nan_to_num(corr_matrix)

    plt.figure(figsize=(5, 4))
    plt.title("Input frequency–frequency correlation (Pearson)")
    plt.imshow(
        corr_matrix, origin="lower", aspect="auto", cmap="coolwarm", vmin=-1, vmax=1
    )
    plt.xlabel("Input frequency f_in")
    plt.ylabel("Input frequency f_in")

    plt.xticks(
        np.arange(0, F_bins, F_bins // 10),
        [
            f"{f * sample_rate / (2 * F_bins):.0f} Hz"
            for f in range(0, F_bins, F_bins // 10)
        ],
    )
    plt.yticks(
        np.arange(0, F_bins, F_bins // 10),
        [
            f"{f * sample_rate / (2 * F_bins):.0f} Hz"
            for f in range(0, F_bins, F_bins // 10)
        ],
    )

    plt.colorbar(label="Pearson correlation")
    plt.tight_layout()
    os.makedirs(
        os.path.dirname(save_path),
        exist_ok=True,
    )
    plt.savefig(
        save_path,
        bbox_inches="tight",
    )
    plt.close()
    print("Input frequency correlation plot saved.")
    h5f.close()
    return corr_matrix


def plot_clustered_freq_correlation(h5_filename, input_basename, F_bins):
    """
    Compute Pearson correlation between input frequencies and visualize clusters of co-activated frequencies.
    The clustering is based on hierarchical linkage using (1 - Pearson correlation) as distance.
    """
    print(f"Clustering input frequency correlation from {h5_filename}...")
    corr_matrix = plot_input_freq_correlation(h5_filename, input_basename, F_bins)
    distance_matrix = 1 - corr_matrix
    np.fill_diagonal(distance_matrix, 0)
    sns.set(style="white")
    clustergrid = sns.clustermap(
        corr_matrix,
        row_cluster=True,
        col_cluster=True,
        row_linkage=linkage(distance_matrix, method="average"),
        col_linkage=linkage(distance_matrix, method="average"),
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        figsize=(6, 6),
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "Pearson correlation"},
    )

    os.makedirs(
        f"DeepShap/attributions/tf_attributions_collapsed/{input_basename}",
        exist_ok=True,
    )
    clustergrid.fig.suptitle("Clustered Input Frequency Correlation", y=1.02)
    clustergrid.savefig(
        f"DeepShap/attributions/tf_attributions_collapsed/{input_basename}/{input_basename}_f_in_corr_clustered.png",
        bbox_inches="tight",
    )
    plt.close()
    print("Clustered input frequency correlation plot saved.")
