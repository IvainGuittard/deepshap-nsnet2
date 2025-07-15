import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import numpy as np
import matplotlib.pyplot as plt
import torch
import h5py
from DeepShap.config.parameters import sample_rate, n_fft, hop_length
from DeepShap.utils.common_utils import load_and_resample
from DeepShap.utils.audio_features import compute_binary_speech_mask, compute_snr_map
from DeepShap.utils.model_utils import load_nsnet2_model
from tqdm import tqdm
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch


def plot_global_influences_separately(h5_filename, input_basename, F_bins, T_frames):
    print(
        f"Plotting global, positive, and negative influence maps from {h5_filename}..."
    )

    # Load attribution tensor and collapse across output
    h5f = h5py.File(h5_filename, "r")
    A_total = np.zeros((F_bins, T_frames), dtype=np.float32)
    for key in h5f:
        if key.startswith("time_division"):
            continue
        A_total += h5f[key][:]
    h5f.close()

    # Prepare all three maps
    A_abs = np.abs(A_total)
    A_pos = np.clip(A_total, 0, None)
    A_neg = -np.clip(A_total, None, 0)

    maps = {
        "global_influence": (
            A_abs,
            f"Log-scaled normalized absolute attributions (log₁₀(|attr|))",
            "viridis",
        ),
        "global_positive_influence": (
            A_pos,
            f"Log-scaled normalized positive attributions (log₁₀(attr))",
            "viridis",
        ),
        "global_negative_influence": (
            A_neg,
            f"Log-scaled normalized negative attributions (log₁₀(–attr))",
            "cividis",
        ),
    }

    time_ticks = np.arange(0, T_frames, T_frames // 5)
    time_labels = [f"{(t * hop_length) / sample_rate:.2f} s" for t in time_ticks]
    freq_ticks = np.arange(0, F_bins, F_bins // 10)
    freq_labels = [f"{f * sample_rate / (2 * F_bins):.0f} Hz" for f in freq_ticks]
    global_max = A_abs.max()

    for name, (A_map, title, cmap) in maps.items():
        A_norm = A_map / (global_max + 1e-12)

        plt.figure(figsize=(6, 4))
        plt.title(f"{title}\n{input_basename}")
        im = plt.imshow(
            A_norm,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            norm=LogNorm(vmin=1e-2, vmax=1e0),  # Logarithmic scale
        )
        plt.xlabel("input time t_in")
        plt.ylabel("input freq f_in")
        plt.xticks(time_ticks, time_labels)
        plt.yticks(freq_ticks, freq_labels)
        plt.colorbar(im, label="normalized attribution (log scale)")
        plt.tight_layout()

        save_path = f"DeepShap/attributions/tf_attributions_collapsed/{input_basename}/{input_basename}_{name}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.show()
        plt.close()
        print(f"{title} plot saved to {save_path}")


def plot_input_time_influence(
    h5_filename, input_basename, T_frames, start_time=0, end_time=None
):
    # A_time[t0, t_in] = Σ_{f0, f_in} |all_attr[f0, t0, f_in, t_in]|
    print(f"Plotting input time influence from {h5_filename}...")
    if start_time == 0 and end_time is None:
        save_path = f"DeepShap/attributions/tf_attributions_collapsed/{input_basename}/{input_basename}_time_influence.png"
    else:
        save_path = f"DeepShap/attributions/tf_attributions_collapsed/{input_basename}/{input_basename}_time_influence_{start_time:.2f}_{end_time:.2f}.png"
    # if os.path.exists(save_path):
    #     print(f"Input time influence plot already exists at {save_path}. Skipping.")
    #     return

    if end_time is None or end_time > T_frames * hop_length / sample_rate:
        end_frame = T_frames
    else:
        end_frame = int(end_time * sample_rate / hop_length)
    start_frame = int(start_time * sample_rate / hop_length)
    plot_length = end_frame - start_frame

    h5f = h5py.File(h5_filename, "r")
    A_time = np.zeros((plot_length, plot_length), dtype=np.float32)
    for key in h5f:
        if key.startswith("time_division"):
            continue
        f0, t0 = map(int, [key.split("_")[0][1:], key.split("_")[1][1:]])
        if t0 < start_frame or (end_time is not None and t0 >= end_frame):
            continue
        attr_map = h5f[key][:, start_frame:end_frame]
        A_time[t0 - start_frame] += np.abs(attr_map).sum(axis=0)
    # Normalize each row so sum over t_in = 1
    A_time_norm = A_time / (A_time.sum(axis=1, keepdims=True) + 1e-12)

    plt.figure(figsize=(6, 4))
    if end_time is None:
        plt.title(
            f"Normalized from {start_time:.2f}s to end: \n How output‐time t0 depends on input‐time t_in \n {input_basename}"
        )
    else:
        plt.title(
            f"Normalized from {start_time:.2f}s to {end_time:.2f}s: \n How output‐time t0 depends on input‐time t_in \n {input_basename}"
        )

    plt.imshow(
        A_time_norm,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        norm=LogNorm(vmin=1e-4, vmax=A_time_norm.max()),  # Logarithmic scale
    )

    plt.xlabel("input time t_in")
    plt.ylabel("output time t0")

    plt.xticks(
        np.arange(0, plot_length, plot_length // 5),
        [
            f"{(t * hop_length) / sample_rate:.2f} s"
            for t in range(0, plot_length, plot_length // 5)
        ],
        rotation=45,
    )
    plt.yticks(
        np.arange(0, plot_length, plot_length // 5),
        [
            f"{(t * hop_length) / sample_rate:.2f} s"
            for t in range(0, plot_length, plot_length // 5)
        ],
    )

    plt.colorbar(label="row‐normalized attribution (log scale)")
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
    print(f"Input time influence plot saved to {save_path}.")
    h5f.close()


def plot_input_freq_influence(
    h5_filename, input_basename, F_bins, start_time=0, end_time=None
):
    # A_freq[f0, f_in] = Σ_{t0, t_in} |all_attr[f0, t0, f_in, t_in]|
    print(f"Plotting input frequency influence from {h5_filename}...")
    if start_time == 0 and end_time is None:
        save_path = f"DeepShap/attributions/tf_attributions_collapsed/{input_basename}/{input_basename}_freq_influence.png"
    else:
        save_path = f"DeepShap/attributions/tf_attributions_collapsed/{input_basename}/{input_basename}_freq_influence_{start_time:.2f}_{end_time:.2f}.png"
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
        if t0 < start_time or (end_time is not None and t0 >= end_time):
            continue
        attr_map = h5f[key][:]
        A_freq[f0] += np.abs(attr_map).sum(axis=1)
    # Normalize each row so sum over f_in = 1
    A_freq_norm = A_freq / (A_freq.sum(axis=1, keepdims=True) + 1e-12)

    plt.figure(figsize=(6, 4))
    if end_time is None:
        plt.title(
            f"Normalized from {start_time:.2f}s to end: \n How output‐freq f0 depends on input‐freq f_in \n {input_basename}"
        )
    else:
        plt.title(
            f"Normalized from {start_time:.2f}s to {end_time:.2f}s: \n How output‐freq f0 depends on input‐freq f_in \n {input_basename}"
        )
    plt.imshow(A_freq_norm, origin="lower", aspect="auto", cmap="plasma")
    plt.xlabel("input freq f_in")
    plt.ylabel("output freq f0")

    plt.xticks(
        np.arange(0, F_bins, F_bins // 5),
        [
            f"{f * sample_rate / (2 * F_bins):.0f} Hz"
            for f in range(0, F_bins, F_bins // 5)
        ],
        rotation=45,
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
    print(f"Input frequency influence plot saved to {save_path}.")
    h5f.close()


def plot_input_low_freq_influence(
    h5_filename,
    input_basename,
    F_bins,
    high_freq_cutoff=1000,
    start_time=0,
    end_time=None,
):
    """
    Compute and plot input frequency influence for a specific frequency window.
    A_freq_window[f0, f_in] = Σ_{t0, t_in} |all_attr[f0, t0, f_in, t_in]| within the window.

    Args:
        h5_filename (str): Path to the HDF5 file containing attribution data.
        input_basename (str): Base name for saving the plot.
        F_bins (int): Total number of frequency bins.
        freq_window (tuple): Frequency window as (start_freq, end_freq) in Hz.
    """
    print(
        f"Plotting input frequency influence for low frequencies from {h5_filename}..."
    )
    save_path = f"DeepShap/attributions/tf_attributions_collapsed/{input_basename}/{input_basename}_low_freq_influence_{high_freq_cutoff}_{start_time:.2f}_{end_time:.2f}.png"
    if start_time == 0 and end_time is None:
        save_path = f"DeepShap/attributions/tf_attributions_collapsed/{input_basename}/{input_basename}_low_freq_influence_{high_freq_cutoff}.png"
    if os.path.exists(save_path):
        print(
            f"Input frequency influence plot for window already exists at {save_path}. Skipping."
        )
        return

    h5f = h5py.File(h5_filename, "r")

    # Compute frequency bin indices for the window
    end_bin = int(high_freq_cutoff * 2 * F_bins / sample_rate)
    A_freq_window = np.zeros((F_bins, end_bin), dtype=np.float32)

    for key in h5f:
        if key.startswith("time_division"):
            continue
        f0, t0 = map(int, [key.split("_")[0][1:], key.split("_")[1][1:]])
        if t0 < start_time or (end_time is not None and t0 >= end_time):
            continue
        attr_map = h5f[key][:]  # shape: [f_in, t_in]
        # Sum only within the frequency window
        A_freq_window[f0] += np.abs(attr_map[:end_bin]).sum(axis=1)

    # Normalize each row so sum over f_in = 1
    A_freq_window_norm = A_freq_window / (
        A_freq_window.sum(axis=1, keepdims=True) + 1e-12
    )

    plt.figure(figsize=(6, 4))
    plt.title(
        f"Normalized: How output‐freq f0 depends on input‐freq f_in \n {input_basename} (low frequencies up to {high_freq_cutoff} Hz)"
    )
    plt.imshow(A_freq_window_norm, origin="lower", aspect="auto", cmap="plasma")
    plt.xlabel("input freq f_in")
    plt.ylabel("output freq f0")

    plt.xticks(
        np.arange(0, end_bin, end_bin // 5),
        [
            f"{f * sample_rate / (2 * F_bins):.0f} Hz"
            for f in range(0, end_bin, end_bin // 5)
        ],
        rotation=45,
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
    print(
        f"Input frequency influence plot for low frequencies up to {high_freq_cutoff} Hz saved."
    )
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
    all_keys = [k for k in h5f.keys() if not k.startswith("time_division")]

    time_vectors = np.zeros((T_frames, T_frames), dtype=np.float32)

    for i, t0_val in enumerate(tqdm(range(T_frames), desc="Processing t0 values")):
        keys_for_t0 = [k for k in all_keys if int(k.split("_")[1][1:]) == t0_val]
        attr_sum_for_t0 = np.zeros(T_frames, dtype=np.float32)
        for k in keys_for_t0:
            attr = np.abs(h5f[k][:])  # shape [f_in, t_in]
            attr_summed = attr.sum(axis=0)  # sum over f_in → [t_in]
            attr_sum_for_t0 += attr_summed
        time_vectors[:, i] = attr_sum_for_t0

    corr_matrix = np.corrcoef(time_vectors)
    corr_matrix = np.nan_to_num(corr_matrix)

    plt.figure(figsize=(6, 4))
    plt.title(f"Input time–time correlation (Pearson) \n {input_basename}")
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
    print("Input time correlation plot saved to {save_path}.")
    h5f.close()
    return corr_matrix


def plot_stft_spectrogram(input_path, input_basename):
    """
    Plot the STFT spectrogram of the input audio file.
    """
    model, device = load_nsnet2_model()
    input, _ = load_and_resample(input_path, sample_rate)
    input = input.to(device)

    input_spec_complex = model.preproc(input)
    input_power = (input_spec_complex.abs() ** 2).squeeze(0).squeeze(0)
    input_db = 10 * torch.log10(input_power + model.eps)
    F_bins, T_frames = input_db.shape[-2:]

    plt.figure(figsize=(6, 4))
    plt.title(f"STFT Spectrogram of {input_basename}")
    plt.imshow(
        input_db.cpu().numpy(),
        origin="lower",
        aspect="auto",
        cmap="magma",
        interpolation="none",
    )
    plt.colorbar(label="Log Power (dB)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")

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

    save_path = (
        f"DeepShap/attributions/spectrograms/{input_basename}_stft_spectrogram.png"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(
        save_path,
        bbox_inches="tight",
    )
    print(f"STFT spectrogram saved to {save_path}")
    plt.close()


def plot_snr_mask(clean_path, noisy_path):
    snr_mask = compute_snr_map(clean_path, noisy_path, sample_rate)
    noisy_name = os.path.basename(noisy_path).replace(".wav", "")

    model, device = load_nsnet2_model()
    input, _ = load_and_resample(clean_path, sample_rate)
    input = input.to(device)

    input_spec_complex = model.preproc(input)
    input_logpower = (
        torch.log(input_spec_complex.abs() ** 2 + model.eps).squeeze(0).squeeze(0)
    )
    F_bins, T_frames = input_logpower.shape[-2:]

    plt.figure(figsize=(6, 4))
    plt.title(f"SNR Mask of {noisy_name}")
    plt.imshow(
        snr_mask,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        interpolation="none",  # Logarithmic scale
        vmin=-np.max(np.abs(snr_mask)),
        vmax=np.max(np.abs(snr_mask)),
    )
    plt.colorbar(label="SNR Mask (log scale)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")

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
    save_path = f"DeepShap/attributions/snr_masks/{noisy_name}_snr_mask.png"
    plt.savefig(save_path, bbox_inches="tight")
    print(f"SNR mask saved to {save_path}")
    plt.close()


def plot_binary_speech_mask(clean_path):
    binary_mask = compute_binary_speech_mask(clean_path, sample_rate)
    file_name = os.path.basename(clean_path).replace(".wav", "").replace("clean_", "")

    model, device = load_nsnet2_model()
    input, _ = load_and_resample(clean_path, sample_rate)
    input = input.to(device)

    input_spec_complex = model.preproc(input)
    input_logpower = (
        torch.log(input_spec_complex.abs() ** 2 + model.eps).squeeze(0).squeeze(0)
    )
    F_bins, T_frames = input_logpower.shape[-2:]

    plt.figure(figsize=(6, 4))
    plt.title(f"SNR Mask of {file_name}")
    plt.imshow(
        binary_mask,
        origin="lower",
        aspect="auto",
        cmap="gray",
        interpolation="none",
    )
    legend_elements = [
        Patch(facecolor="black", edgecolor="black", label="0: Noise Dominant"),
        Patch(facecolor="white", edgecolor="black", label="1: Speech Dominant"),
    ]
    plt.legend(
        handles=legend_elements,
        loc="lower right",
        frameon=True,
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")

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
    plt.savefig(
        f"DeepShap/attributions/speech_binary_masks/{file_name}_binary_mask.png",
    )
    plt.close()


if __name__ == "__main__":
    """
    Script plotting snr mask, stft spectrogram, and binary speech mask.
    """
    clean_path = "data/clean_input_tests_cut/clean_p227_357_cut_0_2.wav" # path to clean audio file for SNR mask
    noisy_path = "data/added_white_noise_input/p227_357_cut_0_2_white_noise_0.0-0.8_amplitude_0.01.wav" # path to noisy audio file for SNR mask
    plot_snr_mask(clean_path, noisy_path)

    input_dir = "data/clean_input_tests" # input directory or single wav file for spectrogram and binary mask plots
    if os.path.isdir(input_dir):
        wav_files = [
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.endswith(".wav")
        ]
    elif os.path.isfile(input_dir) and input_dir.endswith(".wav"):
        wav_files = [input_dir]
    for wav_file in wav_files:
        input_basename = os.path.basename(wav_file).replace(".wav", "")
        plot_stft_spectrogram(wav_file, input_basename)
        plot_binary_speech_mask(wav_file)
