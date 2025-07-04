import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
import torch
import h5py
from config.parameters import sample_rate, hop_length, n_fft
from utils.data_utils import load_and_resample
from utils.model_utils import load_nsnet2_model
import cv2
from tqdm import tqdm
import matplotlib as mpl
import io
from PIL import Image
from matplotlib.colors import LogNorm


def plot_global_influence(h5_filename, input_basename, F_bins, T_frames):
    # A_in2mask[f_in, t_in] = Σ_{f0, t0} |all_attr[f0, t0, f_in, t_in]|
    print(f"Plotting global influence from {h5_filename}...")
    save_path = f"DeepShap/attributions/tf_attributions_collapsed/{input_basename}/{input_basename}_global_influence.png"
    if os.path.exists(save_path):
        print(f"Global influence plot already exists at {save_path}. Skipping.")
        return
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

    plt.figure(figsize=(6, 4))
    plt.title(
        f"Global influence of each input TF‐bin on the entire mask (min–max norm) \n {input_basename}"
    )
    plt.imshow(
        A_in2mask_norm,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        norm=LogNorm(vmin=1e-4, vmax=A_in2mask_norm.max()),  # Logarithmic scale
    )
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
        os.path.dirname(save_path),
        exist_ok=True,
    )
    plt.savefig(
        save_path,
        bbox_inches="tight",
    )
    plt.show()
    plt.close()
    print("Global influence plot saved.")
    h5f.close()


def plot_input_time_influence(
    h5_filename, input_basename, T_frames, start_time=0, end_time=None
):
    # A_time[t0, t_in] = Σ_{f0, f_in} |all_attr[f0, t0, f_in, t_in]|
    print(f"Plotting input time influence from {h5_filename}...")
    save_path = f"DeepShap/attributions/tf_attributions_collapsed/{input_basename}/{input_basename}_time_influence_{start_time:.2f}_{end_time:.2f}.png"
    if os.path.exists(save_path):
        print(f"Input time influence plot already exists at {save_path}. Skipping.")
        return

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
        if t0 < start_frame or t0 >= end_frame:
            continue
        attr_map = h5f[key][:, start_frame:end_frame]
        A_time[t0 - start_frame] += np.abs(attr_map).sum(axis=0)
    A_time_norm = A_time / (A_time.sum(axis=1, keepdims=True) + 1e-12)

    plt.figure(figsize=(6, 4))
    plt.title(
        f"Normalized from {start_time:.2f}s to {end_time:.2f}s: How output‐time t0 depends on input‐time t_in \n {input_basename}"
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

    plt.figure(figsize=(6, 4))
    plt.title(
        f"Normalized: How output‐freq f0 depends on input‐freq f_in \n {input_basename}"
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
    print("Input frequency influence plot saved.")
    h5f.close()


def plot_input_low_freq_influence(
    h5_filename, input_basename, F_bins, high_freq_cutoff=1000
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
    keys = tqdm(h5f.keys(), desc="Processing keys")
    time_vectors = np.zeros((T_frames, len(keys)), dtype=np.float32)
    for idx, key in enumerate(keys):
        keys.set_description(f"Processing key: {key}")
        if key.startswith("time_division"):
            continue
        attr = np.abs(h5f[key][:])  # shape: [f_in, t_in]
        attr_summed = attr.sum(axis=0)  # sum over f_in → [t_in]
        time_vectors[:, idx] = attr_summed

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

    plt.figure(figsize=(6, 4))
    plt.title(f"Input frequency–frequency correlation (Pearson) \n {input_basename}")
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


def plot_stft_spectrogram(input_path, input_basename):
    """
    Plot the STFT spectrogram of the input audio file.
    """
    model, device = load_nsnet2_model()
    input, _ = load_and_resample(input_path, sample_rate)
    input = input.to(device)

    input_spec_complex = model.preproc(input)
    input_logpower = (
        torch.log(input_spec_complex.abs() ** 2 + model.eps).squeeze(0).squeeze(0)
    )
    F_bins, T_frames = input_logpower.shape[-2:]

    plt.figure(figsize=(6, 4))
    plt.title(f"STFT Spectrogram of {input_basename} log-power")
    plt.imshow(
        input_logpower.cpu().numpy(),
        origin="lower",
        aspect="auto",
        cmap="magma",
        interpolation="none",
    )
    plt.colorbar(label="Magnitude")
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

    os.makedirs(f"DeepShap/attributions/spectrograms", exist_ok=True)
    plt.savefig(
        f"DeepShap/attributions/spectrograms/{input_basename}_stft_spectrogram.png",
        bbox_inches="tight",
    )
    plt.close()


def make_time_normalized_video_from_attributions(
    h5_filename, input_basename, F_bins, T_frames, fps=5
):
    """
    Create a video from per-frame attributions using OpenCV.
    Each frame shows the influence of all input TF-bins on one output (f0, t0) bin.
    """
    print(f"Generating attribution video from {h5_filename}...")
    h5f = h5py.File(h5_filename, "r")

    frame_width, frame_height = T_frames, F_bins
    video_path = f"DeepShap/attributions/videos/{input_basename}_time_normalized.mp4"
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    color_bar = create_colorbar(frame_height, cmap="magma")
    color_bar_width = color_bar.shape[1]
    out = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width + color_bar_width, frame_height),
    )

    frame_keys = sorted(
        [k for k in h5f if not k.startswith("time_division")],
        key=lambda x: int(x.split("_")[1][1:]),  # Sort by t0
    )
    A_in2mask = np.zeros((F_bins, T_frames), dtype=np.float32)

    last_frame_t0 = 0
    for idx, key in enumerate(tqdm(frame_keys, desc="Processing frames")):
        current_t0 = int(key.split("_")[1][1:])
        attr_map = h5f[key][:]  # shape: [f_in, t_in]
        A_in2mask += np.abs(attr_map)
        if current_t0 != last_frame_t0:
            # If we have a new t0, normalize and save the frame
            A_in2mask_norm = (A_in2mask - A_in2mask.min()) / (
                A_in2mask.max() - A_in2mask.min() + 1e-12
            )
            A_in2mask_norm = np.nan_to_num(A_in2mask_norm, nan=1)

            time_coefficient_matrix = np.ones((F_bins, T_frames), dtype=np.float32)
            for t in range(current_t0 + 1):
                time_coefficient_matrix[:, t] = current_t0 - t + 1
            A_in2mask_uint8 = np.uint8(
                255 * A_in2mask_norm / (time_coefficient_matrix + 1e-12)
            )
            frame_rgb = cv2.applyColorMap(A_in2mask_uint8, cv2.COLORMAP_MAGMA)
            frame_rgb = cv2.resize(
                frame_rgb, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST
            )
            frame_rgb = np.hstack((frame_rgb, color_bar))
            out.write(frame_rgb)
            last_frame_t0 = current_t0

    out.release()
    h5f.close()
    print(f"Attribution video saved to {video_path}")


def make_video_from_attributions(h5_filename, input_basename, F_bins, T_frames, fps=5):
    """
    Create a video from per-frame attributions using OpenCV.
    Each frame shows the influence of all input TF-bins on one output (f0, t0) bin.
    """
    print(f"Generating attribution video from {h5_filename}...")
    h5f = h5py.File(h5_filename, "r")

    frame_width, frame_height = T_frames, F_bins
    video_path = f"DeepShap/attributions/videos/{input_basename}_frame_by_frame.mp4"
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    color_bar = create_colorbar(frame_height, cmap="magma")
    color_bar_width = color_bar.shape[1]
    out = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width + color_bar_width, frame_height),
    )

    frame_keys = sorted(
        [k for k in h5f if not k.startswith("time_division")],
        key=lambda x: int(x.split("_")[1][1:]),  # Sort by t0
    )
    A_in2mask = np.zeros((F_bins, T_frames), dtype=np.float32)

    last_frame_t0 = 0
    for idx, key in enumerate(tqdm(frame_keys, desc="Processing frames")):
        current_t0 = int(key.split("_")[1][1:])
        attr_map = h5f[key][:]  # shape: [f_in, t_in]
        A_in2mask += np.abs(attr_map)
        if current_t0 != last_frame_t0:
            # If we have a new t0, normalize and save the frame
            A_in2mask_norm = (A_in2mask - A_in2mask.min()) / (
                A_in2mask.max() - A_in2mask.min() + 1e-12
            )
            A_in2mask_norm = np.nan_to_num(A_in2mask_norm, nan=1)

            time_coefficient_matrix = np.ones((F_bins, T_frames), dtype=np.float32)
            for t in range(current_t0 + 1):
                time_coefficient_matrix[:, t] = current_t0 - t + 1
            A_in2mask_uint8 = np.uint8(
                255 * A_in2mask_norm / (time_coefficient_matrix + 1e-12)
            )
            frame_rgb = cv2.applyColorMap(A_in2mask_uint8, cv2.COLORMAP_MAGMA)
            frame_rgb = cv2.resize(
                frame_rgb, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST
            )
            frame_rgb = np.hstack((frame_rgb, color_bar))
            out.write(frame_rgb)
            last_frame_t0 = current_t0

            A_in2mask.fill(0)  # Reset for the next t0

    out.release()
    h5f.close()
    print(f"Attribution video saved to {video_path}")


def make_frame_grouped_video_from_attributions(
    h5_filename, input_basename, F_bins, T_frames, fps=5, frame_grouping=10
):
    """
    Create a video from per-frame attributions using OpenCV.
    Each frame shows the influence of all input TF-bins on one output (f0, t0) bin.
    """
    print(f"Generating attribution video from {h5_filename}...")
    h5f = h5py.File(h5_filename, "r")

    frame_width, frame_height = T_frames, F_bins
    video_path = f"DeepShap/attributions/videos/{input_basename}_frame_grouped_by_{frame_grouping}.mp4"
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    color_bar = create_colorbar(frame_height, cmap="magma")
    color_bar_width = color_bar.shape[1]
    out = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width + color_bar_width, frame_height),
    )

    frame_keys = sorted(
        [k for k in h5f if not k.startswith("time_division")],
        key=lambda x: int(x.split("_")[1][1:]),  # Sort by t0
    )
    A_in2mask = np.zeros((F_bins, T_frames), dtype=np.float32)

    last_frame_t0 = 0
    for idx, key in enumerate(tqdm(frame_keys, desc="Processing frames")):
        current_t0 = int(key.split("_")[1][1:])
        attr_map = h5f[key][:]  # shape: [f_in, t_in]
        A_in2mask += np.abs(attr_map)
        if current_t0 == last_frame_t0 + frame_grouping:
            # If we have a new t0, normalize and save the frame
            A_in2mask_norm = (A_in2mask - A_in2mask.min()) / (
                A_in2mask.max() - A_in2mask.min() + 1e-12
            )
            A_in2mask_norm = np.nan_to_num(A_in2mask_norm, nan=1)

            time_coefficient_matrix = np.ones((F_bins, T_frames), dtype=np.float32)
            for t in range(current_t0 + 1):
                time_coefficient_matrix[:, t] = current_t0 - t + 1
            A_in2mask_uint8 = np.uint8(
                255 * A_in2mask_norm / (time_coefficient_matrix + 1e-12)
            )
            frame_rgb = cv2.applyColorMap(A_in2mask_uint8, cv2.COLORMAP_MAGMA)
            frame_rgb = cv2.resize(
                frame_rgb, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST
            )
            frame_rgb = np.hstack((frame_rgb, color_bar))
            out.write(frame_rgb)
            last_frame_t0 = current_t0

            A_in2mask.fill(0)  # Reset for the next t0

    out.release()
    h5f.close()
    print(f"Attribution video saved to {video_path}")


def create_colorbar(height, cmap="magma"):
    fig, ax = plt.subplots(figsize=(0.5, 3), dpi=100)
    fig.subplots_adjust(left=0.5, right=0.6, top=1, bottom=0)

    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation="vertical")
    cb.set_label("Attribution")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    colorbar_img = Image.open(buf).convert("RGB").resize((40, height))
    return np.array(colorbar_img)


