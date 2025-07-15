from matplotlib.colors import LogNorm
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from DeepShap.models.MaskEstimationBlock import MaskEstimationBlock
from DeepShap.models.NsNet2_model import NsNet2
from DeepShap.utils.common_utils import load_and_resample
import torchaudio
from DeepShap.config.parameters import sample_rate, n_fft, hop_length
import numpy as np
import matplotlib.pyplot as plt


def load_nsnet2_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = NsNet2(n_fft=512, n_feat=257, hd1=400, hd2=400, hd3=600)
    # Load the pre-trained weights
    weights_path = "DeepShap/models/nsnet2_baseline.bin"
    state_dict = torch.load(weights_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device


def generate_and_plot_mask(wav_path):
    """
    Compute and plot NSNet2 mask from a waveform.

    Args:
        wav_path (str): Path to .wav file
    """
    model, device = load_nsnet2_model()
    wrapper = MaskEstimationBlock(model).to(device).train()
    wav, _ = load_and_resample(wav_path, target_sr=16000)
    wav = wav.to(device)
    stft = torchaudio.transforms.Spectrogram(n_fft=n_fft, power=None).to(device)
    spec = stft(wav).squeeze(0)  # [F, T]
    log_power = torch.log(spec.abs() ** 2 + 1e-8).to(device).unsqueeze(0)
    with torch.no_grad():
        output_mask = wrapper(log_power).squeeze(0).squeeze(0)  # [F, T]
    F_bins, T_frames = output_mask.size()
    plt.figure(figsize=(6, 4))
    plt.title(f"Zero Baseline Mask Output")
    plt.imshow(
        output_mask.cpu().numpy(),
        aspect="auto",
        origin="lower",
        cmap="magma",
        interpolation="nearest",
        norm=LogNorm(vmin=1e-2, vmax=1),
    )
    plt.colorbar(label="Value")
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
    """# Calculate statistics
    mask_max = output_mask.max().item()
    mask_min = output_mask.min().item()
    mask_mean = output_mask.mean().item()
    mask_var = output_mask.var().item()

    # Add statistics under the plot
    plt.figtext(
        0.5,
        -0.1,
        f"Max: {mask_max:.4f}, Min: {mask_min:.4f},\nMean: {mask_mean:.4f}, Variance: {mask_var:.4f}",
        wrap=True,
        horizontalalignment="center",
        fontsize=10,
    )"""

    wav_basename = os.path.basename(wav_path).replace(".wav", "")
    save_path = f"DeepShap/{wav_basename}_mask_output.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(
        save_path,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Saved attribution effect plot to {save_path}")

def generate_mask_from_audio(wav):
    """
    Compute NSNet2 mask from a waveform.

    Args:
        wav (torch.Tensor): Waveform

    Returns:
        mask (torch.Tensor): Shape [F, T], values in [0,1]
    """
    model, device = load_nsnet2_model()
    wrapper = MaskEstimationBlock(model).to(device).train()
    wav = wav.to(device)
    stft = torchaudio.transforms.Spectrogram(n_fft=n_fft, power=None).to(device)
    spec = stft(wav).squeeze(0)  # [F, T]
    log_power = torch.log(spec.abs() ** 2 + 1e-8).to(device).unsqueeze(0)
    with torch.no_grad():
        return wrapper(log_power).squeeze(0).squeeze(0)  # [F, T]


def compare_masks_from_audios(original_audio, modified_audio):
    """
    Compute and compare masks generated from clean and noisy audio files.
    Args:
        original_audio (str): Path to the clean audio file
        modified_audio (str): Path to the noisy audio file

    Returns:
        mask_clean (torch.Tensor): Mask from clean input, shape [1, 1, F, T]
        mask_noisy (torch.Tensor): Mask from noisy input, shape [1, 1, F, T]
        diff (torch.Tensor): Absolute difference between masks
    """
    original_mask = generate_mask_from_audio(original_audio)
    modified_mask = generate_mask_from_audio(modified_audio)
    log_mask_diff = torch.log(
        (modified_mask.abs() + 1e-8) / (original_mask.abs() + 1e-8)
    )
    return log_mask_diff


def plot_zero_baseline_mask(input_path):
    model, device = load_nsnet2_model()
    wrapper = MaskEstimationBlock(model).to(device).train()
    wav, _ = load_and_resample(input_path, target_sr=16000)
    wav = wav.to(device)
    stft = torchaudio.transforms.Spectrogram(n_fft=512, power=None).to(device)
    spec = stft(wav).squeeze(0).squeeze(0)  # [F, T]
    F_bins, T_frames = spec.shape
    zero_baseline_logpower = torch.log(
        torch.full((1, F_bins, T_frames), fill_value=model.eps, device="cuda")
    )
    with torch.no_grad():
        output_mask = wrapper(zero_baseline_logpower).squeeze(0).squeeze(0)

    F_bins, T_frames = output_mask.size()
    plt.figure(figsize=(6, 4))
    plt.title(f"Zero Baseline Mask Output")
    plt.imshow(
        output_mask.cpu().numpy(),
        aspect="auto",
        origin="lower",
        cmap="magma",
        interpolation="nearest",
        norm=LogNorm(vmin=1e-2, vmax=1),
    )
    plt.colorbar(label="Value")
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
    # Calculate statistics
    mask_max = output_mask.max().item()
    mask_min = output_mask.min().item()
    mask_mean = output_mask.mean().item()
    mask_var = output_mask.var().item()

    # Add statistics under the plot
    plt.figtext(
        0.5,
        -0.1,
        f"Max: {mask_max:.4f}, Min: {mask_min:.4f},\nMean: {mask_mean:.4f}, Variance: {mask_var:.4f}",
        wrap=True,
        horizontalalignment="center",
        fontsize=10,
    )

    save_path = f"DeepShap/zero_baseline_mask_output.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(
        save_path,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Saved attribution effect plot to {save_path}")
    print(
        f"Max: {mask_max:.4f}, Min: {mask_min:.4f}, Mean: {mask_mean:.4f}, Variance: {mask_var:.4f}"
    )
