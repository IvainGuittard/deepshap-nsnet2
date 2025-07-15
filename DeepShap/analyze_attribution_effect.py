import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchaudio
from matplotlib.patches import Patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DeepShap.utils.common_utils import load_and_resample
from DeepShap.utils.audio_pertubations import amplify_tf_bins
from DeepShap.utils.data_utils import get_wav_files
from DeepShap.utils.model_utils import compare_masks_from_audios, load_nsnet2_model
from DeepShap.utils.attribution_utils import (
    generate_top_percent_mask,
    generate_flop_percent_mask,
)
from DeepShap.config.parameters import sample_rate, hop_length


def analyse_top_attribution_effect(input_path, top_percent=10.0, dB_amplification=6.0):
    model, device = load_nsnet2_model()
    wav, _ = load_and_resample(input_path, target_sr=16000)

    top_percent_binary_mask_save_path = f"DeepShap/attributions/attribution_effects/top_{top_percent}_mask_{os.path.basename(input_path).replace('.wav', '')}.npy"
    if os.path.exists(top_percent_binary_mask_save_path):
        top_percent_binary_mask = np.load(top_percent_binary_mask_save_path)
        top_percent_binary_mask = torch.tensor(
            top_percent_binary_mask, dtype=torch.float32
        ).to(device)
    else:
        top_percent_binary_mask = generate_top_percent_mask(input_path, top_percent).to(
            device
        )
        np.save(
            top_percent_binary_mask_save_path, top_percent_binary_mask.cpu().numpy()
        )
    amplified_wav = amplify_tf_bins(wav, top_percent_binary_mask, dB_amplification).to(
        device
    )
    amplified_wav_path = input_path.replace(
        ".wav", f"_top_{top_percent}_amplified_{dB_amplification}dB.wav"
    )

    # Save the amplified audio
    torchaudio.save(
        amplified_wav_path, amplified_wav.cpu().unsqueeze(0), sample_rate=16000
    )
    print(f"Saved amplified audio to {amplified_wav_path}")

    log_mask_diff = compare_masks_from_audios(
        original_audio=wav,
        modified_audio=amplified_wav
    )

    F_bins, T_frames = log_mask_diff.size()
    plt.figure(figsize=(6, 4))
    plt.title(
        f"Top {top_percent}% Attribution Effect on {os.path.basename(input_path)}"
    )
    plt.imshow(
        log_mask_diff.cpu().numpy(),
        aspect="auto",
        origin="lower",
        cmap="seismic",
        interpolation="nearest",
        vmin=-np.max(np.abs(log_mask_diff.cpu().numpy())),
        vmax=np.max(np.abs(log_mask_diff.cpu().numpy())),
    )
    plt.colorbar(label="Log Mask Difference")
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

    input_basename = os.path.basename(input_path).replace(".wav", "")
    save_path = f"DeepShap/attributions/attribution_effects/{input_basename}_top_{top_percent}_{dB_amplification}dB_amplified_effect.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(
        save_path,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Saved top attributions effect plot to {save_path}")


def plot_top_percent_binary_mask(input_path, top_percent=10.0):

    top_percent_binary_mask_path = f"DeepShap/attributions/attribution_effects/top_{top_percent}_mask_{os.path.basename(input_path).replace('.wav', '')}.npy"
    if os.path.exists(top_percent_binary_mask_path):
        top_percent_binary_mask = np.load(top_percent_binary_mask_path)
    else:
        raise FileNotFoundError(
            f"Top percent binary mask not found at {top_percent_binary_mask_path}"
        )
    F_bins, T_frames = top_percent_binary_mask.shape
    plt.figure(figsize=(6, 4))
    plt.title(f"Top {top_percent}% Binary Mask for {os.path.basename(input_path)}")
    plt.imshow(
        top_percent_binary_mask,
        aspect="auto",
        origin="lower",
        cmap="gray",
        interpolation="nearest",
    )
    legend_elements = [
        Patch(
            facecolor="white",
            edgecolor="black",
            label=f"1: Top {top_percent}% Attribution",
        ),
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

    input_basename = os.path.basename(input_path).replace(".wav", "")
    save_path = f"DeepShap/attributions/attribution_effects/{input_basename}_top_{top_percent}_mask.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(
        save_path,
        bbox_inches="tight",
    )
    print(f"Saved top percent binary mask plot to {save_path}")
    plt.close()


def analyse_flop_attribution_effect(
    input_path, flop_percent=10.0, dB_amplification=6.0
):
    model, device = load_nsnet2_model()
    wav, _ = load_and_resample(input_path, target_sr=16000)

    flop_percent_binary_mask_save_path = f"DeepShap/attributions/attribution_effects/flop_{flop_percent}_mask_{os.path.basename(input_path).replace('.wav', '')}.npy"
    if os.path.exists(flop_percent_binary_mask_save_path):
        flop_percent_binary_mask = np.load(flop_percent_binary_mask_save_path)
        flop_percent_binary_mask = torch.tensor(
            flop_percent_binary_mask, dtype=torch.float32
        ).to(device)
    else:
        flop_percent_binary_mask = generate_flop_percent_mask(
            input_path, flop_percent
        ).to(device)
        np.save(
            flop_percent_binary_mask_save_path, flop_percent_binary_mask.cpu().numpy()
        )
    amplified_wav = amplify_tf_bins(wav, flop_percent_binary_mask, dB_amplification).to(
        device
    )
    amplified_wav_path = input_path.replace(
        ".wav", f"_flop_{flop_percent}_amplified_{dB_amplification}dB.wav"
    )

    # Save the amplified audio
    torchaudio.save(
        amplified_wav_path, amplified_wav.cpu().unsqueeze(0), sample_rate=16000
    )
    print(f"Saved amplified audio to {amplified_wav_path}")

    log_mask_diff = compare_masks_from_audios(
        original_audio=wav,
        modified_audio=amplified_wav,
    )

    F_bins, T_frames = log_mask_diff.size()
    plt.figure(figsize=(6, 4))
    plt.title(
        f"Flop {flop_percent}% Attribution Effect on {os.path.basename(input_path)}"
    )
    plt.imshow(
        log_mask_diff.cpu().numpy(),
        aspect="auto",
        origin="lower",
        cmap="seismic",
        interpolation="nearest",
        vmin=-np.max(np.abs(log_mask_diff.cpu().numpy())),
        vmax=np.max(np.abs(log_mask_diff.cpu().numpy())),
    )
    plt.colorbar(label="Log Mask Difference")
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

    input_basename = os.path.basename(input_path).replace(".wav", "")
    save_path = f"DeepShap/attributions/attribution_effects/{input_basename}_flop_{flop_percent}_{dB_amplification}dB_amplified_effect.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(
        save_path,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Saved flop attributions effect plot to {save_path}")


def plot_flop_percent_binary_mask(input_path, flop_percent=10.0):

    flop_percent_binary_mask_path = f"DeepShap/attributions/attribution_effects/flop_{flop_percent}_mask_{os.path.basename(input_path).replace('.wav', '')}.npy"
    if os.path.exists(flop_percent_binary_mask_path):
        flop_percent_binary_mask = np.load(flop_percent_binary_mask_path)
    else:
        raise FileNotFoundError(
            f"Flop percent binary mask not found at {flop_percent_binary_mask_path}"
        )
    F_bins, T_frames = flop_percent_binary_mask.shape
    plt.figure(figsize=(6, 4))
    plt.title(f"Flop {flop_percent}% Binary Mask for {os.path.basename(input_path)}")
    plt.imshow(
        flop_percent_binary_mask,
        aspect="auto",
        origin="lower",
        cmap="gray",
        interpolation="nearest",
    )
    legend_elements = [
        Patch(
            facecolor="white",
            edgecolor="black",
            label=f"1: Flop {flop_percent}% Attribution",
        ),
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

    input_basename = os.path.basename(input_path).replace(".wav", "")
    save_path = f"DeepShap/attributions/attribution_effects/{input_basename}_flop_{flop_percent}_mask.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(
        save_path,
        bbox_inches="tight",
    )
    print(f"Saved flop percent binary mask plot to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Folder with wav files"
    )
    parser.add_argument(
        "--top_percent",
        type=float,
        default=10.0,
        help="Percentage of top attributions to keep",
    )
    parser.add_argument(
        "--dB_amplification",
        type=float,
        default=6.0,
        help="dB amplification for top attributions",
    )

    args = parser.parse_args()

    wav_files = get_wav_files(args)

    for input_path in wav_files:
        analyse_top_attribution_effect(
            input_path, args.top_percent, args.dB_amplification
        )
        plot_top_percent_binary_mask(input_path, args.top_percent)
        analyse_flop_attribution_effect(
            input_path, args.top_percent, args.dB_amplification
        )
        plot_flop_percent_binary_mask(input_path, args.top_percent)


if __name__ == "__main__":
    main()
