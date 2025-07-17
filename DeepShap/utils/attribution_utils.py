import torch
import os
import h5py
import numpy as np
from DeepShap.utils.common_utils import load_and_resample
from utils.model_utils import load_nsnet2_model


def load_attributions_from_h5(wav_path):
    """
    Load the aggregated attribution map from the h5 file corresponding to the input wav.

    Returns:
        attributions (Tensor): Attribution map of shape [F, T] on the correct device.
    """
    model, device = load_nsnet2_model()
    wav, _ = load_and_resample(wav_path, target_sr=16000)
    wav = wav.to(device)
    spec = model.preproc(wav)
    F_bins, T_frames = spec.shape[-2:]

    input_basename = os.path.basename(wav_path).replace(".wav", "")
    h5_filename = f"DeepShap/attributions/tf_attributions_h5py/{input_basename}_attributions.h5"
    with h5py.File(h5_filename, "r") as h5f:
        A_total = np.zeros((F_bins, T_frames), dtype=np.float32)
        for key in h5f:
            if key.startswith("time_division"):
                continue
            A_total += h5f[key][:]
    attributions = torch.tensor(A_total, dtype=torch.float32).to(device)
    return attributions


def generate_mask_from_attributions(attributions, percent, top=True):
    """
    Generate a binary mask from attributions selecting top or flop percent
    based on raw values (not absolute).

    Args:
        attributions (Tensor): Attribution map of shape [F, T]
        percent (float): Percentage of elements to select (0–100)
        top (bool): If True, select top values; else select lowest values

    Returns:
        mask (Tensor): Binary mask of shape [F, T] with 1s in selected bins
    """
    flat = attributions.flatten()
    k = int(flat.numel() * (percent / 100.0))
    if k == 0:
        return torch.zeros_like(attributions)

    if top:
        threshold = torch.topk(flat, k, largest=True).values.min()
        mask = (attributions >= threshold).float()
    else:
        threshold = torch.topk(flat, k, largest=False).values.max()
        mask = (attributions <= threshold).float()

    return mask



def generate_top_percent_mask(wav_path, top_percent=10.0):
    attributions = load_attributions_from_h5(wav_path)
    return generate_mask_from_attributions(attributions, top_percent, top=True)


def generate_flop_percent_mask(wav_path, flop_percent=10.0):
    attributions = load_attributions_from_h5(wav_path)
    return generate_mask_from_attributions(attributions, flop_percent, top=False)


def load_output_bin_attributions(wav_path, F_bin=0, T_frame=0):
    """
    Load the aggregated attribution map from the h5 file corresponding to the input wav.

    Returns:
        attributions (Tensor): Attribution map of shape [F, T] on the correct device.
    """
    model, device = load_nsnet2_model()
    wav, _ = load_and_resample(wav_path, target_sr=16000)
    wav = wav.to(device)
    spec = model.preproc(wav)
    F_bins, T_frames = spec.shape[-2:]

    input_basename = os.path.basename(wav_path).replace(".wav", "")
    h5_filename = f"DeepShap/attributions/tf_attributions_h5py/{input_basename}_attributions.h5"
    with h5py.File(h5_filename, "r") as h5f:
        A_total = np.zeros((F_bins, T_frames), dtype=np.float32)
        for key in h5f:
            f0, t0 = map(int, [key.split("_")[0][1:], key.split("_")[1][1:]])
            if key.startswith("time_division"):
                continue
            if f0 != F_bin or t0 != T_frame:
                continue
            A_total += h5f[key][:]
    attributions = torch.tensor(A_total, dtype=torch.float32).to(device)
    return attributions