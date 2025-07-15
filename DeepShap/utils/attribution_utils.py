import torch
import os
import h5py
import numpy as np
from DeepShap.utils.common_utils import load_and_resample
from utils.model_utils import load_nsnet2_model


def generate_top_percent_mask(wav_path, top_percent=10.0):
    """
    Generate a binary mask where the top `top_percent`% of attributions (by absolute value) are set to 1.

    Args:
        attributions (Tensor): Attribution map of shape [F, T]
        top_percent (float): Percentage of most important attributions to keep (default: 10%)

    Returns:
        mask (Tensor): Binary mask of shape [F, T]
    """
    model, device = load_nsnet2_model()
    wav, _ = load_and_resample(wav_path, target_sr=16000)
    wav = wav.to(device)

    spec = model.preproc(wav)
    F_bins, T_frames = spec.shape[-2:]
    input_basename = os.path.basename(wav_path).replace(".wav", "")
    h5_filename = (
        f"DeepShap/attributions/tf_attributions_h5py/{input_basename}_attributions.h5"
    )
    h5f = h5py.File(h5_filename, "r")
    A_total = np.zeros((F_bins, T_frames), dtype=np.float32)
    for key in h5f:
        if key.startswith("time_division"):
            continue
        A_total += h5f[key][:]
    h5f.close()
    attributions = torch.tensor(A_total, dtype=torch.float32).to(device)

    flat = attributions.abs().flatten()
    k = int(flat.numel() * (top_percent / 100.0))

    threshold = torch.topk(flat, k).values.min()
    mask = (attributions.abs() >= threshold).float()
    return mask


def generate_flop_percent_mask(wav_path, flop_percent=10.0):
    """
    Generate a binary mask where the flop `flop_percent`% of attributions (by absolute value) are set to 1.

    Args:
        attributions (Tensor): Attribution map of shape [F, T]
        flop_percent (float): Percentage of least important attributions to keep (default: 10%)

    Returns:
        mask (Tensor): Binary mask of shape [F, T]
    """
    model, device = load_nsnet2_model()
    wav, _ = load_and_resample(wav_path, target_sr=16000)
    wav = wav.to(device)

    spec = model.preproc(wav)
    F_bins, T_frames = spec.shape[-2:]
    input_basename = os.path.basename(wav_path).replace(".wav", "")
    h5_filename = (
        f"DeepShap/attributions/tf_attributions_h5py/{input_basename}_attributions.h5"
    )
    h5f = h5py.File(h5_filename, "r")
    A_total = np.zeros((F_bins, T_frames), dtype=np.float32)
    for key in h5f:
        if key.startswith("time_division"):
            continue
        A_total += h5f[key][:]
    h5f.close()
    attributions = torch.tensor(A_total, dtype=torch.float32).to(device)

    flat = attributions.abs().flatten()
    k = int(flat.numel() * (flop_percent / 100.0))

    threshold = torch.topk(flat, k, largest=False).values.max()
    mask = (attributions.abs() <= threshold).float()
    return mask
