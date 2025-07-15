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
    Generate a binary mask from attributions selecting top or flop percent.

    Args:
        attributions (Tensor): Attribution map of shape [F, T]
        percent (float): Percentage to keep
        top (bool): If True, select top percent; else select flop percent.

    Returns:
        mask (Tensor): Binary mask of shape [F, T]
    """
    flat = attributions.abs().flatten()
    k = int(flat.numel() * (percent / 100.0))
    if top:
        threshold = torch.topk(flat, k).values.min()
        mask = (attributions.abs() >= threshold).float()
    else:
        threshold = torch.topk(flat, k, largest=False).values.max()
        mask = (attributions.abs() <= threshold).float()
    return mask


def generate_top_percent_mask(wav_path, top_percent=10.0):
    attributions = load_attributions_from_h5(wav_path)
    return generate_mask_from_attributions(attributions, top_percent, top=True)


def generate_flop_percent_mask(wav_path, flop_percent=10.0):
    attributions = load_attributions_from_h5(wav_path)
    return generate_mask_from_attributions(attributions, flop_percent, top=False)
