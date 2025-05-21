import torch
import torch.nn as nn
from DeepShap.config.parameters import bands, n_fft, freqs

def extract_band_features(x):
    """
    Extracts band features from the input signal using STFT.
    Args:
        x (torch.Tensor): Input signal of shape [B, 1, T].
    Returns:
        torch.Tensor: Band features of shape [B, nb_bands].
    """
    stft = torch.stft(
        x.squeeze(1),
        n_fft=n_fft,
        hop_length=128,
        window=torch.hann_window(n_fft, device=x.device),
        return_complex=True
    )
    mag = stft.abs()
    band_feats = []
    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        band_energy = mag[:, mask, :].mean(dim=(1, 2))  # [B], mean over freq and time
        band_feats.append(band_energy)
    return torch.stack(band_feats, dim=1)  # [B, nb_bands]

class BandFeatureModel(nn.Module):
    """
    A wrapper for the model to extract band features, in order to use it with Captum.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        enhanced = self.model(x)  # [B, 1, T]
        return extract_band_features(enhanced)  # [B, nb_bands, T']