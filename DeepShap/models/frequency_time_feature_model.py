import torch
import torch.nn as nn
from torchaudio.functional import amplitude_to_DB
from config.parameters import freq_bands, n_fft, hop_length, freqs


def extract_band_time_features(x, time_bands) -> torch.Tensor:
    """
    Extracts features by frequency and time bands via STFT.
    Args:
        x (torch.Tensor): Input signal of shape [B, 1, T].
    Returns:
        torch.Tensor: Tensor of shape [B, nb_freq_bands, nb_time_bands].
    """
    B, _, T = x.shape
    stft = torch.stft(
        x.squeeze(1),
        n_fft=n_fft,
        hop_length=hop_length,
        window=torch.hann_window(n_fft, device=x.device),
        return_complex=True,
    )  # → [B, F, T']
    mag = stft.abs()  # Magnitude → [B, F, T']
    band_time_feats = []
    for f_low, f_high in freq_bands:
        freq_mask = (freqs >= f_low) & (freqs < f_high)
        masked_mag = mag[:, freq_mask, :]  # [B, f_band, T']

        time_feats = []
        for t_start, t_end in time_bands:
            time_mask = torch.zeros(
                masked_mag.shape[-1], dtype=torch.bool, device=x.device
            )
            time_mask[t_start:t_end] = True
            selected_frames = masked_mag[:, :, time_mask]  # [B, f_band, t_band]
            time_feats.append(
                amplitude_to_DB(
                    selected_frames, multiplier=10.0, amin=1e-10, db_multiplier=0.0
                ).mean(dim=(1, 2))
            )  # [B] (mean over f_band and t_band)
        band_time_feats.append(torch.stack(time_feats, dim=1))  # [B, nb_time_bands]
    # Stack all frequency band features
    stacked_feats = []
    stacked_feats = torch.stack(
        band_time_feats, dim=1
    )  # [B, nb_freq_bands, nb_time_bands]
    if torch.isnan(stacked_feats).any():
        raise ValueError("NaN values detected in the extracted features.")
    return stacked_feats  # [B, nb_freq_bands, nb_time_bands]


class BandFeatureFrequencyTimeModel(nn.Module):
    """
    Wrapper to extract frequency band × frequency time features from the model.
    This is used to make the model compatible with Captum.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, time_bands):
        enhanced = self.model(x)  # [B, 1, T]
        features = extract_band_time_features(
            enhanced, time_bands
        )  # [B, nb_freq_bands, nb_time_bands]
        B, nb_freq_bands, nb_time_bands = features.shape
        return features.view(
            B, nb_freq_bands * nb_time_bands
        )  # Flatten to [B, nb_freq_bands * nb_time_bands]
