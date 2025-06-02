import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from tqdm import tqdm

from captum.attr import DeepLiftShap
from utils.data_utils import load_and_resample, batchify_targets
from models.frequency_time_feature_model import BandFeatureFrequencyTimeModel
from config.parameters import sample_rate, freqs
from scipy.interpolate import interp1d


def compute_frequency_time_bands_attributions(
    noisy_input,
    model,
    device,
    baseline,
    batch_size=16,
    division=1,
    freq_bands=None,
    time_bands=None,
):
    wrapped_model = BandFeatureFrequencyTimeModel(model)
    deep_shap = DeepLiftShap(wrapped_model)
    nb_freq_bands = len(freq_bands)
    nb_time_bands = len(time_bands)
    attr_map = np.zeros((nb_freq_bands, nb_time_bands))
    all_targets = [i for i in range(nb_freq_bands * nb_time_bands)]
    noisy_deepshap_input = noisy_input.to(device).unsqueeze(0).requires_grad_(True)
    progress_bar = tqdm(total=len(all_targets), desc="Computing attributions")
    for batch in batchify_targets(all_targets, batch_size=batch_size):
        input_batch = noisy_deepshap_input.to(device).repeat(len(batch), 1, 1)
        model.train()
        attrs = deep_shap.attribute(
            input_batch,
            baselines=baseline,
            target=batch,
            additional_forward_args=(time_bands,),
        )
        attrs = attrs.detach().cpu().numpy()
        model.eval()
        for idx, b in enumerate(batch):
            # Store the attribution in the map
            freq_idx = b // nb_time_bands
            time_idx = b % nb_time_bands
            attr_map[freq_idx, time_idx] = attrs[idx].mean()
        progress_bar.update(len(batch))
    progress_bar.close()
    return attr_map


def convert_attr_map_to_mel_scale(
    input_path, attr_map, time_bands_sec, freq_bands, n_mels=62
):
    print(f"Converting attribution map to Mel scale for {input_path}...")

    file, _ = load_and_resample(input_path, sample_rate)
    frames = np.arange(file.shape[-1])
    seconds = frames / sample_rate

    # Create a linear map from frequency bands to time bands
    freq_band_idx_for_f = np.zeros(len(freqs), dtype=int)
    for idx, (f_low, f_high) in enumerate(freq_bands):
        mask = (freqs >= f_low) & (freqs < f_high)
        freq_band_idx_for_f[mask] = idx
    time_band_idx_for_sec = np.zeros(len(seconds), dtype=int)
    for idx, (start, end) in enumerate(time_bands_sec):
        mask = (seconds >= start) & (seconds < end)
        time_band_idx_for_sec[mask] = idx
    linear_map = attr_map[freq_band_idx_for_f[:, None], time_band_idx_for_sec[None, :]]

    # Mel frequency interpolation
    mel_min = 0
    mel_max = 2595 * np.log10(1 + (sample_rate // 2) / 700)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    mel_freqs = hz_points[1:-1]

    interp_func = interp1d(
        freqs, linear_map, kind="linear", axis=0, fill_value="extrapolate"
    )
    mel_attr_map = interp_func(mel_freqs)

    return mel_attr_map, mel_freqs
