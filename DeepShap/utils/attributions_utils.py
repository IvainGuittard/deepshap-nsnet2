import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

from captum.attr import DeepLiftShap
from utils.data_utils import load_and_resample, batchify_targets
from models.frequency_time_feature_model import BandFeatureFrequencyTimeModel
from config.parameters import freq_bands, sample_rate, freqs


def compute_frequency_time_bands_attributions(
    noisy_input, model, device, baseline, batch_size=16, division=1, time_bands=None
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
        print(f"Computing attribution for batch {batch}...")
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


def process_frequency_time_pair(i, f, seconds, freq_bands, time_bands_sec, attr_map):
    converted_row = np.zeros(len(seconds))
    # Trouver l'indice de la bande de fréquence
    freq_band_idx = next(
        (idx for idx, (f_low, f_high) in enumerate(freq_bands) if f_low <= f < f_high),
        (
            0
            if f < freq_bands[0][0]
            else len(freq_bands) - 1 if f >= freq_bands[-1][1] else None
        ),
    )
    if freq_band_idx is None:
        raise ValueError(
            f"Frequency {f} Hz does not fall within any defined frequency band."
        )

    for j, second in enumerate(seconds):
        # Trouver l'indice de la bande de temps
        time_band_idx = next(
            (
                idx
                for idx, (start, end) in enumerate(time_bands_sec)
                if start <= second < end
            ),
            0 if second < time_bands_sec[0][0] else None,
        )
        if time_band_idx is None:
            continue
        # Stocker la valeur d'attribution
        converted_row[j] = attr_map[freq_band_idx, time_band_idx]
    return i, converted_row


def convert_attr_map_to_freq_time(input_path, attr_map, time_bands_sec):
    """
    Convert the attribution map to a 2D array with frequency bands on the y-axis and time bands on the x-axis.

    Parameters:
    - attr_map: 2D numpy array of shape (num_freq_bands, num_time_bands)
    - freq_bands: List of tuples representing frequency bands
    - time_bands: List of tuples representing time bands

    Returns:
    - 2D array of shape (len(freqs), len(frames))
    """
    print(f"Converting attribution map to frequency-time format for {input_path}...")
    file, _ = load_and_resample(input_path, sample_rate)
    frames = np.arange(file.shape[-1])
    seconds = frames / sample_rate
    converted_map = np.zeros((len(freqs), len(seconds)))
    results = Parallel(n_jobs=-1)(
        delayed(process_frequency_time_pair)(
            i, f, seconds, freq_bands, time_bands_sec, attr_map
        )
        for i, f in tqdm(list(enumerate(freqs)))
    )
    for i, converted_row in results:
        converted_map[i, :] = converted_row
    return converted_map
