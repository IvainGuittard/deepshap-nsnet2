import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torchaudio
import json


def batchify_targets(targets, batch_size):
    """Yield successive batches from the targets list."""
    for i in range(0, len(targets), batch_size):
        yield targets[i:i + batch_size]


def load_and_resample(path, target_sr):
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    return waveform, target_sr


def is_already_processed(input_path, division):
    """
    Check if the input file has already been processed by looking for its attribution map.
    Args:
        input_path (str): Path to the input audio file.
    Returns:
        bool: True if the file has already been processed, False otherwise.
    """
    folder_name_1 = input_path.split('/')[-1].replace('.wav', '')
    file_name_1 = input_path.split('/')[-1].replace('.wav', f'_div{division}_attribution_map.json')
    save_path_1 = f'DeepShap/attributions/maps/{folder_name_1}/{file_name_1}'

    folder_name_2 = input_path.split('/')[-1].replace('.wav', '')
    file_name_2 = input_path.split('/')[-1].replace('.wav', f'_div{division}_attribution_plot.png')
    save_path_2 = f'DeepShap/attributions/plots/{folder_name_2}/{file_name_2}'
    return os.path.exists(save_path_1) and os.path.exists(save_path_2)


def save_attr_map(attr_map, input_path, freq_bands, division):
    folder_name = input_path.split('/')[-1].replace('.wav', '')
    file_name = input_path.split('/')[-1].replace('.wav', f'_div{division}_attribution_map.json')
    save_path = f'DeepShap/attributions/maps/{folder_name}/{file_name}'
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    new_entry = {
        "frequency_bands": freq_bands,
        "division": division,
        "attributions": {
            f"{freq_bands[i][0]}-{freq_bands[i][1]}Hz_time_band_{j}": float(value)
            for i, band in enumerate(attr_map)
            for j, value in enumerate(band)
        }
    }

    with open(save_path, "w") as f:
        json.dump(new_entry, f, indent=4)

    print(f"Attribution map saved to {save_path}")
    return
