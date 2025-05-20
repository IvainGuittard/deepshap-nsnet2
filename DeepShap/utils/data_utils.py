import torchaudio
import json
import os

save_path="DeepShap/attributions/deepshap_attributions.json"

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

def save_attributions_to_json(attr_map, bands, noisy_path, save_path = save_path):
    """
    Save the attribution map to a JSON file with automatic numbering of input file names.
    Args:
        attr_map (np.ndarray): Attribution map of shape [nb_bands].
        bands (list): List of frequency bands.
        noisy_path (str): Path to the noisy input file.
        save_path (str): Path to save the attribution JSON file.
    """
    filename = os.path.basename(noisy_path)

    # Charger les données existantes
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    # Déterminer le prochain index
    next_index = len(data) + 1
    numbered_filename = f"{next_index:05d}_{filename}"

    new_entry = {
        "input_file": numbered_filename,
        "attributions": {
            f"{low}-{high}Hz": float(attr_map[i])
            for i, (low, high) in enumerate(bands)
        }
    }

    data.append(new_entry)

    with open(save_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"[INFO] Attribution for {numbered_filename} added to {save_path}")

def is_already_processed(json_path, input_filename):
    """Check if a given input file (by original name) has already been processed."""
    if not os.path.exists(json_path):
        return False
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return False  # JSON malformé

    # Cherche si un fichier contenant le nom d'origine est déjà présent
    for entry in data:
        if entry.get("input_file", "").endswith(input_filename):
            return True
    return False