import os
import torch
import numpy as np
from captum.attr import DeepLiftShap
from asteroid.models import DCCRNet
from tqdm import tqdm

from utils.data_utils import load_and_resample, batchify_targets, save_attributions_to_json, is_already_processed, save_path
from models.band_feature_model import BandFeatureModel
from config.bands import bands, sample_rate

"""
process_dataset.py loads noisy inputs"""
# Directory containing the noisy input files
data_dir = "/home/azureuser/cloudfiles/code/Users/iguittard/XAI-Internship/data/clean_trainset_28spk_wav"

# Load the model and the DeepLIFTShap wrapper
model = DCCRNet.from_pretrained("JorisCos/DCCRNet_Libri1Mix_enhsingle_16k")
model.eval()
wrapped_model = BandFeatureModel(model)
deep_shap = DeepLiftShap(wrapped_model)

# Load the noisy input
all_files = [f for f in os.listdir(data_dir) if f.endswith(".wav")]
print(f"Found {len(all_files)} audio files.")

for fname in tqdm(all_files, desc="Processing files"):
    input_path = os.path.join(data_dir, fname)
    # Check if the file has already been processed
    if is_already_processed(save_path, fname):
        print(f"{fname} already processed, skipping.")
        continue
    try:
        noisy_input, sr = load_and_resample(input_path, sample_rate)
        noisy_input = noisy_input.requires_grad_(True)

        # Baseline
        N = 10
        T = noisy_input.shape[-1]
        baseline = torch.zeros((N, 1, T), device=noisy_input.device)

        # Compute the attribution map
        nb_bands = len(bands)
        attr_map = np.zeros(nb_bands)
        all_targets = list(range(nb_bands))

        for batch in batchify_targets(all_targets, batch_size=2):
            print(f"Computing attribution for file {fname}, batch {batch}")
            input_batch = noisy_input.repeat(len(batch), 1, 1)
            attrs = deep_shap.attribute(input_batch, baselines=baseline, target=batch)
            attrs = attrs.squeeze().cpu().detach().numpy()
            for idx, b in enumerate(batch):
                attr_map[b] = attrs[idx].mean()

        # Save the attribution map to a JSON file
        save_attributions_to_json(attr_map, bands, input_path)

    except Exception as e:
        print(f"[ERROR] Failed processing {fname}: {e}")
