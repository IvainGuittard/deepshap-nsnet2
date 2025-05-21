"""
main.py loads a noisy input, computes the DeepLIFTShap attributions for each frequency band. It saves the attribution map to a JSON file and plots the attributions.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchaudio

from captum.attr import DeepLiftShap
from asteroid.models import DCCRNet
import torch.nn as nn

from utils.data_utils import load_and_resample, batchify_targets, save_attributions_to_json
from models.band_feature_model import BandFeatureModel
from DeepShap.config.parameters import bands, n_fft, freqs, sample_rate

# Load the model and the DeepLIFTShap wrapper
model = DCCRNet.from_pretrained("JorisCos/DCCRNet_Libri1Mix_enhsingle_16k")
model.eval()
wrapped_model = BandFeatureModel(model)
deep_shap = DeepLiftShap(wrapped_model)

# Load the noisy input
input_path = "/home/azureuser/cloudfiles/code/Users/iguittard/XAI-Internship/noisy_input.wav"
noisy_input, sr = load_and_resample(input_path, sample_rate)
noisy_deepshap_input = noisy_input.requires_grad_(True)

# Baseline for DeepLIFTShap
N = 10
T = noisy_deepshap_input.shape[-1]
baseline = torch.zeros((N, 1, T), device=noisy_deepshap_input.device)

###Compute attributions for a frequency bin
nb_bands = len(bands)
attr_map = np.zeros(nb_bands)
all_targets = list(range(nb_bands))

for batch in batchify_targets(all_targets, batch_size = 4):
    print(f"Computing attribution for batch {batch}...")
    input_batch = noisy_deepshap_input.repeat(len(batch), 1, 1)
    attrs = deep_shap.attribute(input_batch, baselines=baseline, target=batch)
    attrs = attrs.squeeze().cpu().detach().numpy()
    for idx, b in enumerate(batch):
        attr_map[b] = attrs[idx].mean()
        print(f"Attribution computed for band {b}: {attr_map[b]:.6f}")

"""#Compute attributions without batchifying
for b in all_targets:
    print(f"Computing attribution for band {b}...")
    attrs = deep_shap.attribute(noisy_deepshap_input, baselines=baseline, target=b)
    attr_map[b] = attrs.squeeze().cpu().detach().numpy().mean()
    print(f"Attribution computed for band {b}: {attr_map[b]:.6f}")"""


# Save the attribution map to a JSON file
save_attributions_to_json(attr_map, bands, input_path)

# Plot the attribution map
plt.bar(range(nb_bands), attr_map)
plt.xticks(range(nb_bands), ["0-1k", "1k-2k", "2k-3k", "3k-4k", "4k-5k", "5k-6k", "6k-7k", "7k-8k"])
plt.ylabel("Attribution")
plt.title("DeepLiftShap Attributions (Bandes de fréquences, moyenne sur le temps)")
plt.tight_layout()
plt.savefig("/home/azureuser/cloudfiles/code/Users/iguittard/XAI-Internship/DeepShap/deepshap_attributions_bandes.png")
plt.close()