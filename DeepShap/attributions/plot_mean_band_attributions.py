import json
import matplotlib.pyplot as plt
import numpy as np

# Load the JSON file
json_path = "DeepShap/attributions/deepshap_attributions_noisy.json"
normalization = True
if normalization:
    save_path = "DeepShap/attributions/deepshap_mean_attributions_noisy.png"
else:
    save_path = "DeepShap/attributions/deepshap_mean_attributions_noisy_normalized.png"
with open(json_path, "r") as f:
    data = json.load(f)

# Frequency bands
freq_bands = ["0-1000Hz", "1000-2000Hz", "2000-3000Hz", "3000-4000Hz",
              "4000-5000Hz", "5000-6000Hz", "6000-7000Hz", "7000-8000Hz"]

if data:
    freq_bands = list(data[0]["attributions"].keys())
else:
    raise ValueError("The JSON file is empty or improperly formatted.")


nb_bands = len(freq_bands)

# Collect all attributions per band
band_values = {band: [] for band in freq_bands}

for entry in data:
    attribs = entry["attributions"]
    total_attribution = sum(attribs.values())  # Sum of all attributions for normalization
    if normalization:
        # Normalize the attributions
        if total_attribution > 0:
            normalized_attribs = {band: value / total_attribution for band, value in attribs.items()}
        else:
            normalized_attribs = {band: 0.0 for band in freq_bands} 
        for band in freq_bands:
            band_values[band].append(normalized_attribs.get(band, 0.0))
    else:
        for band in freq_bands:
            band_values[band].append(attribs.get(band, 0.0))
        

# Compute mean and standard deviation
means = [np.mean(band_values[band]) for band in freq_bands]
stds = [np.std(band_values[band]) for band in freq_bands]

# Plot with error bars
plt.figure(figsize=(10, 5))
plt.bar(range(nb_bands), means, yerr=stds, capsize=5, color="skyblue", edgecolor="black")
plt.xticks(range(nb_bands), ["0–1k", "1k–2k", "2k–3k", "3k–4k", "4k–5k", "5k–6k", "6k–7k", "7k–8k"])
plt.ylabel("Attribution")
plt.xlabel("Frequency Bands")
plt.title("DeepLiftShap Attributions (Mean ± Std Dev per Frequency Band)")
plt.tight_layout()
plt.savefig(save_path)
plt.close()
print(f'[INFO] Plot saved to {save_path}')
