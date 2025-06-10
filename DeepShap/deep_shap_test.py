import os
import sys
import torch
import torchaudio
import torch.nn as nn
from captum.attr import DeepLiftShap
import matplotlib.pyplot as plt
from utils.model_utils import load_nsnet2_model
from utils.data_utils import load_and_resample, prepare_logpower_deepshap_input_and_baseline
from tqdm import tqdm
import numpy as np
from models.MaskFromLogPower import MaskFromLogPower
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─── A) Load NSNet2 ────────────────
# and that you have already defined `MaskFromLogPower` from the previous steps.
model, device = load_nsnet2_model()
# Wrap the log‐power → mask logic in Captum:

wrapper = MaskFromLogPower(model).cuda().train()
dl_shap = DeepLiftShap(wrapper)

# ─── B) Prepare one test log‐power spectrogram + a set of baselines ───────
# 1 second of noisy waveform (e.g. 16 kHz) → compute STFT → log‐power:
x_test_path = "/home/azureuser/cloudfiles/code/Users/iguittard/XAI-Internship/p226_016_freq_1-125_rms_0.1.wav"
x_test, _ = load_and_resample(x_test_path, target_sr=16000)
x_test = x_test.to(device)
# x_test = torch.randn((1, 16000), device="cuda")      # shape [1, waveform_len]

input_logpower, baseline_logpower = prepare_logpower_deepshap_input_and_baseline(model, x_test)
F_bins, T_frames = input_logpower.shape[-2:]
# Make a directory to save all attribution maps
os.makedirs("tf_attributions", exist_ok=True)

# ─── C) Loop over every TF‐bin and save its attribution map ─────────────
all_attr = np.zeros((F_bins, T_frames, F_bins, T_frames))



for f0 in tqdm(range(F_bins)):
    for t0 in range(T_frames):
        # Specify the single‐pixel target: (channel_index, freq_index, time_index).
        # Here channel_index is always 0 (because wrapper output is [B,1,F,T]).
        target = (0, f0, t0)

        # Compute attribution map for mask[b,0,f0,t0]
        attributions = dl_shap.attribute(
            inputs=input_logpower,       # [1, 257, T_frames]
            baselines=baseline_logpower, # [50, 257, T_frames]
            target=target                # which mask‐pixel to explain
        )
        # attributions: [1, 257, T_frames]
        attr_map = attributions[0].detach().cpu().numpy()
        all_attr[f0, t0] = attributions[0].detach().cpu().numpy()  # store [F, T] map for this (f0,t0)

        # Plot & save the 2D heatmap
        plt.figure(figsize=(4, 3))
        plt.imshow(attr_map, origin="lower", aspect="auto", cmap="seismic")
        plt.title(f"Attribution for mask[0,0,{f0},{t0}]")
        plt.xlabel("Time frame t")
        plt.ylabel("Freq bin f")
        plt.colorbar(fraction=0.046, pad=0.04)

        filename = f"tf_attributions/attr_f{f0}_t{t0}.png"
        plt.savefig(filename, bbox_inches="tight")
        plt.close()

        # (Optional) Print progress every 1000 maps
        if (f0 * T_frames + t0) % 1000 == 0:
            print(f"Generated attribution map for (f={f0}, t={t0}) → {filename}")


print("Done! All TF‐bin attribution maps are in the folder: tf_attributions/")


# ─── D) Collapse along (f0, f_in) to see “input‐bins’ global influence” ─────
# A_in2mask[f_in, t_in] = Σ_{f0, t0} |all_attr[f0, t0, f_in, t_in]|
A_in2mask = all_attr.abs().sum(dim=(0, 1))  # → [F, T]
# Min–max normalize entire map so it’s in [0,1]
A_in2mask_norm = (A_in2mask - A_in2mask.min()) / (A_in2mask.max() - A_in2mask.min() + 1e-12)

plt.figure(figsize=(5, 4))
plt.title("Global influence of each input TF‐bin on the entire mask (min–max norm)")
plt.imshow(A_in2mask_norm.cpu().numpy(), origin="lower", aspect="auto", cmap="magma")
plt.xlabel("input time t_in")
plt.ylabel("input freq f_in")
plt.colorbar(label="normalized attribution")
plt.tight_layout()
plt.show()

# ─── E) Collapse along (f0, f_in) but keep distinction for output-time (t0) ──
# A_time[t0, t_in] = Σ_{f0, f_in} |all_attr[f0, t0, f_in, t_in]|
A_time = all_attr.abs().sum(dim=0).sum(dim=1)  # → [T, T]
# Normalize each row so sum over t_in = 1
A_time_norm = A_time / (A_time.sum(dim=1, keepdim=True) + 1e-12)

plt.figure(figsize=(5, 4))
plt.title("Normalized: How output‐time t0 depends on input‐time t_in")
plt.imshow(A_time_norm.cpu().numpy(), origin="lower", aspect="auto", cmap="viridis")
plt.xlabel("input time t_in")
plt.ylabel("output time t0")
plt.colorbar(label="row‐normalized attribution")
plt.tight_layout()
plt.show()

# ─── F) (Optional) Collapse along (t0, t_in) for output-frequency patterns: ──
# A_freq[f0, f_in] = Σ_{t0, t_in} |all_attr[f0, t0, f_in, t_in]|
A_freq = all_attr.abs().sum(dim=1).sum(dim=3)  # → [F, F]
# Normalize each row so sum over f_in = 1
A_freq_norm = A_freq / (A_freq.sum(dim=1, keepdim=True) + 1e-12)

plt.figure(figsize=(5, 4))
plt.title("Normalized: How output‐freq f0 depends on input‐freq f_in")
plt.imshow(A_freq_norm.cpu().numpy(), origin="lower", aspect="auto", cmap="plasma")
plt.xlabel("input freq f_in")
plt.ylabel("output freq f0")
plt.colorbar(label="row‐normalized attribution")
plt.tight_layout()
plt.show()