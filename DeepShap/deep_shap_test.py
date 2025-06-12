import os
import sys
from captum.attr import DeepLiftShap
import matplotlib.pyplot as plt
from utils.model_utils import load_nsnet2_model
from utils.data_utils import load_and_resample, prepare_logpower_deepshap_input_and_baseline, create_h5_file_and_keys
from utils.plot_utils import plot_global_influence, plot_input_time_influence, plot_input_freq_influence
from tqdm import tqdm
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

h5_filename = f"tf_attributions/{os.path.basename(x_test_path).split('.')[0]}_attributions.h5"
h5f, existing_keys = create_h5_file_and_keys(h5_filename)

progress_bar = tqdm(total=F_bins * T_frames, desc="Computing attributions")

for f0 in range(F_bins):
    for t0 in range(T_frames):
        # Specify the single‐pixel target: (channel_index, freq_index, time_index).
        # Here channel_index is always 0 (because wrapper output is [B,1,F,T]).
        target = (0, f0, t0)
        key = f"f{f0}_t{t0}"
        if key in existing_keys:
            print(f"Attribution for (f={f0}, t={t0}) already exists. Skipping computation.")
            progress_bar.update(1)
            continue

        # Compute attribution map for mask[b,0,f0,t0]
        attributions = dl_shap.attribute(
            inputs=input_logpower,       # [1, 257, T_frames]
            baselines=baseline_logpower, # [50, 257, T_frames]
            target=target                # which mask‐pixel to explain
        )
        # attributions: [1, 257, T_frames]
        attr_map = attributions[0].detach().cpu().numpy()
        h5f.create_dataset(key, data=attr_map, compression="gzip")

        plt.figure(figsize=(4, 3))
        plt.imshow(attr_map, origin="lower", aspect="auto", cmap="seismic")
        plt.title(f"Attribution for mask[0,0,{f0},{t0}]")
        plt.xlabel("Time frame t")
        plt.ylabel("Freq bin f")
        plt.colorbar(fraction=0.046, pad=0.04)

        filename = f"tf_attributions/attr_f{f0}_t{t0}.png"
        plt.savefig(filename, bbox_inches="tight")
        plt.close()

        progress_bar.update(1)

progress_bar.close()
h5f.close()
print("Done! All TF‐bin attribution maps are in the folder: tf_attributions/")


# ─── D) Collapse along (f0, f_in) to see “input‐bins’ global influence” ─────

plot_global_influence(h5_filename, F_bins, T_frames)

# ─── E) Influence of input-time on each output-time ─────

plot_input_time_influence(h5_filename, T_frames)

# ─── F) Influence of input-frequency on output-frequency ─────

plot_input_freq_influence(h5_filename, F_bins)