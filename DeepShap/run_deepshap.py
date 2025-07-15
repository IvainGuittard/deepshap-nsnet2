import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from captum.attr import DeepLiftShap
import matplotlib.pyplot as plt
from DeepShap.utils.common_utils import load_and_resample
from DeepShap.utils.model_utils import load_nsnet2_model
from DeepShap.utils.data_utils import (
    prepare_logpower_deepshap_input_and_baseline,
    create_h5_file_and_keys,
    detect_and_remove_incomplete_keys,
)
from tqdm import tqdm
from DeepShap.models.MaskEstimationBlock import MaskEstimationBlock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load the NSNet2 model
model, device = load_nsnet2_model()

# Wrap the log‐power → mask logic in Captum:
wrapper = MaskEstimationBlock(model).cuda().train()
dl_shap = DeepLiftShap(wrapper)


# Prepare one test log‐power spectrogram + a set of baselines
def run_deep_shap_on_file(input_path):
    """Run DeepLIFTShap on the specified input file or directory.
    Args:
        input_path (str): Path to the input WAV file or directory containing WAV files.
    """
    if os.path.isdir(input_path):
        wav_files = [
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.endswith(".wav")
        ]
    elif os.path.isfile(input_path) and input_path.endswith(".wav"):
        wav_files = [input_path]
    else:
        raise ValueError(
            f"Invalid input: {input_path}. Must be a WAV file or a directory containing WAV files."
        )

    if not wav_files:
        raise ValueError("No WAV files found in the specified input.")

    # ─── Process Each WAV File ────────────────
    for x_input_path in wav_files:
        print(f"\n Processing {x_input_path}...")
        x_input, _ = load_and_resample(x_input_path, target_sr=16000)
        x_input = x_input.to(device)
        # x_test = torch.randn((1, 16000), device="cuda")      # shape [1, waveform_len]

        input_logpower, baseline_logpower = (
            prepare_logpower_deepshap_input_and_baseline(model, x_input)
        )
        F_bins, T_frames = input_logpower.shape[-2:]

        # ─── C) Loop over every TF‐bin and save its attribution map ─────────────
        input_basename = os.path.basename(x_input_path).replace(".wav", "")
        h5_filename = f"DeepShap/attributions/tf_attributions_h5py/{input_basename}_attributions.h5"
        os.makedirs(os.path.dirname(h5_filename), exist_ok=True)
        if os.path.exists(h5_filename):
            detect_and_remove_incomplete_keys(h5_filename)
        h5f, existing_keys = create_h5_file_and_keys(h5_filename)
        time_division = 1
        if "time_division" not in h5f:
            h5f.create_dataset("time_division", data=time_division)
        progress_bar = tqdm(
            total=F_bins * T_frames // time_division, desc="Computing attributions"
        )

        for f0 in range(F_bins):
            for t0 in range(0, T_frames, time_division):
                # Specify the single‐pixel target: (channel_index, freq_index, time_index).
                # Here channel_index is always 0 (because wrapper output is [B,1,F,T]).
                target = (0, f0, t0)
                key = f"f{f0}_t{t0}"
                if key in existing_keys:
                    print(
                        f"Attribution for (f={f0}, t={t0}) already exists. Skipping computation."
                    )
                    progress_bar.update(1)
                    continue

                # Compute attribution map for mask[b,0,f0,t0]
                attributions = dl_shap.attribute(
                    inputs=input_logpower,  # [1, 257, T_frames]
                    baselines=baseline_logpower,  # [20, 257, T_frames]
                    target=target,  # which mask‐pixel to explain
                )
                # attributions: [1, 257, T_frames]
                attr_map = attributions[0].detach().cpu().numpy()
                h5f.create_dataset(
                    key, data=attr_map.astype("float16"), compression="gzip"
                )

                os.makedirs(
                    f"DeepShap/attributions/tf_attribution_maps/{input_basename}_tf_attributions",
                    exist_ok=True,
                )
                filename = f"DeepShap/attributions/tf_attribution_maps/{input_basename}_tf_attributions/attr_f{f0}_t{t0}.png"
                if (t0 % 10 == 0 or t0 == T_frames - 1) and f0 % 10 == 0:
                    plt.savefig(filename, bbox_inches="tight")
                plt.close()

                progress_bar.update(1)

        progress_bar.close()
        h5f.close()
        print(f"Done processing {x_input_path}!")
