"""
# This script processes WAV files or directories containing WAV files to compute and visualize already computed DeepLIFTShap attributions.
"""

import argparse
import os
import sys
from utils.data_utils import (
    detect_and_remove_incomplete_keys,
    load_and_resample,
    prepare_logpower_deepshap_input_and_baseline,
    get_wav_files,
)
from utils.plot_utils import (
    plot_global_influence,
    plot_input_freq_influence,
    plot_input_low_freq_influence,
    plot_input_time_influence,
    plot_input_time_correlation,
    make_time_normalized_video_from_attributions,
    make_video_from_attributions,
    make_frame_grouped_video_from_attributions,
)
from utils.model_utils import load_nsnet2_model

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def plot_attributions(wav_file):

    model, device = load_nsnet2_model()
    # Wrap the log‐power → mask logic in Captum:

    wav, _ = load_and_resample(wav_file, target_sr=16000)
    wav = wav.to(device)
    # x_test = torch.randn((1, 16000), device="cuda")      # shape [1, waveform_len]

    spec = model.preproc(wav)
    F_bins, T_frames = spec.shape[-2:]
    input_basename = os.path.basename(wav_file).replace(".wav", "")
    h5_filename = (
        f"DeepShap/attributions/tf_attributions_h5py/{input_basename}_attributions.h5"
    )

    if not os.path.exists(h5_filename):
        print(f"File {h5_filename} does not exist. Skipping...")
        return

    detect_and_remove_incomplete_keys(h5_filename)

    # A) Collapse along (f0, f_in) to see “input‐bins’ global influence”
    plot_global_influences_separately(h5_filename, input_basename, F_bins, T_frames)

    # B) Influence of input-time on each output-time
    plot_input_freq_influence(h5_filename, input_basename, F_bins)
    plot_input_low_freq_influence(
        h5_filename, input_basename, F_bins, high_freq_cutoff=1000
    )

    # C) Influence of input-frequency on each output-frequency
    plot_input_time_influence(h5_filename, input_basename, T_frames)

    # D) Correlation between input-time and output-time
    plot_input_time_correlation(h5_filename, input_basename, T_frames)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Folder with wav files or a single wav file",
    )
    args = parser.parse_args()

    wav_files = get_wav_files(args)

    for wav_file in wav_files:
        print(f"Processing {wav_file}...")
        plot_attributions(wav_file)
        print(f"Finished processing {wav_file}.")


if __name__ == "__main__":
    main()
