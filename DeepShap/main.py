"""
main.py loads a noisy input, computes the DeepLIFTShap attributions for each frequency band. It saves the attribution map to a JSON file and plots the attributions.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import torch
from utils.data_utils import load_and_resample, save_attr_map, is_already_processed
from config.parameters import sample_rate, freq_bands, hop_length
from utils.plot_utils import plot_spectrogram_and_attributions
from utils.model_utils import load_nsnet2_model
from utils.attributions_utils import (
    compute_frequency_time_bands_attributions,
    convert_attr_map_to_freq_time,
)


def process_file(input_path, division):
    model, device = load_nsnet2_model()
    noisy_input, _ = load_and_resample(input_path, sample_rate)
    nb_frames = noisy_input.shape[-1]
    time_bands_sec = [
        (i / division, (i + 1) / division)
        for i in range(int(division * nb_frames / sample_rate))
    ]
    time_bands = [
        (int(start * sample_rate / hop_length), int(end * sample_rate / hop_length))
        for start, end in time_bands_sec
    ]
    baseline = torch.zeros((10, 1, noisy_input.shape[-1]), device=device)
    attr_map = compute_frequency_time_bands_attributions(
        noisy_input,
        model,
        device,
        baseline,
        batch_size=16,
        time_bands=time_bands,
        division=division,
    )
    save_attr_map(attr_map, input_path, freq_bands, division)
    converted_attr_map = convert_attr_map_to_freq_time(
        input_path, attr_map, time_bands_sec
    )
    plot_spectrogram_and_attributions(input_path, converted_attr_map, division)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Folder with wav files"
    )
    parser.add_argument(
        "--divisions", type=int, nargs="+", default=[16], help="List of division values"
    )
    args = parser.parse_args()

    wav_files = [
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.endswith(".wav")
    ]
    for input_path in wav_files:
        for division in args.divisions:
            if is_already_processed(input_path, division):
                print(f"Skipping {input_path} division {division}, already processed.")
                continue
            print(f"Processing {input_path} with division {division}")
            process_file(input_path, division)
            print(f"Done {input_path} division {division}")


if __name__ == "__main__":
    main()
    print("DeepLIFTShap attributions computed and saved successfully.")
