"""
main.py loads a noisy input, computes the DeepLIFTShap attributions for each frequency band. It saves the attribution map to a JSON file and plots the attributions.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import torch
from itertools import product
from utils.data_utils import (
    save_attr_map,
    is_already_processed,
    compute_time_bands,
    prepare_deepshap_input_and_baseline,
)
from config.parameters import sample_rate, freq_bands, hop_length
from utils.plot_utils import (
    plot_spectrogram_and_attributions,
)
from utils.model_utils import load_nsnet2_model
from utils.attributions_utils import (
    compute_frequency_time_bands_attributions,
    convert_attr_map_to_mel_scale,
)


def process_file(
    input_path, division, noise_type, baseline_type, freq_range=None, rms_amplitude=None
):
    torch.cuda.empty_cache()
    file_basename = os.path.split(input_path)[-1]
    if is_already_processed(
        file_basename, division, noise_type, baseline_type, freq_range, rms_amplitude
    ):
        return

    print(
        f"\nPROCESSING {input_path} \nDivision: {division} \nNoise type: {noise_type} \nFrequency range: {freq_range} \nRMS amplitude: {rms_amplitude}"
    )
    model, device = load_nsnet2_model()

    deepshap_input, baseline, deepshap_input_path = prepare_deepshap_input_and_baseline(
        input_path,
        file_basename,
        noise_type,
        baseline_type,
        device,
        sample_rate,
        freq_range=freq_range,
        rms_amplitude=rms_amplitude,
    )

    time_bands_sec, time_bands = compute_time_bands(
        deepshap_input, sample_rate, hop_length, division
    )

    attr_map = compute_frequency_time_bands_attributions(
        deepshap_input,
        model,
        device,
        baseline,
        batch_size=8,
        division=division,
        freq_bands=freq_bands,
        time_bands=time_bands,
    )
    save_attr_map(
        attr_map,
        file_basename,
        freq_bands,
        division,
        noise_type,
        baseline_type,
        freq_range if noise_type == "sinusoidal" else None,
        rms_amplitude=rms_amplitude if noise_type == "sinusoidal" else None,
    )

    converted_attr_map, _ = convert_attr_map_to_mel_scale(
        input_path, attr_map, time_bands_sec, freq_bands, n_mels=62
    )

    plot_spectrogram_and_attributions(
        deepshap_input_path,
        file_basename,
        converted_attr_map,
        division,
        noise_type,
        baseline_type,
        freq_range,
        rms_amplitude,
    )
    print(f"Attributions saved for {input_path} with division {division}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Folder with wav files"
    )
    parser.add_argument(
        "--divisions", type=int, nargs="+", default=[16], help="List of division values"
    )
    parser.add_argument(
        "--baseline_type",
        type=str,
        default="zero",
        choices=["zero", "clean_audio"],
        help="Baseline type for attributions: 'zero' for zero baseline, 'clean' for clean audio baseline",
    )
    parser.add_argument(
        "--noise_type",
        type=str,
        default="not added",
        choices=["not_added", "impulsive", "sinusoidal"],
        help="Type of noise to add: 'not_added', 'impulsive' for impulsive noise, 'sinusoidal' for sinusoidal noise",
    )
    parser.add_argument(
        "--freq_ranges",
        type=str,
        nargs="+",
        default=None,
        help="List of frequency ranges for sinusoidal noise (e.g., '1000-2000')",
    )
    parser.add_argument(
        "--rms_amplitudes",
        type=float,
        nargs="+",
        default=None,
        help="List of RMS amplitudes for sinusoidal noise (e.g., 0.01)",
    )
    args = parser.parse_args()

    wav_files = [
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.endswith(".wav")
    ]
    for input_path in wav_files:
        for division in args.divisions:
            if args.noise_type == "sinusoidal":
                for freq_range, rms_amplitude in product(
                    args.freq_ranges, args.rms_amplitudes
                ):
                    freq_range = tuple(map(int, freq_range.split("-")))
                    process_file(
                        input_path,
                        division,
                        args.noise_type,
                        args.baseline_type,
                        freq_range,
                        rms_amplitude,
                    )
            else:
                process_file(
                    input_path,
                    division,
                    args.noise_type,
                    args.baseline_type,
                    None,
                    None,
                )


if __name__ == "__main__":
    main()
    print("DeepLIFTShap attributions computed and saved successfully.")
