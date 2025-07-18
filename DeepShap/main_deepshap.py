"""
main_deepshap.py is the main script for computing DeepLIFTShap attributions on audio files.
It processes WAV files or directories containing WAV files, applies noise if specified, and computes attributions using the NSNet2 model.

"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import torch
from itertools import product
from config.parameters import sample_rate
from utils.data_utils import (
    prepare_deepshap_input,
    get_wav_files,
)
from DeepShap.run_deepshap import run_deep_shap_on_file


def process_file(
    input_path,
    noise_type,
    freq_range=None,
    rms_amplitude=None,
    reverberance=None,
    time_range=None,
):
    torch.cuda.empty_cache()
    file_basename = os.path.split(input_path)[-1]

    print(
        f"\nPROCESSING {input_path} \nNoise type: {noise_type} \nFrequency range: {freq_range} \nRMS amplitude: {rms_amplitude}"
    )

    _, deepshap_input_path = prepare_deepshap_input(
        input_path,
        file_basename,
        noise_type,
        sample_rate,
        freq_range=freq_range,
        rms_amplitude=rms_amplitude,
        reverberance=reverberance,
        time_range=time_range,
    )

    run_deep_shap_on_file(deepshap_input_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Folder with wav files"
    )
    parser.add_argument(
        "--noise_type",
        type=str,
        default="not added",
        choices=["not_added", "sinusoidal", "reverberation", "white"],
        help="Type of noise to add: 'not_added', 'impulsive' for impulsive noise, 'sinusoidal' for sinusoidal noise",
    )
    parser.add_argument(
        "--time_ranges",
        type=str,
        nargs="+",
        default=None,
        help="List of time ranges for white noise (e.g., '0-1')",
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
    parser.add_argument(
        "--reverberances",
        type=float,
        nargs="+",
        default=0.5,
        help="Reverberation time for reverberances noise (default: 0.5)",
    )
    args = parser.parse_args()

    wav_files = get_wav_files(args)

    for input_path in wav_files:
        if args.noise_type == "sinusoidal":
            for freq_range, rms_amplitude in product(
                args.freq_ranges, args.rms_amplitudes
            ):
                freq_range = tuple(map(int, freq_range.split("-")))
                process_file(
                    input_path=input_path,
                    noise_type=args.noise_type,
                    freq_range=freq_range,
                    rms_amplitude=rms_amplitude,
                    reverberance=None,
                    time_range=None,
                )
        elif args.noise_type == "reverberation":
            for reverberance in args.reverberances:
                process_file(
                    input_path=input_path,
                    noise_type=args.noise_type,
                    freq_range=None,
                    rms_amplitude=None,
                    reverberance=reverberance,
                    time_range=None,
                )
        elif args.noise_type == "white":
            for rms_amplitude, time_range in product(
                args.rms_amplitudes, args.time_ranges
            ):
                time_range = tuple(map(float, time_range.split("-")))
                rms_amplitude = float(rms_amplitude)
                process_file(
                    input_path=input_path,
                    noise_type=args.noise_type,
                    freq_range=None,
                    rms_amplitude=rms_amplitude,
                    reverberance=None,
                    time_range=time_range,
                )
        else:
            process_file(
                input_path=input_path,
                noise_type=args.noise_type,
                freq_range=None,
                rms_amplitude=None,
                reverberance=None,
                time_range=None,
            )
    print(
        "DeepLIFTShap attributions computed and saved successfully : noise type: "
        f"{args.noise_type}"
    )


if __name__ == "__main__":
    main()
