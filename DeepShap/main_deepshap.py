"""
main_deepshap.py is the main script for computing DeepLIFTShap attributions on audio files.
It processes WAV files or directories containing WAV files, applies noise if specified, and computes attributions using the NSNet2 model.

"""

import os
import sys

from DeepShap.utils.data_utils import get_wav_files

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import torch
from itertools import product
from config.parameters import sample_rate
from utils.data_utils import (
    prepare_deepshap_input,
)
import subprocess


def process_file(
    input_path,
    noise_type,
    freq_range=None,
    rms_amplitude=None,
    reverberance=None,
):
    torch.cuda.empty_cache()
    file_basename = os.path.split(input_path)[-1]

    print(
        f"\nPROCESSING {input_path} \nNoise type: {noise_type} \nFrequency range: {freq_range} \nRMS amplitude: {rms_amplitude}"
    )

    deepshap_input, deepshap_input_path = prepare_deepshap_input(
        input_path,
        file_basename,
        noise_type,
        sample_rate,
        freq_range=freq_range,
        rms_amplitude=rms_amplitude,
        reverberance=reverberance,
    )

    subprocess.run(
        [
            "python",
            "DeepShap/deepshap_tf.py",
            "--input",
            deepshap_input_path,
        ],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Folder with wav files"
    )
    parser.add_argument(
        "--noise_type",
        type=str,
        default="not added",
        choices=["not_added", "impulsive", "sinusoidal", "reverberation"],
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
    parser.add_argument(
        "--reverberances",
        type=float,
        nargs="+",
        default=0.5,
        help="Reverberation time for reverberances noise (default: 0.5)",
    )
    args = parser.parse_args()

    if os.path.isfile(args.input_dir):
        wav_files = [args.input_dir]  # Single file
    elif os.path.isdir(args.input_dir):
        wav_files = [
            os.path.join(args.input_dir, f)
            for f in os.listdir(args.input_dir)
            if f.endswith(".wav")
        ]  # All .wav files in the directory
    else:
        raise ValueError(
            f"Invalid input: {args.input_dir}. Must be a WAV file or a directory."
        )

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
                )
        elif args.noise_type == "reverberation":
            for reverberance in args.reverberances:
                process_file(
                    input_path=input_path,
                    noise_type=args.noise_type,
                    freq_range=None,
                    rms_amplitude=None,
                    reverberance=reverberance,
                )
        else:
            process_file(
                input_path=input_path,
                noise_type=args.noise_type,
                freq_range=None,
                rms_amplitude=None,
                reverberance=None,
            )
    print(
        "DeepLIFTShap attributions computed and saved successfully : noise type: "
        f"{args.noise_type}, baseline type: {args.baseline_type}"
    )


if __name__ == "__main__":
    main()
