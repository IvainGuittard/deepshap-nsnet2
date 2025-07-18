import os
import sys

from DeepShap.utils.audio_pertubations import (
    add_reverb,
    add_sinusoidal_noise,
    add_white_noise,
)
from DeepShap.utils.common_utils import load_and_resample

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torchaudio
import torch
import h5py


def get_wav_files(args):
    if os.path.isfile(args.input_dir):
        wav_files = [args.input_dir]
    elif os.path.isdir(args.input_dir):
        wav_files = [
            os.path.join(args.input_dir, f)
            for f in os.listdir(args.input_dir)
            if f.endswith(".wav")
        ]
    else:
        raise ValueError(
            f"Invalid input: {args.input_dir}. Must be a WAV file or a directory."
        )
    return wav_files


def prepare_deepshap_input(
    input_path,
    file_basename,
    noise_type,
    sample_rate,
    freq_range=None,
    rms_amplitude=None,
    reverberance=50.0,
    time_range=None,
):
    if noise_type == "not_added":
        deepshap_input, _ = load_and_resample(input_path, sample_rate)
        deepshap_input_path = input_path

    elif noise_type == "sinusoidal":
        if freq_range is None or rms_amplitude is None:
            raise ValueError(
                "For 'sinusoidal' noise type, freq_range and rms_amplitude must be provided."
            )

        clean_input, _ = load_and_resample(input_path, sample_rate)
        deepshap_input = add_sinusoidal_noise(
            clean_input,
            sample_rate,
            freq_range=freq_range,
            total_rms_amplitude=rms_amplitude,
        )
        # Save the input with added noise
        deepshap_input_path = f"data/added_sinusoidal_noise_input/{file_basename.replace('.wav', '')}_freq_{freq_range[0]}-{freq_range[1]}_rms_{rms_amplitude}.wav"
        torchaudio.save(deepshap_input_path, deepshap_input, sample_rate=sample_rate)

    elif noise_type == "reverberation":
        clean_input, _ = load_and_resample(input_path, sample_rate)
        deepshap_input = add_reverb(clean_input, sample_rate, reverberance=reverberance)

        # Save the input with added reverberation
        deepshap_input_path = f"data/added_reverberation_input/{file_basename.replace('.wav', '')}_reverb_{reverberance}.wav"
        torchaudio.save(deepshap_input_path, deepshap_input, sample_rate=sample_rate)

    elif noise_type == "white":
        clean_input, _ = load_and_resample(input_path, sample_rate)
        deepshap_input = add_white_noise(
            clean_input,
            sample_rate,
            time_range=time_range,
            amplitude=rms_amplitude,
        )

        # Save the input with added white noise
        deepshap_input_path = f"data/added_white_noise_input/{file_basename.replace('.wav', '')}_white_noise_{time_range[0]}-{time_range[1]}_amplitude_{rms_amplitude}.wav"
        torchaudio.save(deepshap_input_path, deepshap_input, sample_rate=sample_rate)

    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    return deepshap_input, deepshap_input_path


def prepare_logpower_deepshap_input_and_baseline(model, input):
    """
    Prepares the log-power spectrogram and silent baseline for DeepLiftShap input.

    Args:
        model: The NSNet2 model instance.
        input: The input waveform tensor.

    Returns:
        input_logpower: Log-power spectrogram tensor.
        baseline_logpower: Silent baseline tensor.
    """
    input_spec_complex = model.preproc(input)  # [1, 1, 257, T_frames], complex
    input_logpower = torch.log(input_spec_complex.abs() ** 2 + model.eps).squeeze(1) # [1, 257, T_frames]

    # Build a “silent‐baseline” :
    B_baseline = 20
    F_bins, T_frames = input_logpower.shape[1], input_logpower.shape[2]
    baseline_logpower = torch.log(
        torch.full((B_baseline, F_bins, T_frames), fill_value=model.eps, device="cuda")
    )  # → [20, 257, T_frames]

    return input_logpower, baseline_logpower


def create_h5_file_and_keys(h5_filename):
    h5f = h5py.File(h5_filename, "a")
    existing_keys = set(h5f.keys())
    return h5f, existing_keys


def detect_and_remove_incomplete_keys(h5_filename):
    """
    Detect and remove incomplete or corrupted keys in an HDF5 file.

    Args:
        h5_filename (str): Path to the HDF5 file.
    """
    try:
        print(f"Checking for corrupted keys in {h5_filename}...")
        with h5py.File(h5_filename, "a") as h5f:
            keys_to_remove = []
            for key in h5f.keys():
                try:
                    _ = h5f[key][:]
                except Exception as e:
                    print(f"Key '{key}' is corrupted: {e}")
                    keys_to_remove.append(key)

            # Remove corrupted keys
            for key in keys_to_remove:
                print(f"Removing corrupted key: {key}")
                del h5f[key]
        print(f"Completed checking {h5_filename}. Corrupted keys removed if any.")
        h5f.close()

    except Exception as e:
        print(f"Error while processing {h5_filename}: {e}")
