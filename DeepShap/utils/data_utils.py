import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torchaudio
import torch
import h5py


def load_and_resample(path, target_sr):
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    return waveform, target_sr


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


def add_sinusoidal_noise(
    waveform, sample_rate, freq_range=(1000, 2000), total_rms_amplitude=0.01
):
    device = waveform.device
    duration = waveform.shape[1] / sample_rate
    time = torch.linspace(0, duration, waveform.shape[1], device=device).unsqueeze(0)
    freqs = torch.linspace(freq_range[0], freq_range[1], steps=300, device=device)

    noise_components = [
        torch.sin(2 * torch.pi * f * time + 2 * torch.pi * torch.rand(1, device=device))
        for f in freqs
    ]

    noise = sum(noise_components)

    rms = torch.sqrt(torch.mean(noise**2))
    noise = noise * (total_rms_amplitude / rms)

    return waveform + noise


def add_white_noise(waveform, sample_rate, time_range=(0, 1), amplitude=0.01):
    """
    Adds white noise to a waveform within a specified time range.

    Args:
        waveform (Tensor): Input waveform tensor of shape [1, num_samples].
        sample_rate (int): Sampling rate of the audio.
        time_range (tuple): Start and end times in seconds for adding noise.
        amplitude (float): Amplitude of the white noise.

    Returns:
        Tensor: Waveform with added white noise.
    """
    device = waveform.device
    num_samples = waveform.shape[1]

    start_idx = int(time_range[0] * sample_rate)
    end_idx = int(time_range[1] * sample_rate)
    start_idx = max(0, start_idx)
    end_idx = min(num_samples, end_idx)

    # Generate white noise
    noise = torch.rand(1, end_idx - start_idx, device=device) * 2 - 1
    noise *= amplitude

    waveform[:, start_idx:end_idx] += noise

    return waveform


def add_reverb(waveform, sample_rate, reverberance=50.0, damping=50.0, room_scale=50.0):
    """
    Adds reverberation to the waveform using SoX effects.

    Args:
        waveform (Tensor): Input waveform tensor of shape [1, num_samples].
        sample_rate (int): Sampling rate of the audio.
        reverberance (float): Amount of reverb (0-100), higher means more echo.
        damping (float): High-frequency damping (0-100).
        room_scale (float): Size of the virtual room (0-100).

    Returns:
        Tensor: Reverberated waveform.
    """
    effects = [
        ["reverb", str(reverberance), str(damping), str(room_scale)],
    ]
    waveform_reverb, _ = torchaudio.sox_effects.apply_effects_tensor(
        waveform, sample_rate, effects
    )
    return waveform_reverb


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
    input_spec_complex = model.preproc(input)  # [1, 1, 257, ~62], complex
    input_logpower = torch.log(input_spec_complex.abs() ** 2 + model.eps).squeeze(1)
    # → input_logpower: [1, 257, T_frames]

    # Build a “silent‐baseline” :
    B_baseline = 20
    F_bins, T_frames = input_logpower.shape[1], input_logpower.shape[2]
    baseline_logpower = torch.log(
        torch.full((B_baseline, F_bins, T_frames), fill_value=model.eps, device="cuda")
    )  # → shape [20, 257, T_frames]

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
        with h5py.File(h5_filename, "a") as h5f:  # Open in read/write mode
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
