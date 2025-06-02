import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torchaudio
import torch
import json


def batchify_targets(targets, batch_size):
    """Yield successive batches from the targets list."""
    for i in range(0, len(targets), batch_size):
        yield targets[i : i + batch_size]


def load_and_resample(path, target_sr):
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    return waveform, target_sr


def compute_time_bands(waveform, sample_rate, hop_length=512, division=10):
    """
    Computes time bands based on the number of frames in the input audio.
    Args:
        waveform (torch.Tensor): The input audio waveform.
        division (int): The number of divisions for the time bands.
    Returns:
        tuple: A tuple containing time bands in seconds, and time bands in samples.
    """
    nb_frames = waveform.shape[-1]
    time_bands_sec = [
        (i / division, (i + 1) / division)
        for i in range(int(division * nb_frames / sample_rate))
    ]
    time_bands = [
        (int(start * sample_rate / hop_length), int(end * sample_rate / hop_length))
        for start, end in time_bands_sec
    ]
    return time_bands_sec, time_bands


def is_already_processed(
    file_basename,
    division,
    noise_type,
    baseline_type,
    freq_range=None,
    rms_amplitude=None,
):
    """
    Check if the input file has already been processed by looking for its attribution map.
    Args:
        input_path (str): Path to the input audio file.
    Returns:
        bool: True if the file has al
        ready been processed, False otherwise.
    """
    folder_name_1 = file_basename.replace(".wav", "")
    file_name_1 = file_basename.replace(".wav", f"_div{division}_attribution_map.json")
    if freq_range is not None and rms_amplitude is not None:
        file_name_1 = file_basename.replace(
            ".wav",
            f"_div{division}_freq_{freq_range[0]}_{freq_range[1]}_rms_{rms_amplitude}_attribution_map.json",
        )
    save_path_1 = f"DeepShap/attributions/{noise_type}_noise_{baseline_type}_baseline/maps/{folder_name_1}/{file_name_1}"

    folder_name_2 = file_basename.replace(".wav", "")
    file_name_2 = file_basename.replace(".wav", f"_div{division}_attribution_plot.png")
    if freq_range is not None and rms_amplitude is not None:
        file_name_2 = file_basename.replace(
            ".wav",
            f"_div{division}_freq_{freq_range[0]}_{freq_range[1]}_rms_{rms_amplitude}_attribution_plot.png",
        )
    save_path_2 = f"DeepShap/attributions/{noise_type}_noise_{baseline_type}_baseline/plots/{folder_name_2}/{file_name_2}"

    if os.path.exists(save_path_1) and os.path.exists(save_path_2):
        print(
            f"\nAlready processed {file_basename} : division {division}, noise type {noise_type}"
        )
        if freq_range is not None and rms_amplitude is not None:
            print(f"Frequency range {freq_range}, RMS amplitude {rms_amplitude}")
        return True
    return False


def save_attr_map(
    attr_map,
    file_basename,
    freq_bands,
    division,
    noise_type,
    baseline_type,
    freq_range=None,
    rms_amplitude=None,
):
    folder_name = file_basename.replace(".wav", "")
    file_name = file_basename.replace(".wav", f"_div{division}_attribution_map.json")
    if freq_range is not None and rms_amplitude is not None:
        file_name = file_basename.replace(
            ".wav",
            f"_div{division}_freq_{freq_range[0]}_{freq_range[1]}_rms_{rms_amplitude}_attribution_map.json",
        )
    save_path = f"DeepShap/attributions/{noise_type}_noise_{baseline_type}_baseline/maps/{folder_name}/{file_name}"

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    new_entry = {
        "frequency_bands": freq_bands,
        "division": division,
        "attributions": {
            f"{freq_bands[i][0]}-{freq_bands[i][1]}Hz_time_band_{j}": float(value)
            for i, band in enumerate(attr_map)
            for j, value in enumerate(band)
        },
    }

    with open(save_path, "w") as f:
        json.dump(new_entry, f, indent=4)

    print(f"Attribution map saved to {save_path}")
    return


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


def add_impulsive_noise(waveform, probability=0.001, amplitude=0.5):
    device = waveform.device
    noise_mask = torch.bernoulli(torch.full_like(waveform, probability)).to(device)
    impulses = (2 * torch.rand_like(waveform) - 1) * amplitude
    impulsive_noise = noise_mask * impulses
    return waveform + impulsive_noise


def prepare_deepshap_input_and_baseline(
    input_path,
    file_basename,
    noise_type,
    baseline_type,
    device,
    sample_rate,
    freq_range=None,
    rms_amplitude=None,
):
    baseline = None

    if noise_type == "not_added":
        deepshap_input, _ = load_and_resample(input_path, sample_rate)
        deepshap_input_path = input_path
        if baseline_type == "clean_audio":
            baseline, _ = load_and_resample(
                f"data/clean_trainset_28spk_wav/{file_basename}", sample_rate
            )
            baseline = baseline.unsqueeze(0).to(device).repeat(10, 1, 1)
        elif baseline_type == "zero":
            baseline = torch.zeros((10, 1, deepshap_input.shape[-1]), device=device)

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
        deepshap_input_path = f"data/added_noise_input/{file_basename.replace('.wav', '')}_freq_{freq_range[0]}-{freq_range[1]}_rms_{rms_amplitude}.wav"
        torchaudio.save(deepshap_input_path, deepshap_input, sample_rate=sample_rate)

        if baseline_type == "clean_audio":
            baseline, _ = load_and_resample(input_path, sample_rate)
            baseline = baseline.unsqueeze(0).to(device).repeat(10, 1, 1)
        elif baseline_type == "zero":
            baseline = torch.zeros((10, 1, deepshap_input.shape[-1]), device=device)

    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    return deepshap_input, baseline, deepshap_input_path
