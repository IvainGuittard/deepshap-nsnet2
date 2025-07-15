import os
import sys
import torch
import torchaudio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


from DeepShap.config.parameters import n_fft


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


def amplify_tf_bins(waveform, spec_mask, dB):
    """
    Amplifies specific time-frequency bins in the waveform, indicated by a binary mask.

    Args:
        audio_path (str): Path to the audio file.
        sample_rate (int): Sampling rate of the audio.
        spec_mask (Tensor): Binary mask tensor of shape [F, T] indicating bins to amplify.
        dB (float): Gain in decibels to apply to the specified bins.

    Returns:
        wav_amplified (Tensor): Time-domain waveform after amplification.
    """
    spec_complex = torchaudio.transforms.Spectrogram(n_fft=n_fft, power=None)(
        waveform
    ).squeeze(
        0
    )  # shape: [F, T], complex
    magnitude = spec_complex.abs()
    phase = torch.angle(spec_complex)

    gain = 10 ** (dB / 10)
    spec_mask = spec_mask.to(magnitude.device)
    amplified_magnitude = magnitude * (1 + (gain - 1) * spec_mask)
    spec_amplified = amplified_magnitude * torch.exp(
        1j * phase
    )  # Reconstruct complex spectrum

    # Inverse STFT
    istft = torchaudio.transforms.InverseSpectrogram(n_fft=n_fft)
    wav_amplified = istft(spec_amplified.unsqueeze(0)).squeeze(0).squeeze(0)
    return wav_amplified


# if __name__ == "__main__":
#     input_path = "data/noisy_input_tests/p226_016.wav"
#     sample_rate = 16000
#     waveform, _ = load_and_resample(input_path, sample_rate)
#     spec_wav = torchaudio.transforms.Spectrogram(n_fft=n_fft, power=None)(waveform)
#     amp = amplify_tf_bins(waveform, sample_rate, torch.ones(spec_wav.shape), 15.0)
#     torchaudio.save("data/noisy_input_tests/p226_016_amplified.wav", amp.unsqueeze(0), sample_rate)
