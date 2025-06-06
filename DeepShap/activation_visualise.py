import os
import dill
from PIL import Image

import torch
import torchaudio
import torchlens as tl
import matplotlib.pyplot as plt
from torch import nn

from utils.model_utils import load_nsnet2_model

# ───────────────────────────────────────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────────────────────────────────────

# Path to a clean WAV file for testing
X_TEST_PATH = "path/to/your/clean_audio.wav"

# Directory where frame images and GIFs will be saved
FRAME_DIR = "path/to/save/frames"
GIF_DIR = "path/to/save/gifs"

# Random seed for reproducibility
torch.manual_seed(42)

# Noise parameters
NUM_LEVELS     = 50
BURST_PROB     = 0.0005    # 0.05% chance to start a burst at any sample index
BURST_LENGTH   = 400       # each burst spans 400 consecutive time steps
NOISE_STD      = 1.0       # standard deviation for pre‐generated white noise
USE_BURST_NOISE = True     # toggle burst‐noise behavior


# ───────────────────────────────────────────────────────────────────────────────
# Helper Classes and Functions
# ───────────────────────────────────────────────────────────────────────────────

class MaskFromLogPower(nn.Module):
    """
    Wraps the NSNet2 model so that it directly accepts a log‐power spectrogram
    (shape [B, F, T]) and returns the predicted mask (shape [B, 1, F, T]).
    """
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base = base_model

    def forward(self, log_power: torch.Tensor):
        """
        Args:
            log_power: Tensor of shape [B, F, T]
                       (i.e. log(|STFT|^2 + eps), which NSNet2 normally computes internally)

        Returns:
            mask_pred: Tensor of shape [B, 1, F, T] with values in [0, 1].
        """
        B, F, T = log_power.shape
        device = log_power.device

        # 1) Rearrange to RNN‐friendly format [B, T, F]
        x = log_power.permute(0, 2, 1)  # → [B, T, F]

        # 2) Initialize RNN hidden states
        h1_0 = torch.zeros(1, B, self.base.hidden_2, device=device)
        h2_0 = torch.zeros(1, B, self.base.hidden_2, device=device)

        # 3) Forward pass through NSNet2’s layers (fc1, rnn1, rnn2, fc2 → ReLU, fc3 → ReLU, fc4 → Sigmoid)
        x = self.base.fc1(x)                     # → [B, T, hidden_1]
        x, _h1 = self.base.rnn1(x, h1_0)         # → [B, T, hidden_2]
        x, _h2 = self.base.rnn2(x, h2_0)         # → [B, T, hidden_2]
        x = self.base.fc2(x)                     # → [B, T, hidden_3]
        x = nn.functional.relu(x)
        x = self.base.fc3(x)                     # → [B, T, hidden_3]
        x = nn.functional.relu(x)
        x = self.base.fc4(x)                     # → [B, T, F]
        x = torch.sigmoid(x)                     # mask values in [0,1]

        # 4) Reshape back to [B, 1, F, T]
        mask_pred = x.permute(0, 2, 1).unsqueeze(1)
        return mask_pred


def generate_burst_noise(batch_size: int, length: int, device: torch.device):
    """
    Pre‐generate a reservoir of white noise, then carve out occasional bursts.
    Returns a tensor of shape [1, 1, length] containing zero‐padded bursts.
    """
    # 1) Create a full white‐noise reservoir [1, 1, length]
    rand_noise = torch.randn(1, 1, length, device=device) * NOISE_STD

    # 2) Initialize an all‐zeros buffer for burst_noise
    burst_noise = torch.zeros_like(rand_noise, device=device)

    # 3) Randomly copy slices from rand_noise into burst_noise
    i = 0
    while i < length:
        if torch.rand(1).item() < BURST_PROB:
            end = min(i + BURST_LENGTH, length)
            burst_noise[:, :, i:end] = rand_noise[:, :, i:end]
            i = end  # skip to end of this burst
        else:
            i += 1

    return burst_noise  # shape [1, 1, length]


def create_noisy_batch(x_clean: torch.Tensor, device: torch.device):
    """
    Given a clean waveform (shape [1, 1, L]), build a batch of noisy waveforms
    at NUM_LEVELS different amplitudes (SNRs). Returns:
        x_noisy_batch: [NUM_LEVELS, 1, 2*L]
        x_clean_batch: [NUM_LEVELS, 1, L]
        amplitude_noise: [NUM_LEVELS]  (the scale factors applied to pre‐generated noise)
        snr_values: [NUM_LEVELS]       (in dB, for debugging/plotting)

    If USE_BURST_NOISE is True, the same burst structure is repeated across all levels.
    """
    _, _, L = x_clean.shape

    # 1) Compute a linspace of amplitude scales, and their corresponding SNRs
    amplitude_noise = torch.linspace(0.0001, 0.07, steps=NUM_LEVELS, device=device)  # [50]
    snr_values = 10 * torch.log10((x_clean.abs().mean() ** 2) / (amplitude_noise**2))
    print(f"Noise levels (SNR in dB): {snr_values.cpu().numpy()}")

    # 2) Repeat the clean waveform for each noise level
    x_clean_batch = x_clean.repeat(NUM_LEVELS, 1, 1)  # [50, 1, L]

    # 3) Generate one burst_noise reservoir of shape [1, 1, L]
    if USE_BURST_NOISE:
        burst_noise = generate_burst_noise(NUM_LEVELS, L, device)
        plt.figure(figsize=(10, 4))
        plt.plot(burst_noise.cpu().squeeze(), label="Burst Noise (one channel)")
        plt.title("Debug: Burst Noise Waveform")
        plt.xlabel("Time Index")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.show()

        # Repeat that same burst pattern across all NUM_LEVELS
        rand_noise = burst_noise.repeat(NUM_LEVELS, 1, 1)  # [50, 1, L]
    else:
        # If no burst noise, just generate white noise and repeat
        rand_noise = torch.randn(NUM_LEVELS, 1, L, device=device) * NOISE_STD

    # 4) Scale each copy by its amplitude factor
    noise = amplitude_noise.view(NUM_LEVELS, 1, 1) * rand_noise  # [50, 1, L]

    # 5) Add noise to clean waveform → [50, 1, L]
    x_noisy_batch = x_clean_batch + noise

    # 6) Concatenate each waveform with itself to double the length: [50, 1, 2L]
    x_noisy_batch = torch.cat((x_noisy_batch, x_noisy_batch), dim=-1)

    return x_noisy_batch, x_clean_batch, amplitude_noise, snr_values


def visualize_and_save_activations(model_history, snr_values, output_dir):
    """
    For each captured layer in model_history, create an animated GIF of the
    layer’s activations as noise amplitude increases.
    """
    os.makedirs(output_dir, exist_ok=True)

    for layer_name, layer_act in model_history.layer_dict_main_keys.items():
        # Skip unused layers by name if desired
        if layer_name in {'zeros_1_2', 'zeros_2_3', 'gru_1_5:2', 'gru_2_6:2'}:
            continue

        acts = model_history[layer_name].tensor_contents  # e.g. [NUM_LEVELS, T, F] or [1, NUM_LEVELS, F, T]
        acts = acts.squeeze(0)  # remove leading batch dim if present

        # If shape is [NUM_LEVELS, F, T], swap to [NUM_LEVELS, T, F]
        if acts.ndim == 3 and acts.shape[1] == 257:
            acts = acts.permute(0, 2, 1)

        # Keep only the second half of the time dimension
        T = acts.shape[1]
        acts = acts[:, T // 2 :, :]  # [NUM_LEVELS, T/2, F]

        acts_np = acts.cpu().detach().numpy()
        snr_np = snr_values.cpu().numpy()

        # Global color bounds for all frames in this layer
        vmin, vmax = acts_np.min(), acts_np.max()

        # Directory for this layer’s frames
        layer_frame_dir = os.path.join(output_dir, f"frames_{layer_name}")
        os.makedirs(layer_frame_dir, exist_ok=True)

        frame_filenames = []
        for i in range(acts_np.shape[0]):
            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(
                acts_np[i].T,
                origin='lower',
                aspect='auto',
                vmin=vmin,
                vmax=vmax,
                cmap='viridis'
            )
            ax.set_title(f"{layer_name} @ SNR={snr_np[i]:.2f} dB")
            ax.set_xlabel("Time Index")
            ax.set_ylabel("Feature Index")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Activation Value")

            # Save frame
            fname = os.path.join(layer_frame_dir, f"frame_{i:03d}.png")
            plt.savefig(fname, bbox_inches='tight')
            plt.close(fig)
            frame_filenames.append(fname)

        # Build GIF for this layer
        frames = [Image.open(fname) for fname in frame_filenames]
        gif_path = os.path.join(output_dir, f"activations_{layer_name}.gif")
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=200,    # 200 ms per frame
            loop=0           # loop forever
        )
        print(f"Saved GIF for layer '{layer_name}' → {gif_path}")


# ───────────────────────────────────────────────────────────────────────────────
# Main Execution
# ───────────────────────────────────────────────────────────────────────────────

def main():
    # Load NSNet2 base model
    base_model, device = load_nsnet2_model()
    base_model = base_model.to(device).eval()

    # Wrap NSNet2 so we can feed it log‐power directly
    mask_model = MaskFromLogPower(base_model).to(device).eval()

    # Load and (re)sample the clean audio
    x_test, sample_rate = torchaudio.load(X_TEST_PATH)  # shape [1, waveform_len]
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        x_test = resampler(x_test)
    x_test = x_test.unsqueeze(0).to(device)  # → [1, 1, L]

    # Build a batch of noisy waveforms [NUM_LEVELS, 1, 2*L]
    x_noisy_batch, x_clean_batch, amplitude_noise, snr_values = create_noisy_batch(x_test, device)

    # Compute complex spectrograms and log‐power
    spec_complex = base_model.preproc(x_noisy_batch)  # [NUM_LEVELS, 1, F, T] (complex)
    log_power_test = torch.log(spec_complex.abs() ** 2 + base_model.eps).squeeze(1)  # [NUM_LEVELS, F, T]

    # Log forward pass through MaskFromLogPower with TorchLens
    model_history = tl.log_forward_pass(
        mask_model,
        log_power_test,
        layers_to_save="all",
        vis_opt="rolled"
    )

    # Save the activation dictionary to disk
    torch.save(model_history, "nsnet2_activations.pt", pickle_module=dill)
    print("Saved torchlens activation history → nsnet2_activations.pt")

    # Visualize activations as animated GIFs
    visualize_and_save_activations(model_history, snr_values, FRAME_DIR)


if __name__ == "__main__":
    main()
