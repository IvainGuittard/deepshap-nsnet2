import torch

n_fft = 512
sample_rate = 16000
freqs = torch.fft.rfftfreq(n_fft, 1 / sample_rate)
bands = [
    (0, 250),
    (250, 1000),
    (1000, 3000),
    (3000, 8000)
]