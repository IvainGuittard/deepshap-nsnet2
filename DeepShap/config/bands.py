import torch

n_fft = 512
sample_rate = 16000
freqs = torch.fft.rfftfreq(n_fft, 1 / sample_rate)
bands = [
    (0, 1000),
    (1000, 2000),
    (2000, 3000),
    (3000, 4000),
    (4000, 5000),
    (5000, 6000),
    (6000, 7000),
    (7000, 8000)
]