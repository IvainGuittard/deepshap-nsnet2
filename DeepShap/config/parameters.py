import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""input_path = "p228_078.wav"
division = 64  # Number of time divisions per second"""
n_fft = 512
sample_rate = 16000
hop_length = n_fft // 4
freqs = torch.fft.rfftfreq(n_fft, 1 / sample_rate)
total_freqs = len(freqs)
freq_bands = [(1, 125), (125, 250), (250, 500), (500, 1000), (1000, 3000), (3000, 8000)]
