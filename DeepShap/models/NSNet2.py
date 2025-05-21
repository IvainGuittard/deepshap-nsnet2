import torch, torchaudio
from torch import nn


class NsNet2(nn.Module):
    def __init__(self, n_fft, n_features, hidden_1, hidden_2, hidden_3):
        super().__init__()
        self.n_fft = n_fft
        self.n_features = n_features
        self.hidden_1 = hidden_1
        self.hidden_2 = hidden_2
        self.hidden_3 = hidden_3
        # fc1
        self.fc1 = nn.Linear(n_features, hidden_1)
        # rnn
        self.rnn1 = nn.GRU(input_size=hidden_1, hidden_size=hidden_2, num_layers=1, batch_first=True)
        self.rnn2 = nn.GRU(input_size=hidden_2, hidden_size=hidden_2, num_layers=1, batch_first=True)
        # fc2
        self.fc2 = nn.Linear(hidden_2, hidden_3)
        # fc3
        self.fc3 = nn.Linear(hidden_3, hidden_3)
        # fc4
        self.fc4 = nn.Linear(hidden_3, n_features)
        # eps
        self.eps = 1e-8
        self.preproc = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            power=None,
        )
        self.postproc = torchaudio.transforms.InverseSpectrogram(
            n_fft=n_fft,
        )

    def forward(self, x_noisy):
        stft_noisy = self.preproc(x_noisy)
        mask_pred = self._forward(stft_noisy)
        # apply mask
        stft_pred = stft_noisy * mask_pred
        x_pred = self.postproc(stft_pred)
        return x_pred

    def _forward(self, stft_noisy):
        # log power
        log_stft_noisy = torch.log(stft_noisy.abs() ** 2 + self.eps)
        # sort shape
        log_stft_noisy = log_stft_noisy.squeeze(1).permute(0, 2, 1)
        # neural network layers
        x = self.fc1(log_stft_noisy)
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        x = nn.functional.relu(x)
        x = self.fc4(x)
        x = torch.sigmoid(x)
        # sort shape
        mask_pred = x.permute(0, 2, 1).unsqueeze(1)
        return mask_pred
 
model = NsNet2(
    n_fft=512,
    n_features=257,
    hidden_1=400,
    hidden_2=400,
    hidden_3=600
)
