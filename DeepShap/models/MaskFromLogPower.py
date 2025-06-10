import torch
import torch.nn as nn

class MaskFromLogPower(nn.Module):
    """
    A small wrapper that reuses NsNet2’s internal layers but
    *skips* the torchaudio STFT and log‐power steps. Instead,
    it expects `log_power` as input and returns `mask_pred`.
    """

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base = base_model

    def forward(self, log_power: torch.Tensor):
        """
        Input:
          log_power: real‐valued Tensor of shape [B, F, T]
                     (this is exactly what model._forward expects 
                      after you've done `log_stft_noisy = torch.log(|STFT|^2 + eps)`)
        Output:
          mask_pred: shape [B, 1, F, T], with values in [0,1].
        """
        B, F, T = log_power.shape
        # 1) Reconstruct the “RNN input” format [B, T, F]:
        x = log_power.permute(0, 2, 1)  # → [B, T, F]

        # 2) Pass through fc1 → rnn1 → rnn2 → fc2→ReLU→fc3→ReLU→fc4→Sigmoid
        #    We must create zero initial states for both GRUs:
        device = log_power.device
        h1_0 = torch.zeros(1, B, self.base.hidden_2, device=device)
        h2_0 = torch.zeros(1, B, self.base.hidden_2, device=device)

        x = self.base.fc1(x)              # → [B, T, hd1]
        x, h1 = self.base.rnn1(x, h1_0)   # → [B, T, hd2]
        x, h2 = self.base.rnn2(x, h2_0)   # → [B, T, hd2]
        x = self.base.fc2(x)              # → [B, T, hd3]
        x = nn.functional.relu(x)
        x = self.base.fc3(x)              # → [B, T, hd3]
        x = nn.functional.relu(x)
        x = self.base.fc4(x)              # → [B, T, n_feat]
        x = torch.sigmoid(x)              # mask values in [0,1]

        # 3) Now reshape back to [B, 1, F, T]:
        #    Right now `x` is [B, T, F], so permute back:
        mask_pred = x.permute(0, 2, 1).unsqueeze(1)  # → [B, 1, F, T]
        return mask_pred
    