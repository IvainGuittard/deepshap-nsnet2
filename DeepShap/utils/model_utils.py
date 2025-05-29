import torch
import sys
import os
from models.NsNet2_model import NsNet2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_nsnet2_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = NsNet2(n_fft=512, n_feat=257, hd1=400, hd2=400, hd3=600)
    # Load the pre-trained weights
    weights_path = "DeepShap/models/nsnet2_baseline.bin"
    state_dict = torch.load(weights_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device

