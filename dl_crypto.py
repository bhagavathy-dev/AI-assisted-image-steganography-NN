import torch
import torch.nn as nn

MAX_CHARS = 32
MSG_BITS = MAX_CHARS * 8
KEY_BITS = 64

class Encryptor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(MSG_BITS + KEY_BITS, 256),
            nn.ReLU(),
            nn.Linear(256, MSG_BITS),
            nn.Sigmoid()
        )

    def forward(self, msg, key):
        x = torch.cat([msg, key], dim=1)
        return self.net(x)

class Decryptor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(MSG_BITS + KEY_BITS, 256),
            nn.ReLU(),
            nn.Linear(256, MSG_BITS),
            nn.Sigmoid()
        )

    def forward(self, cipher, key):
        x = torch.cat([cipher, key], dim=1)
        return self.net(x)

def load_models(path="model_weights.pth"):
    enc = Encryptor()
    dec = Decryptor()
    state = torch.load(path, map_location="cpu")
    enc.load_state_dict(state["encryptor"])
    dec.load_state_dict(state["decryptor"])
    enc.eval()
    dec.eval()
    return enc, dec
