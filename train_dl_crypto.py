import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
import string

# ---------------- CONFIG ----------------
MAX_CHARS = 32
MSG_BITS = MAX_CHARS * 8
KEY_BITS = 64
EPOCHS = 30
BATCH_SIZE = 64
LR = 0.001

# ---------------- UTILITIES ----------------
def normalize_text(msg):
    msg = msg[:MAX_CHARS]
    return msg.ljust(MAX_CHARS, "~")

def text_to_bits(text):
    text = normalize_text(text)
    bits = []
    for c in text:
        b = format(ord(c), '08b')
        bits.extend([int(x) for x in b])
    return np.array(bits, dtype=np.float32)

def key_to_bits(key):
    key = key[:KEY_BITS].ljust(KEY_BITS, "0")
    return np.array([int(b) for b in key], dtype=np.float32)

def random_message():
    chars = string.ascii_letters + string.digits + " .,!?@#"
    l = random.randint(5, MAX_CHARS)
    return ''.join(random.choice(chars) for _ in range(l))

def random_key():
    return ''.join(random.choice("01") for _ in range(KEY_BITS))

# ---------------- DATASET ----------------
def generate_dataset(n=10000):
    X_msg, X_key, Y = [], [], []
    for _ in range(n):
        msg = random_message()
        key = random_key()
        msg_bits = text_to_bits(msg)
        key_bits = key_to_bits(key)

        X_msg.append(msg_bits)
        X_key.append(key_bits)
        Y.append(msg_bits)

    return np.array(X_msg), np.array(X_key), np.array(Y)

# ---------------- MODELS ----------------
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

# ---------------- TRAINING ----------------
def train():
    print("Generating dataset...")
    X_msg, X_key, Y = generate_dataset(15000)

    dataset = TensorDataset(
        torch.tensor(X_msg), 
        torch.tensor(X_key), 
        torch.tensor(Y)
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    encryptor = Encryptor()
    decryptor = Decryptor()

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        list(encryptor.parameters()) + list(decryptor.parameters()), lr=LR
    )

    print("Training started...")
    for epoch in range(EPOCHS):
        total_loss = 0
        for msg_bits, key_bits, target in loader:
            cipher = encryptor(msg_bits, key_bits)
            decoded = decryptor(cipher, key_bits)

            loss = criterion(decoded, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(loader):.6f}")

    # Save trained models
    torch.save({
        "encryptor": encryptor.state_dict(),
        "decryptor": decryptor.state_dict()
    }, "model_weights.pth")

    print("Training completed.")
    print("Model saved as model_weights.pth")

if __name__ == "__main__":
    train()
