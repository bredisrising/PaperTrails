import torch
import numpy as np
from torch.utils.data.dataset import Dataset

class WebtextDataset(Dataset):
    def __init__(self, path, seq_len, virtual_len=1_000_000):
        self.path = path
        self.seq_len = seq_len
        self.virtual_len = virtual_len
        self.data = np.memmap(path, dtype=np.uint16, mode='r')
        self.max_start = len(self.data) - seq_len - 1

    def __len__(self):
        return self.virtual_len

    def __getitem__(self, idx):
        # idx is ignored - sample a random window each call
        start = np.random.randint(0, self.max_start)
        chunk = torch.tensor(self.data[start : start + self.seq_len + 1], dtype=torch.long)
        return chunk[:-1], chunk[1:]   # input, label (both length seq_len, label shifted by 1)
    

if __name__ == "__main__":
    dataset = WebtextDataset("data.bin", 32)
