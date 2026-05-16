import torch
import torch.nn as nn


class Maxout(nn.Module):
    def __init__(self, chunks):
        super().__init__()
        self.chunks = chunks

    def forward(self, x):
        batches, features = x.shape
        out_features = features // self.chunks
        x = x.view(batches, out_features, self.chunks) 
        return x.max(dim=2).values

# Fully Connected
class FullyConnectedGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, chunks=4):
        super().__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(.2),
            nn.Linear(input_size, hidden_size * chunks),
            Maxout(chunks),
            nn.Dropout(),
            nn.Linear(hidden_size, hidden_size * chunks),
            Maxout(chunks),
            nn.Dropout(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
