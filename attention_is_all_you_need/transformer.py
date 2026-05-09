import torch 
import torch.nn as nn
import math

def positional_encoding(seq_len, dmodel):
    pe = torch.zeros((seq_len, dmodel), dtype=torch.float32)

    pos = torch.arange(0, seq_len).unsqueeze(1)
    i = torch.arange(0, dmodel, 2)
    div_term = torch.exp(-i * math.log(10000) / dmodel)

    angles = pos * div_term

    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)

    return pe

class MultiHeadAttention(nn.Module):
    def __init__(self, dmodel, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.dk = dmodel // num_heads

        self.W_Q = nn.Linear(dmodel, dmodel)
        self.W_K = nn.Linear(dmodel, dmodel)
        self.W_V = nn.Linear(dmodel, dmodel)
        self.W_O = nn.Linear(dmodel, dmodel)

    def forward(self, x):
        B, T, C = x.shape

        Q = self.W_Q(x)
        K = self.W_Q(x)
        V = self.W_Q(x)

        Q = Q.view(B, T, self.num_heads, self.dk).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.dk).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.dk).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / (self.dk ** .5)
        weights = scores.softmax(dims=-1)
        out = weights @ V

        out = out.transpose(1, 2).contiguous(B, T, -1)
        return self.W_O(out)

class SubLayer(nn.Module):
    pass

class Layer(nn.Module):
    pass

class Encoder(nn.Module):
    pass

class Decoder(nn.Module):
    pass

class Transformer(nn.Module):
    pass



if __name__ == "__main__":
    pass