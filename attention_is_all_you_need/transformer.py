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
    def __init__(self, dmodel, num_heads, masked=False):
        super().__init__()
        self.num_heads = num_heads
        self.dk = dmodel // num_heads

        self.causal = masked

        self.W_Q = nn.Linear(dmodel, dmodel)
        self.W_K = nn.Linear(dmodel, dmodel)
        self.W_V = nn.Linear(dmodel, dmodel)
        self.W_O = nn.Linear(dmodel, dmodel)

    def forward(self, x, context=None):
        B, T, C = x.shape

        Q = self.W_Q(x)

        kv_src = context if context is not None else x
        K = self.W_K(kv_src)
        V = self.W_V(kv_src)

        Q = Q.view(B, T, self.num_heads, self.dk).transpose(1, 2)
        K = K.view(B, -1, self.num_heads, self.dk).transpose(1, 2)
        V = V.view(B, -1, self.num_heads, self.dk).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / (self.dk ** .5)

        # ts the part that is needed for self-attention 
        # so that tokens can't attend to future tokens
        if self.causal:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float('-inf'))

        weights = scores.softmax(dim=-1)
        out = weights @ V

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_O(out)

class TransformerLayer(nn.Module):
    def __init__(self, seq_len, dmodel, numheads):
        super().__init__()
        self.attention = MultiHeadAttention(dmodel, numheads) 

        self.ff1 = nn.Linear(dmodel, dmodel * 4)
        self.ff2 = nn.Linear(dmodel * 4, dmodel)

        self.layernorm1 = nn.LayerNorm(dmodel)
        self.layernorm2 = nn.LayerNorm(dmodel)

    def forward(self, x):
        x = self.layernorm1(x + self.attention(x))
        x = self.layernorm2(x + self.ff2(torch.relu(self.ff1(x))))
        return x

class Encoder(nn.Module):
    pass

class Decoder(nn.Module):
    pass

class Transformer(nn.Module):
    pass



if __name__ == "__main__":
    pass