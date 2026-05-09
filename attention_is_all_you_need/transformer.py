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
    def __init__(self, dmodel, numheads, causal=False, cross_attention=False):
        super().__init__()
        self.self_attention = MultiHeadAttention(dmodel, numheads, causal) 
        self.cross_attention = MultiHeadAttention(dmodel, numheads) if cross_attention else None

        self.ff1 = nn.Linear(dmodel, dmodel * 4)
        self.ff2 = nn.Linear(dmodel * 4, dmodel)

        self.layernorm1 = nn.LayerNorm(dmodel)
        self.layernorm2 = nn.LayerNorm(dmodel)
        self.layernorm3 = nn.LayerNorm(dmodel) if cross_attention else None

    def forward(self, x, context=None):
        x = self.layernorm1(x + self.self_attention(x))
        if self.cross_attention is not None and context is not None:
            x = self.layernorm2(x + self.cross_attention(x, context))
            x = self.layernorm3(x + self.ff2(torch.relu(self.ff1(x))))
        else:
            x = self.layernorm2(x + self.ff2(torch.relu(self.ff1(x))))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, seq_len, dmodel, num_heads, num_layers):
        super().__init__() 
        self.embedding = nn.Embedding(vocab_size, dmodel)
        self.seq_len = seq_len
        self.dmodel = dmodel

        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(TransformerLayer(dmodel, num_heads))
    
    def forward(self, x):
        T = x.shape[1]
        x = self.embedding(x)
        x = x + positional_encoding(T, self.dmodel).to(x.device)

        for layer in self.layers:
            x = layer(x)

        return x



class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, seq_len, dmodel, num_heads, num_layers):
        super().__init__() 
        self.embedding = nn.Embedding(vocab_size, dmodel)
        self.seq_len = seq_len
        self.dmodel = dmodel

        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(TransformerLayer(dmodel, num_heads, True, True))
        
        self.end_linear = nn.Linear(dmodel, vocab_size)
    
    def forward(self, x, encoder_output):
        T = x.shape[1]
        x = self.embedding(x)
        x = x + positional_encoding(T, self.dmodel).to(x.device)

        for layer in self.layers:
            x = layer(x, context=encoder_output)

        x = self.end_linear(x)

        return x
    
class Transformer(nn.Module):
    def __init__(self, vocab_size, seq_len, dmodel, num_heads, num_layers):
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size, seq_len, dmodel, num_heads, num_layers)
        self.decoder = TransformerDecoder(vocab_size, seq_len, dmodel, num_heads, num_layers)

    def forward(self, src, tgt):
        encoder_output = self.encoder(src)
        return self.decoder(tgt, encoder_output)

if __name__ == "__main__":
    pass