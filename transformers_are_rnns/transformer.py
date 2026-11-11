import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt

#print(f"CUDA: {torch.cuda.is_available()}")
print()

SEQ_LEN = 16 
TOKEN_DIM = 1024  
ATTENTION_PROJECTION = TOKEN_DIM

class QuadraticAttention(nn.Module):
    def __init__(self, token_dim):
        super().__init__()

        self.dk = ATTENTION_PROJECTION 

        self.Q_w = nn.Linear(token_dim,  ATTENTION_PROJECTION) 
        self.K_w = nn.Linear(token_dim, ATTENTION_PROJECTION)
        self.V_w = nn.Linear(token_dim, ATTENTION_PROJECTION)

    def forward(self, x):
        T, C = x.shape
        queries = self.Q_w(x) 
        keys = self.K_w(x) 
        values = self.V_w(x) 
        
        out = F.scaled_dot_product_attention(queries, keys, values, is_causal=False)

        return out 


class LinearAttention(nn.Module): 
    def __init__(self, token_dim):
        super().__init__()

        self.dk = ATTENTION_PROJECTION 

        self.Q_w = nn.Linear(token_dim,  ATTENTION_PROJECTION) 
        self.K_w = nn.Linear(token_dim, ATTENTION_PROJECTION)
        self.V_w = nn.Linear(token_dim, ATTENTION_PROJECTION)
            
    def feature_map(self, x):
        return F.elu(x) + 1
        
    def forward(self, x):
        queries = self.Q_w(x) 
        keys = self.K_w(x)
        values = self.V_w(x)

        queries = self.feature_map(queries)
        keys = self.feature_map(keys)
        
        denominator = queries @ keys.sum(dim=0, keepdim=True).T

        tokens = queries @ (keys.T @ values) / denominator

        return tokens




if __name__ == "__main__":
    quad_attention = QuadraticAttention(TOKEN_DIM) 
    lin_attention = LinearAttention(TOKEN_DIM)

    token_sequence = torch.randn((SEQ_LEN, TOKEN_DIM))
    
    n = 16
    q = 16

    quad_times = []
    print("Quadratic Attention:")
    for i in range(q):
        seq_len = 2**i
        token_sequence = torch.randn((seq_len, TOKEN_DIM))
        start_time = time.time()
        quad_tokens = quad_attention(token_sequence)
        end_time = time.time()

        quad_times.append(end_time - start_time)

        print(f"{2**i}: {end_time-start_time}")
 
    lin_times = [] 
    print("\nLinear Attention:")
    for i in range(n):
        seq_len = 2**i
        token_sequence = torch.randn((seq_len, TOKEN_DIM))

        start_time = time.time()
        quad_tokens = lin_attention(token_sequence)
        end_time = time.time()

        lin_times.append(end_time - start_time)

        print(f"{i}, {2**i}, {seq_len * TOKEN_DIM * 4 / 1e6:.2f}MB: {end_time-start_time}")
    plt.plot([2**i for i in range(n)], lin_times, label="lin")
    plt.plot([2**i for i in range(n)], quad_times, label="quad")
    plt.legend()
    plt.show()
        
       



