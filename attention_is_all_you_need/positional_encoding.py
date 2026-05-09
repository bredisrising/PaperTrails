import torch
import math
import matplotlib.pyplot as plt

# PE(pos, 2i) = sin(pos/10000^2i/dmodel)
# PE(pos, 2i+1) = cos(pos/10000^2i/dmodel)


# input = (batch, seq_len, dmodel)


BATCH = 1
SEQ_LEN = 512
DMODEL = 8

# print("TEST ARANGE FOR POSITIONAL ENCODING")
# test_arange = torch.permute(torch.stack([torch.arange(0, SEQ_LEN)] * DMODEL), (1, 0))
# print(test_arange, test_arange.shape)

input = torch.randn((4, SEQ_LEN, DMODEL))

# print()
# print("EXAMPLE INPUT:")
# print(input)

def positional_encoding():
    pe = torch.zeros((SEQ_LEN, DMODEL), dtype=torch.float32)

    pos = torch.arange(0, SEQ_LEN).unsqueeze(1)
    i = torch.arange(0, DMODEL, 2)
    div_term = torch.exp(-i * math.log(10000) / DMODEL)

    angles = pos * div_term

    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)

    return pe


pe = positional_encoding()    

fig, axes = plt.subplots(DMODEL, 1, figsize=(10, 8))
for dim in range(DMODEL):
    axes[dim].plot(pe[:, dim])
    axes[dim].set_ylabel(f"dim {dim}")
plt.tight_layout()
plt.show()

