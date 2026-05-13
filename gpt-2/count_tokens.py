import numpy as np

data = np.memmap("data.bin", dtype=np.uint16, mode='r')
print(f"tokens: {len(data):,}")
print(f"tokens: {len(data)/1e6:.1f}M")
print(f"size:   {len(data)*2/1e9:.2f} GB")
