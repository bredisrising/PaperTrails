import pickle
import numpy as np
import torch
from pathlib import Path
import jsonlines
import json


# with open("data.pkl", "rb") as f:
#     data = pickle.load(f)

# print("data loaded")

# with jsonlines.open('data.jsonl', 'w') as f:
#     print('writing:')
#     print()
#     for i, row in enumerate(data):
#         f.write(row)
#         print(f"{i}/{len(data)}", end="\r")

with open("data.bin", 'wb') as bin_f:
    with jsonlines.open('data.jsonl', 'r') as f:
        for i, line in enumerate(f):
            arr = np.asarray(line, dtype=np.int16)
            arr.tofile(bin_f)
            print(f"{i}/500000", end='\r')
