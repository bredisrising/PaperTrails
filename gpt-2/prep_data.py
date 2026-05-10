import numpy as np
import torch
import tiktoken
from pathlib import Path
import jsonlines
import json
import pickle
from datasets import load_dataset, load_dataset_builder

ds = load_dataset("openwebtext", split="train", streaming=True)

iterator = iter(ds)

enc = tiktoken.get_encoding('gpt2')

eot = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})

with open("data.bin", "wb") as bin_f:
    for i in range(1000000):
        text = next(iterator)
        tokens = enc.encode(text['text']) + eot
        arr = np.asarray(tokens, dtype=np.uint16)
        arr.tofile(bin_f)
        print(f"{i}/1000000", end='\r')
        
    


# with open("data.pkl", "rb") as f:
#     data = pickle.load(f)

# print("data loaded")

# with jsonlines.open('data.jsonl', 'w') as f:
#     print('writing:')
#     print()
#     for i, row in enumerate(data):
#         f.write(row)
#         print(f"{i}/{len(data)}", end="\r")


#with open("data.bin", 'wb') as bin_f:
#    with jsonlines.open('data.jsonl', 'r') as f:
#        for i, line in enumerate(f):
#            arr = np.asarray(line, dtype=np.int16)
#            arr.tofile(bin_f)
#            print(f"{i}/500000", end='\r')



