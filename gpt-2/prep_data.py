import numpy as np
import torch
import tiktoken
from pathlib import Path
import jsonlines
import json
import pickle
from datasets import load_dataset, load_dataset_builder

ds = load_dataset("HuggingFaceFW/fineweb-edu", name="CC-MAIN-2024-10", split="train", streaming=True)

iterator = iter(ds)

enc = tiktoken.get_encoding('gpt2')

eot = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})

grabbed = 0
doc_count = 0

SKIP_DOCS = 2_433_000

try:
    with open("data.bin", "ab") as bin_f:
        while grabbed < 6_000_000_000:
            doc_count += 1
            try: 
                text = next(iterator)
            except Exception as e:
                print(f"error")
                break 
            
            if doc_count <= SKIP_DOCS:
                if doc_count % 10000 == 0:
                    print(f"skipping...", flush=True)
                continue
            
            tokens = enc.encode(text['text'], disallowed_special=()) + eot

            grabbed += len(tokens)

            arr = np.asarray(tokens, dtype=np.uint16)
            arr.tofile(bin_f)

            if doc_count % 1000 == 0: 
                print(f"{grabbed}/6_000_000_000", flush=True)
except Exception as e:
    print(f"{e}", flush=True) 
    


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



