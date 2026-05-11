import math
import time
import tiktoken
import torch
import sys
from transformer import Transformer

DMODEL = 384
LAYERS = 6
NUMHEADS = 6
MAX_SEQ_LEN = 512


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab


# claude generated this
# the raw moodel wont be an assistant more like continuation model ideally - fingers crossed
# prompts that play nice with openwebtext (news/blog/wiki style):
#   "The president announced today that"
#   "In a recent study, researchers found"
#   "According to the report,"
#   "BREAKING: "
#   "The first thing you need to know about"
#   "Posted by u/"                          # reddit-style, OWT was scraped from reddit links
#   "Q: What is  A:"
#   "def "                                  # OWT has a fair bit of code/tech blogs
# avoid: chat-style ("hi how are you"), instructions, anything assuming a fine-tuned model



text = '''The president announced today that'''

tokens = enc.encode(text)

model = Transformer(vocab_size, MAX_SEQ_LEN, DMODEL, NUMHEADS, LAYERS).to(device)

# new ckpts are dicts w/ 'model' key, old ones are bare state_dicts
ckpt = torch.load(sys.argv[1], map_location=device)
sd = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
# strip torch.compile prefix
sd = {k.removeprefix('_orig_mod.'): v for k, v in sd.items()}
model.load_state_dict(sd)

model.eval()

temperature = 1.0
top_k = 50


printed_text = enc.decode(tokens)
print(printed_text, end="", flush=True)

while True:
    tensor_tokens = torch.tensor(tokens[-MAX_SEQ_LEN:]).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor_tokens)
        logits = logits[:, -1, :] / temperature
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = -float('inf')
        probs = torch.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1).item()
    
    #if len(tokens) < MAX_SEQ_LEN:
    #    tokens.append(next_tok)
    #else:
    #    tokens = tokens[1:]
   
    tokens.append(next_tok)
    
    decoded = enc.decode(tokens)
    new_text = decoded[len(printed_text):]
       
    print(new_text, end="", flush=True)
    printed_text += new_text
    time.sleep(.125)

