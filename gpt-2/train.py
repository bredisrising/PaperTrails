import time
import math
import tiktoken
import torch
import pickle
from torch.utils.data.dataloader import DataLoader
from transformer import Transformer
from dataset import WebtextDataset

print("importing done.")

EPOCHS = 10
DMODEL = 384
LAYERS = 6
NUMHEADS = 6
MAX_SEQ_LEN = 512
BATCH_SIZE = 32
LR = 2.5e-4
STEPS_PER_EPOCH = 2000
NUM_WORKERS = 4

PAD, SOS, EOS = 0, 1, 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

vocab_size = len(vocab["tokens"])

model = Transformer(vocab_size, MAX_SEQ_LEN, DMODEL, NUMHEADS, LAYERS).to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

dataset = WebtextDataset("data.bin", MAX_SEQ_LEN)
dataloader = DataLoader(dataset, BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)

n_params = sum(p.numel() for p in model.parameters())
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("=" * 60)
print("MODEL ARCHITECTURE")
print("=" * 60)
print(model)
print("=" * 60)
print(f"device:           {device}")
print(f"vocab size:       {vocab_size:,}")
print(f"max seq len:      {MAX_SEQ_LEN}")
print(f"dmodel:           {DMODEL}")
print(f"layers:           {LAYERS}")
print(f"heads:            {NUMHEADS}")
print(f"batch size:       {BATCH_SIZE}")
print(f"learning rate:    {LR}")
print(f"epochs:           {EPOCHS}")
print(f"total params:     {n_params:,}  (~{n_params/1e6:.2f}M)")
print(f"trainable params: {n_trainable:,}")
print(f"dataset samples:  {len(dataset):,}")
print(f"steps per epoch:  {STEPS_PER_EPOCH:,}  (capped; full pass would be {len(dataloader):,})")
print("=" * 60)


for epoch in range(EPOCHS):
    epoch_start = time.time()
    running_loss = 0.0
    steps_done = 0

    for i, (x, y) in enumerate(dataloader):
        if i >= STEPS_PER_EPOCH:
            break

        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        logits = model(x)
        # predict next token at every position: flatten (B,T,V) and (B,T)
        loss = loss_fn(logits.reshape(-1, vocab_size), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        steps_done += 1

        if i % 20 == 0:
            elapsed = time.time() - epoch_start
            steps_per_sec = (i + 1) / elapsed
            print(
                f"epoch {epoch:>2} step {i:>5}/{STEPS_PER_EPOCH} "
                f"loss {loss.item():.4f} ppl {math.exp(loss.item()):.2f} "
                f"({steps_per_sec:.1f} it/s)",
                flush=True,
            )

    avg_loss = running_loss / max(steps_done, 1)
    epoch_time = time.time() - epoch_start
    print(
        f"\n[epoch {epoch} done] avg loss {avg_loss:.4f} "
        f"avg ppl {math.exp(avg_loss):.2f} time {epoch_time:.1f}s\n",
        flush=True,
    )

    torch.save(model.state_dict(), f"ckpt_{epoch}.pt")
    print(f"saved ckpt_{epoch}.pt\n", flush=True)
