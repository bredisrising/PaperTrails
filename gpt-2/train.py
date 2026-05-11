import time
import math
import tiktoken
import torch
import pickle
from torch.utils.data.dataloader import DataLoader
from transformer import Transformer
from dataset import WebtextDataset

print("importing done.")

EPOCHS = 20
DMODEL = 384
LAYERS = 6
NUMHEADS = 6
MAX_SEQ_LEN = 512
BATCH_SIZE = 32
ACCUM_STEPS = 8 # effective batch ~= 524k tokens
LR = 1e-3
MIN_LR = 6e-5
WARMUP = 500 
STEPS_PER_EPOCH = 2000
NUM_WORKERS = 4
GRAD_CLIP = 1.0

# PAD, SOS, EOS = 0, 1, 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('high')

# with open("vocab.pkl", "rb") as f:
#     vocab = pickle.load(f)


enc = tiktoken.get_encoding('gpt2')
vocab_size = enc.n_vocab

model = Transformer(vocab_size, MAX_SEQ_LEN, DMODEL, NUMHEADS, LAYERS).to(device)

ckpt = torch.load('ckpt_2.pt', map_location=device)
sd = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
sd = {k.removeprefix('_orig_mod.'): v for k, v in sd.items()}
model.load_state_dict(sd)

model = torch.compile(model)

loss_fn = torch.nn.CrossEntropyLoss()

# wd only on 2d params
decay = [p for p in model.parameters() if p.dim() >= 2]
nodecay = [p for p in model.parameters() if p.dim() < 2]
optimizer = torch.optim.AdamW(
    [{"params": decay, "weight_decay": 0.1},
     {"params": nodecay, "weight_decay": 0.0}],
    lr=LR, betas=(0.9, 0.95), eps=1e-8,
)

# warmup then cosine to MIN_LR
def get_lr(step):
    if step < WARMUP:
        return LR * (step + 1) / WARMUP
    total = EPOCHS * STEPS_PER_EPOCH
    progress = (step - WARMUP) / max(1, total - WARMUP)
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * progress))

dataset = WebtextDataset("data.bin", MAX_SEQ_LEN)
dataloader = DataLoader(dataset, BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

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


global_step = 0
for epoch in range(3, EPOCHS):
    epoch_start = time.time()
    running_loss = 0.0
    steps_done = 0

    data_iter = iter(dataloader)
    for i in range(STEPS_PER_EPOCH):
        # set lr for this step
        lr = get_lr(global_step)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad()
        loss_accum = 0.0
        # accumulate ACCUM_STEPS micro batches then step
        for micro in range(ACCUM_STEPS):
            x, y = next(data_iter)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(x)
                # predict next token at every position: flatten (B,T,V) and (B,T)
                loss = loss_fn(logits.reshape(-1, vocab_size), y.reshape(-1)) / ACCUM_STEPS
            loss.backward()
            loss_accum += loss.detach()

        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        loss_val = loss_accum.item()
        running_loss += loss_val
        steps_done += 1
        global_step += 1

        if i % 20 == 0:
            elapsed = time.time() - epoch_start
            steps_per_sec = (i + 1) / elapsed
            print(
                f"epoch {epoch:>2} step {i:>5}/{STEPS_PER_EPOCH} "
                f"loss {loss_val:.4f} ppl {math.exp(loss_val):.2f} "
                f"lr {lr:.2e} ({steps_per_sec:.2f} it/s)",
                flush=True,
            )

    avg_loss = running_loss / max(steps_done, 1)
    epoch_time = time.time() - epoch_start
    print(
        f"\n[epoch {epoch} done] avg loss {avg_loss:.4f} "
        f"avg ppl {math.exp(avg_loss):.2f} time {epoch_time:.1f}s\n",
        flush=True,
    )

    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': global_step,
    }, f"ckpt_{epoch}.pt")
    print(f"saved ckpt_{epoch}.pt\n", flush=True)
