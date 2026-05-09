import torch
import torch.nn as nn
from toy_arithmetic_data import generate_single_sample
from transformer import Transformer


if __name__ == "__main__":
    EPOCHS = 10
    DMODEL = 64
    LAYERS = 2
    NUMHEADS = 4
    DATA_SAMPLES = 1024 * 8
    BATCH_SIZE = 128
    SRC_LEN = 5   # max "10*10"
    TGT_LEN = 4   # max "-10" + EOS

    PAD, SOS, EOS = 0, 1, 2
    vocab = ["<pad>", "<sos>", "<eos>",
             "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
             "+", "-", "*", "/", "="]
    w2i = {w: i for i, w in enumerate(vocab)}
    i2w = {i: w for i, w in enumerate(vocab)}
    VOCAB_SIZE = len(vocab)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}, vocab: {VOCAB_SIZE}")

    # build dataset
    src_data, dec_in_data, dec_out_data = [], [], []
    for _ in range(DATA_SAMPLES):
        s_in, s_out = generate_single_sample()

        src = [w2i[c] for c in s_in]
        src += [PAD] * (SRC_LEN - len(src))

        body = [w2i[c] for c in s_out]
        dec_in = [SOS] + body + [PAD] * (TGT_LEN - 1 - len(body))
        dec_out = body + [EOS] + [PAD] * (TGT_LEN - 1 - len(body))

        src_data.append(src)
        dec_in_data.append(dec_in)
        dec_out_data.append(dec_out)

    src_data = torch.tensor(src_data, dtype=torch.long, device=device)
    dec_in_data = torch.tensor(dec_in_data, dtype=torch.long, device=device)
    dec_out_data = torch.tensor(dec_out_data, dtype=torch.long, device=device)

    model = Transformer(VOCAB_SIZE, max(SRC_LEN, TGT_LEN), DMODEL, NUMHEADS, LAYERS).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters())}")

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)

    n_batches = DATA_SAMPLES // BATCH_SIZE
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(DATA_SAMPLES, device=device)
        total_loss = 0.0

        for b in range(n_batches):
            idx = perm[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            logits = model(src_data[idx], dec_in_data[idx])
            loss = loss_fn(logits.reshape(-1, VOCAB_SIZE), dec_out_data[idx].reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # token accuracy on first 1024 samples 
        model.eval()
        with torch.no_grad():
            logits = model(src_data[:1024], dec_in_data[:1024])
            preds = logits.argmax(dim=-1)
            mask = dec_out_data[:1024] != PAD
            tok_acc = (preds[mask] == dec_out_data[:1024][mask]).float().mean().item()

        print(f"epoch {epoch+1:2d} | loss {total_loss/n_batches:.4f} | tok_acc {tok_acc:.3f}")

    # greedy decode a few examples
    print("\npredictions:")
    model.eval()
    with torch.no_grad():
        sample = src_data[:10]
        enc = model.encoder(sample)
        tgt = torch.full((10, 1), SOS, dtype=torch.long, device=device)
        for _ in range(TGT_LEN):
            next_tok = model.decoder(tgt, enc)[:, -1].argmax(dim=-1, keepdim=True)
            tgt = torch.cat([tgt, next_tok], dim=1)

    for i in range(10):
        src_str = "".join(i2w[t] for t in sample[i].tolist() if t != PAD)
        pred = []
        for t in tgt[i, 1:].tolist():
            if t == EOS:
                break
            pred.append(i2w[t])
        print(f"  {src_str:>6} -> {''.join(pred)}")
