"""Train the from-scratch GPT on Tiny Shakespeare, character by character, on whatever accelerator
is available (Apple GPU via MPS, CUDA, or CPU).

The configuration is the classic small "shakespeare-char" setup (6 layers, 6 heads, 384 wide,
256 characters of context), so the result is comparable with other small GPT reproductions.

    .venv/bin/python train.py                 # 2,000 steps, writes runs/shakespeare_char.json
    .venv/bin/python train.py --steps 200     # a quick look
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import time
import urllib.request

import torch

from gpt2 import GPT, Config

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "data" / "tinyshakespeare.txt"
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    if not DATA.exists():
        DATA.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(URL, DATA)
    text = DATA.read_text()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    train, val = data[:n], data[n:]

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    cfg = Config(vocab_size=len(chars), block_size=256, n_layer=6, n_head=6, n_embd=384, dropout=0.2)
    model = GPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{n_params / 1e6:.2f}M parameters on {device}")

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}], lr=1e-3, betas=(0.9, 0.99))
    batch, max_lr, min_lr, warmup = 64, 1e-3, 1e-4, 100
    # bfloat16 mixed precision: matmuls in bf16, weights and optimizer state stay in fp32 (about 2x on an M5)
    autocast = lambda: torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device != "cpu")

    def lr_at(step):
        if step < warmup:
            return max_lr * (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, args.steps - warmup))
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

    def get_batch(src):
        ix = torch.randint(len(src) - cfg.block_size - 1, (batch,), generator=gen)
        x = torch.stack([src[i : i + cfg.block_size] for i in ix])
        y = torch.stack([src[i + 1 : i + 1 + cfg.block_size] for i in ix])
        return x.to(device), y.to(device)

    @torch.no_grad()
    def estimate(src, iters=20):
        model.eval()
        with autocast():
            losses = [model(*get_batch(src))[1].item() for _ in range(iters)]
        model.train()
        return sum(losses) / len(losses)

    log = {"config": vars(cfg), "params": n_params, "device": device, "precision": "bf16 autocast", "steps": [], "train": [], "val": []}
    t0 = time.time()
    for step in range(args.steps + 1):
        if step % 100 == 0 or step == args.steps:
            tr, va = estimate(train), estimate(val)
            log["steps"].append(step)
            log["train"].append(round(tr, 4))
            log["val"].append(round(va, 4))
            print(f"step {step:5d}  train {tr:.4f}  val {va:.4f}  lr {lr_at(step):.2e}  {time.time() - t0:5.0f}s", flush=True)
        if step == args.steps:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y = get_batch(train)
        with autocast():
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    start = torch.tensor([[stoi["\n"]]], device=device)
    out = model.generate(start, max_new_tokens=500, temperature=0.8, top_k=40, generator=torch.Generator(device=device).manual_seed(args.seed) if device != "mps" else None)
    sample = "".join(chars[i] for i in out[0, 1:].tolist())
    print("\n" + sample)
    log["sample"] = sample
    log["seconds"] = round(time.time() - t0)
    (ROOT / "runs").mkdir(exist_ok=True)
    (ROOT / "runs" / "shakespeare_char.json").write_text(json.dumps(log, indent=1))
    torch.save({"model": model.state_dict(), "config": vars(cfg), "chars": chars}, ROOT / "runs" / "shakespeare_char.pt")


if __name__ == "__main__":
    main()
