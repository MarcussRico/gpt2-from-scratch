"""GPT-2, written out by hand.

Everything the model does is in this file: token and position embeddings, pre-norm transformer
blocks with causal multi-head self-attention, a tanh-GELU MLP, a final LayerNorm, and an output
head tied to the token embedding. Attention is computed explicitly (scores, causal mask, softmax,
weighted sum), not through torch's fused kernel, so every step is visible. A key/value cache makes
generation linear in length instead of quadratic.

`GPT.from_pretrained("gpt2")` reads OpenAI's released weights from the safetensors file on the
Hugging Face hub and maps them onto these modules by hand.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class Config:
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0


SIZES = {
    "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M
    "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M
    "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M
    "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M
}


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_head, self.n_embd = cfg.n_head, cfg.n_embd
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)  # q, k, v in one matmul
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.c_proj.residual = True  # scaled at init: see GPT._init
        self.drop = nn.Dropout(cfg.dropout)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size), persistent=False)

    def forward(self, x: torch.Tensor, cache: dict | None = None) -> torch.Tensor:
        B, T, C = x.shape
        hd = C // self.n_head
        q, k, v = self.c_attn(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, hd).transpose(1, 2) for t in (q, k, v))  # (B, H, T, hd)

        past = 0
        if cache is not None:
            if "k" in cache:
                past = cache["k"].shape[2]
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v

        # scores: how much each position attends to each earlier one, scaled by 1/sqrt(head dim)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)  # (B, H, T, past+T)
        att = att.masked_fill(~self.mask[:, :, past : past + T, : past + T], float("-inf"))
        att = self.drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)  # re-join the heads
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.c_proj.residual = True

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))  # GPT-2 used the tanh form


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln_1, self.attn = nn.LayerNorm(cfg.n_embd), CausalSelfAttention(cfg)
        self.ln_2, self.mlp = nn.LayerNorm(cfg.n_embd), MLP(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cache=None):
        x = x + self.drop(self.attn(self.ln_1(x), cache))  # pre-norm: normalise, then add back
        return x + self.drop(self.mlp(self.ln_2(x)))


class GPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.h = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # weight tying: one matrix reads tokens in and out
        self.apply(self._init)

    def _init(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            std = 0.02
            if getattr(m, "residual", False):  # each block adds 2 residual branches: keep the sum's variance flat
                std *= (2 * self.cfg.n_layer) ** -0.5
            nn.init.normal_(m.weight, 0.0, std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, caches: list | None = None):
        B, T = idx.shape
        past = caches[0]["k"].shape[2] if caches and "k" in caches[0] else 0
        assert past + T <= self.cfg.block_size, f"context of {past + T} is longer than {self.cfg.block_size}"
        pos = torch.arange(past, past + T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for i, block in enumerate(self.h):
            x = block(x, caches[i] if caches is not None else None)
        logits = self.lm_head(self.ln_f(x))
        if targets is None:
            return logits
        return logits, F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None, greedy: bool = False, use_cache: bool = True, generator=None):
        caches = [{} for _ in self.h] if use_cache else None
        step_input = idx
        for _ in range(max_new_tokens):
            if caches is not None and caches[0] and caches[0]["k"].shape[2] + step_input.shape[1] > self.cfg.block_size:
                # the window is full: start a fresh cache from the last block_size - 1 tokens
                caches = [{} for _ in self.h]
                step_input = idx[:, -(self.cfg.block_size - 1) :]
            if caches is None:
                logits = self(idx[:, -self.cfg.block_size :])
            else:
                logits = self(step_input, caches=caches)
            logits = logits[:, -1, :]
            if greedy:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                nxt = torch.multinomial(F.softmax(logits, dim=-1), 1, generator=generator)
            idx = torch.cat([idx, nxt], dim=1)
            step_input = nxt
        return idx

    # ───────── OpenAI's weights ─────────

    @classmethod
    def from_pretrained(cls, name: str = "gpt2") -> GPT:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        model = cls(Config(**SIZES[name]))
        sd = load_file(hf_hub_download(f"openai-community/{name}", "model.safetensors"))
        sd = {k.removeprefix("transformer."): v for k, v in sd.items()}
        ours = model.state_dict()
        # OpenAI's checkpoint stores these four as Conv1D weights: (in, out) rather than Linear's (out, in)
        transposed = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")
        with torch.no_grad():
            for key, param in ours.items():
                if key == "lm_head.weight":
                    continue  # tied to wte
                src = sd[key]
                if key.endswith(transposed):
                    src = src.t()
                assert src.shape == param.shape, (key, src.shape, param.shape)
                param.copy_(src)
        return model
