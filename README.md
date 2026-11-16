# gpt2-from-scratch

GPT-2, written out by hand in PyTorch, and checked against OpenAI's released weights.

Everything the model does is in one file, [`gpt2/model.py`](gpt2/model.py): token and position
embeddings, pre-norm transformer blocks, causal multi-head self-attention computed explicitly
(scores, mask, softmax, weighted sum), the tanh-GELU MLP, a final LayerNorm, an output head tied to
the token embedding, GPT-2's scaled initialisation of residual projections, and a key/value cache
for generation. It doesn't import Hugging Face; `transformers` appears only in the tests, as the
referee.

## It computes what OpenAI's GPT-2 computes

`GPT.from_pretrained("gpt2")` reads OpenAI's 124M checkpoint (`model.safetensors` from the Hugging
Face hub) and maps it onto these modules by hand, transposing the four weights OpenAI stored as
`Conv1D`. Then:

```
.venv/bin/python -m pytest -q
4 passed
```

| Check | Result |
| --- | --- |
| Logits, against `transformers`' `GPT2LMHeadModel` on the same prompt | largest absolute difference **1.8 × 10⁻⁴** (logits are in the tens to hundreds: float32 rounding) |
| 20 greedy tokens from "The meaning of life is" | **identical** to the reference |
| Generation with the KV cache against without it | **identical** |
| Parameter count | **124,439,808**, as released (the head shares the embedding) |

```python
import tiktoken, torch
from gpt2 import GPT

enc = tiktoken.get_encoding("gpt2")
model = GPT.from_pretrained("gpt2").eval()
out = model.generate(torch.tensor([enc.encode("The meaning of life is")]), max_new_tokens=20, greedy=True)
print(enc.decode(out[0].tolist()))
# The meaning of life is not the same as the meaning of death.
#
# The meaning of life is not the same as
```

## Training it from scratch

```
.venv/bin/python train.py
```

The same model, configured small (6 layers, 6 heads, 384 wide, 256 characters of context,
dropout 0.2: 10.77M parameters), trained character by character on Tiny Shakespeare with AdamW,
warmup and cosine decay, gradient clipping and bfloat16 mixed precision, on an Apple M5 GPU (MPS) in
29 minutes:

| step | train loss | validation loss |
| ---: | ---: | ---: |
| 0 | 4.261 | 4.253 |
| 100 | 2.441 | 2.458 |
| 500 | 1.517 | 1.691 |
| 1,000 | 1.260 | 1.506 |
| 1,500 | 1.133 | 1.460 |
| 2,000 | 1.066 | 1.445 |

Validation loss is flattening near 1.44 while training loss keeps falling, so much longer training
would mostly memorise (the run was capped at 2,000 steps to keep it near half an hour). A sample at
temperature 0.8, top-k 40:

```
LUCIO:
Ay, any thing? Dead; and ripe here again.

DUKE VINCENTIO:
We are some returned and with me. Why do you met?
For I think there I do look words to me to the death?

ISABELLA:
O condemned comfort! then shall be for a fellower.

DUKE VINCENTIO:
By his cause hath been a husband and take the prisoner
of the murderous lawful the town of a place.

```

On the M5, the explicit attention here ran faster than PyTorch's fused `scaled_dot_product_attention`
(37 ms vs 64 ms per layer, forward and backward, at this size), so the readable version is also the
fast one. bfloat16 autocast halved the step time (1.46 s → 0.78 s). The full log is in
[`runs/shakespeare_char.json`](runs/shakespeare_char.json).

## Setup

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q     # first run downloads the 124M weights (~550 MB)
```

## Related

- [autograd-from-scratch](https://github.com/MarcussRico/autograd-from-scratch): the same kind of
  model trained by my own autograd engine, without PyTorch.
- [muon-newtonschulz](https://github.com/MarcussRico/muon-newtonschulz): the Muon optimiser's
  Newton-Schulz orthogonalisation.

MIT licence. The architecture follows OpenAI's GPT-2 paper and released code.
