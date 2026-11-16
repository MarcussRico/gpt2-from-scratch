"""The from-scratch model, given OpenAI's released GPT-2 (124M) weights, must compute what the
reference implementation computes: same logits, same greedy continuation.

Hugging Face `transformers` is used here only as the referee; gpt2/ never imports it.
The first run downloads the weights (~550 MB) into the Hugging Face cache.
"""

import pytest
import tiktoken
import torch

from gpt2 import GPT

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def models():
    torch.manual_seed(0)
    ours = GPT.from_pretrained("gpt2").eval()
    ref = transformers.GPT2LMHeadModel.from_pretrained("openai-community/gpt2").eval()
    return ours, ref


def test_logits_match_the_reference(models):
    ours, ref = models
    enc = tiktoken.get_encoding("gpt2")
    idx = torch.tensor([enc.encode("Hello, I'm a language model, and attention is all I need to")])
    with torch.no_grad():
        a = ours(idx)
        b = ref(idx).logits
    assert a.shape == b.shape == (1, idx.shape[1], 50257)
    assert torch.allclose(a, b, atol=1e-3, rtol=1e-4), (a - b).abs().max()


def test_greedy_continuation_is_identical(models):
    ours, ref = models
    enc = tiktoken.get_encoding("gpt2")
    idx = torch.tensor([enc.encode("The meaning of life is")])
    with torch.no_grad():
        a = ours.generate(idx, max_new_tokens=20, greedy=True)
        b = ref.generate(idx, max_new_tokens=20, do_sample=False, pad_token_id=enc.eot_token)
    assert a.tolist() == b.tolist()


def test_the_kv_cache_changes_nothing(models):
    ours, _ = models
    enc = tiktoken.get_encoding("gpt2")
    idx = torch.tensor([enc.encode("In a shocking finding, scientists discovered")])
    with torch.no_grad():
        cached = ours.generate(idx, max_new_tokens=24, greedy=True, use_cache=True)
        uncached = ours.generate(idx, max_new_tokens=24, greedy=True, use_cache=False)
    assert cached.tolist() == uncached.tolist()


def test_parameter_count_is_124m(models):
    ours, _ = models
    n = sum(p.numel() for p in ours.parameters())  # the output head shares the token embedding
    assert n == 124_439_808
