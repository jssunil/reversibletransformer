import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import GPT, GPTConfig  # noqa: E402
from reversible import reconstruction_error  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def tiny(trunk, n_layer=4, h=0.5):
    return GPTConfig(vocab_size=97, seq_len=16, n_layer=n_layer, n_head=2, d_model=32, trunk=trunk, h=h)


def grads(trunk, stored_sd, idx, tgt, h):
    torch.manual_seed(0)
    m = GPT(tiny(trunk, h=h)).to(DEV).double()
    m.load_state_dict(stored_sd)
    loss = m(idx, tgt)
    loss.backward()
    return loss.item(), {n: p.grad.clone() for n, p in m.named_parameters()}


@pytest.mark.parametrize("base,h", [("midpoint", 0.5), ("midpoint", 0.25), ("reveuler", 1.0)])
def test_grad_equivalence(base, h):
    torch.manual_seed(0)
    ref = GPT(tiny(base, h=h)).to(DEV).double()
    sd = ref.state_dict()
    idx = torch.randint(0, 97, (3, 16), device=DEV)
    tgt = torch.randint(0, 97, (3, 16), device=DEV)
    l1, g1 = grads(base, sd, idx, tgt, h)
    l2, g2 = grads(base + "_rev", sd, idx, tgt, h)
    assert abs(l1 - l2) < 1e-10
    for n in g1:
        assert torch.allclose(g1[n], g2[n], rtol=1e-6, atol=1e-9), n


@pytest.mark.parametrize("trunk", ["midpoint_rev", "reveuler_rev"])
def test_reconstruction(trunk):
    torch.manual_seed(0)
    m = GPT(tiny(trunk, n_layer=8)).to(DEV)
    idx = torch.randint(0, 97, (2, 16), device=DEV)
    assert reconstruction_error(m, idx) < 1e-5


@pytest.mark.skipif(DEV != "cuda", reason="needs CUDA memory stats")
def test_memory_flat_in_depth():
    def peak(trunk, L):
        torch.manual_seed(0)
        cfg = GPTConfig(vocab_size=256, seq_len=256, n_layer=L, n_head=4, d_model=256, trunk=trunk)
        m = GPT(cfg).to(DEV)
        idx = torch.randint(0, 256, (16, 256), device=DEV)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        m(idx, idx).backward()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() - base

    def act(trunk, L):  # activation part: subtract param+grad bytes
        n = sum(p.numel() for p in GPT(GPTConfig(vocab_size=256, seq_len=256, n_layer=L, n_head=4,
                                                 d_model=256, trunk=trunk)).parameters())
        return peak(trunk, L) - 4 * n

    stored = act("midpoint", 16) - act("midpoint", 4)
    rev = act("midpoint_rev", 16) - act("midpoint_rev", 4)
    assert rev < 0.2 * stored, (rev, stored)
