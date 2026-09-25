"""Small GPT with interchangeable trunks.

trunk options
  residual       standard pre-LN transformer (baseline)
  midpoint       reversible-midpoint (leapfrog) architecture, ordinary autograd (stores activations)
  midpoint_rev   same network, memory-free backward (reconstructs states in reverse)
  reveuler       two-stream reversible (symplectic) Euler / RevNet coupling, ordinary autograd
  reveuler_rev   same network, memory-free backward
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from reversible import RevEulerFn, RevMidpointFn

TRUNKS = ("residual", "midpoint", "midpoint_rev", "reveuler", "reveuler_rev")


@dataclass
class GPTConfig:
    vocab_size: int = 8192
    seq_len: int = 512
    n_layer: int = 10
    n_head: int = 6
    d_model: int = 384
    trunk: str = "residual"
    h: float = 0.5  # integrator step size (midpoint uses 2h per layer)
    loss_chunk: int = 8192  # tokens per checkpointed loss chunk (0 = plain CE)
    stream_dtype: str = "fp32"  # residual-stream precision; fp64 makes reverse reconstruction ~exact


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, C // self.n_head).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=False)
        self.proj = nn.Linear(4 * cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))


def _compute_in(x, like):
    """Blocks compute in the parameter dtype even when the residual stream is fp64."""
    return x.to(like.dtype)


class Block(nn.Module):
    """A transformer block exposed as residual *deltas* so integrators can compose them."""

    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def attn_delta(self, x):
        return self.attn(self.ln1(_compute_in(x, self.ln1.weight))).to(x.dtype)

    def mlp_delta(self, x):
        return self.mlp(self.ln2(_compute_in(x, self.ln2.weight))).to(x.dtype)

    def delta(self, x):
        """f(x) = Block(x) - x for a standard pre-LN block."""
        a = self.attn_delta(x)
        return a + self.mlp_delta(x + a)


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.trunk in TRUNKS, cfg.trunk
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # tied
        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def block_params(self):
        return [p for b in self.blocks for p in b.parameters()]

    # ---- trunks -------------------------------------------------------
    def trunk(self, x):
        t, h, blocks = self.cfg.trunk, self.cfg.h, self.blocks
        if t == "residual":
            for b in blocks:
                x = x + b.delta(x)
            return x
        if t.startswith("midpoint"):
            # Euler starter step, then leapfrog: x_{l+1} = x_{l-1} + 2h f_l(x_l)
            x1 = x + h * blocks[0].delta(x)
            if t == "midpoint_rev":
                params = [p for b in blocks[1:] for p in b.parameters()]
                return RevMidpointFn.apply(x, x1, h, blocks[1:], *params)
            prev, cur = x, x1
            for b in blocks[1:]:
                prev, cur = cur, prev + 2 * h * b.delta(cur)
            return cur
        # reveuler: y' = y + h F(z);  z' = z + h G(y')
        if t == "reveuler_rev":
            y, z = RevEulerFn.apply(x, x, h, blocks, *self.block_params())
        else:
            y = z = x
            for b in blocks:
                y = y + h * b.attn_delta(z)
                z = z + h * b.mlp_delta(y)
        return 0.5 * (y + z)

    def forward(self, idx, targets=None):
        T = idx.shape[1]
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        sd = torch.float64 if self.cfg.stream_dtype == "fp64" else torch.float32
        x = x.to(torch.promote_types(x.dtype, sd))
        x = _compute_in(self.trunk(x), self.ln_f.weight)
        if targets is None:
            return self.head(self.ln_f(x))
        x, targets = x.reshape(-1, x.size(-1)), targets.reshape(-1)
        c = self.cfg.loss_chunk
        if c <= 0:
            return self._ce_sum(x, targets) / targets.numel()
        # checkpointed chunks: only one chunk's logits ever exist (same for every trunk)
        total = sum(checkpoint(self._ce_sum, x[i:i + c], targets[i:i + c], use_reentrant=False)
                    for i in range(0, targets.numel(), c))
        return total / targets.numel()

    def _ce_sum(self, x, targets):
        return F.cross_entropy(self.head(self.ln_f(x)).float(), targets, reduction="sum")
