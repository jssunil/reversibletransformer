"""Memory-free backward passes for reversible trunks.

The forward runs under no_grad and keeps only the final state(s). The backward
walks the layers in reverse, *reconstructing* each layer's input from its output,
recomputing that single layer with grad enabled, and back-propagating through it.
Peak activation memory is therefore one layer, independent of depth.
"""
import torch

_fwd = torch.amp.custom_fwd(device_type="cuda")
_bwd = torch.amp.custom_bwd(device_type="cuda")


def _grads(out, inputs, grad_out):
    gs = torch.autograd.grad(out, inputs, grad_out, allow_unused=True)
    return [torch.zeros_like(i) if g is None else g for g, i in zip(gs, inputs)]


class RevMidpointFn(torch.autograd.Function):
    """Leapfrog: x_{l+1} = x_{l-1} + 2h f_l(x_l);  inverse x_{l-1} = x_{l+1} - 2h f_l(x_l)."""

    @staticmethod
    @_fwd
    def forward(ctx, x0, x1, h, blocks, *params):
        with torch.no_grad():
            prev, cur = x0, x1
            for b in blocks:
                prev, cur = cur, prev + 2 * h * b.delta(cur)
        ctx.save_for_backward(prev, cur)
        ctx.h, ctx.blocks = h, blocks
        return cur

    @staticmethod
    @_bwd
    def backward(ctx, g_out):
        a, b = ctx.saved_tensors          # (x_l, x_{l+1}), starting at l = L-1
        ga, gb = torch.zeros_like(g_out), g_out
        h, blocks = ctx.h, ctx.blocks
        param_grads = []
        for blk in reversed(blocks):
            params = list(blk.parameters())
            with torch.enable_grad():
                a_ = a.detach().requires_grad_()
                y = blk.delta(a_)
            gs = _grads(y, [a_] + params, 2 * h * gb)
            x_prev = b - 2 * h * y.detach()          # reconstruct x_{l-1}
            # pair becomes (x_{l-1}, x_l); x_{l-1} feeds x_{l+1} with identity
            a, b = x_prev, a
            ga, gb = gb, ga + gs[0]
            param_grads.append(gs[1:])
        flat = [g for pg in reversed(param_grads) for g in pg]
        return (ga, gb, None, None, *flat)


class RevEulerFn(torch.autograd.Function):
    """Two-stream symplectic Euler (RevNet coupling):
    y' = y + h F(z);  z' = z + h G(y')   with exact inverse
    z = z' - h G(y');  y = y' - h F(z).
    """

    @staticmethod
    @_fwd
    def forward(ctx, y, z, h, blocks, *params):
        with torch.no_grad():
            for b in blocks:
                y = y + h * b.attn_delta(z)
                z = z + h * b.mlp_delta(y)
        ctx.save_for_backward(y, z)
        ctx.h, ctx.blocks = h, blocks
        return y, z

    @staticmethod
    @_bwd
    def backward(ctx, gy, gz):
        y, z = ctx.saved_tensors
        h, blocks = ctx.h, ctx.blocks
        gy = torch.zeros_like(y) if gy is None else gy
        gz = torch.zeros_like(z) if gz is None else gz
        param_grads = []
        for blk in reversed(blocks):
            params = list(blk.parameters())
            # undo z' = z + h G(y')
            with torch.enable_grad():
                y_ = y.detach().requires_grad_()
                m = blk.mlp_delta(y_)
            gm = _grads(m, [y_] + params, h * gz)
            z = z - h * m.detach()
            gy = gy + gm[0]
            # undo y' = y + h F(z)
            with torch.enable_grad():
                z_ = z.detach().requires_grad_()
                a = blk.attn_delta(z_)
            ga = _grads(a, [z_] + params, h * gy)
            y = y - h * a.detach()
            gz = gz + ga[0]
            param_grads.append([p1 + p2 for p1, p2 in zip(gm[1:], ga[1:])])
        flat = [g for pg in reversed(param_grads) for g in pg]
        return (gy, gz, None, None, *flat)


@torch.no_grad()
def reconstruction_error(model, idx):
    """Run the reversible trunk forward then invert it; return ||x0_hat - x0|| / ||x0||."""
    cfg, blocks = model.cfg, model.blocks
    T = idx.shape[1]
    x0 = model.tok_emb(idx) + model.pos_emb(torch.arange(T, device=idx.device))
    x0 = x0.to(torch.float64 if cfg.stream_dtype == "fp64" else torch.float32)
    h = cfg.h
    if cfg.trunk.startswith("midpoint"):
        prev, cur = x0, x0 + h * blocks[0].delta(x0)
        x1 = cur
        for b in blocks[1:]:
            prev, cur = cur, prev + 2 * h * b.delta(cur)
        for b in reversed(blocks[1:]):
            prev, cur = cur - 2 * h * b.delta(prev), prev
        return max(((prev - x0).norm() / x0.norm()).item(), ((cur - x1).norm() / x1.norm()).item())
    y = z = x0
    for b in blocks:
        y = y + h * b.attn_delta(z)
        z = z + h * b.mlp_delta(y)
    for b in reversed(blocks):
        z = z - h * b.mlp_delta(y)
        y = y - h * b.attn_delta(z)
    return max(((y - x0).norm() / x0.norm()).item(), ((z - x0).norm() / x0.norm()).item())
