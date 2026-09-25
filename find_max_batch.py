"""Find the largest batch (multiple of --granularity) that trains without OOM.

Each probe builds the model + fused AdamW and runs a few full training steps,
so the peak includes weights, grads, optimizer state and activations.
Example:  python find_max_batch.py --trunk midpoint_rev --stream fp64
"""
import argparse
import gc
import json
import os

import torch

from model import GPT, TRUNKS, GPTConfig
from train import default_h


def probe(trunk, batch, seq, h, stream, dtype=torch.bfloat16, steps=3):
    torch.manual_seed(0)
    model = opt = x = loss = None
    try:
        model = GPT(GPTConfig(seq_len=seq, trunk=trunk, h=h, stream_dtype=stream)).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
        torch.cuda.reset_peak_memory_stats()
        for _ in range(steps):
            x = torch.randint(0, model.cfg.vocab_size, (batch, seq + 1), device="cuda")
            with torch.autocast("cuda", dtype=dtype):
                loss = model(x[:, :-1], x[:, 1:])
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_reserved()
    except torch.cuda.OutOfMemoryError:
        return None
    finally:
        del model, opt, x, loss
        gc.collect()
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trunk", choices=TRUNKS, required=True)
    ap.add_argument("--h", type=float, default=None)
    ap.add_argument("--stream", choices=["fp32", "fp64"], default="fp32")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--start", type=int, default=32)
    ap.add_argument("--granularity", type=int, default=8)
    ap.add_argument("--headroom", type=float, default=0.80,
                    help="fraction of free memory allowed; near-full VRAM on Windows spills to host RAM and runs slow")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16", help="fp16 on GPUs without bf16 (T4)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    h = args.h if args.h is not None else default_h(args.trunk)

    free, _ = torch.cuda.mem_get_info()
    budget = args.headroom * free
    fits = lambda b: (lambda r: r is not None and r <= budget)(probe(args.trunk, b, args.seq, h, args.stream, dt))

    lo, hi = 0, args.start
    while fits(hi):
        print(f"  batch {hi}: fits")
        lo, hi = hi, hi * 2
    print(f"  batch {hi}: does not fit")
    g = args.granularity
    while hi - lo > g:
        mid = (lo + hi) // 2 // g * g
        if mid <= lo:
            break
        ok = fits(mid)
        print(f"  batch {mid}: {'fits' if ok else 'does not fit'}")
        lo, hi = (mid, hi) if ok else (lo, mid)
    print(f"max batch for {args.trunk} (stream={args.stream}): {lo}  (budget {budget / 2**30:.1f} GiB)")
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"maxbatch_{args.trunk}_{args.stream}.json"), "w") as f:
        json.dump(dict(trunk=args.trunk, stream=args.stream, h=h, seq=args.seq, max_batch=lo,
                       budget_gib=budget / 2**30), f, indent=2)


if __name__ == "__main__":
    main()
