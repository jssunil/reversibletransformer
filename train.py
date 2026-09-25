"""Train the 20M GPT on a fixed token budget and record loss / tokens/s / peak memory.

Example:
  python train.py --name run1_baseline --trunk residual --batch 32
  python train.py --name run2_rev_same --trunk midpoint_rev --batch 32
"""
import argparse
import csv
import json
import math
import os
import time

import numpy as np
import torch

from model import GPT, TRUNKS, GPTConfig
from reversible import reconstruction_error


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--trunk", choices=TRUNKS, default="residual")
    ap.add_argument("--h", type=float, default=None, help="step size (default 0.5 midpoint, 1.0 reveuler)")
    ap.add_argument("--stream", choices=["fp32", "fp64"], default="fp32", help="residual-stream dtype")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--tokens", type=float, default=50e6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--scale_lr", action="store_true", help="lr *= sqrt(batch/32), capped at 2x")
    ap.add_argument("--warmup", type=float, default=0.02)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--val_tokens", type=float, default=500_000)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    return ap.parse_args(argv)


def default_h(trunk):
    return 1.0 if trunk.startswith("reveuler") else 0.5


class Loader:
    def __init__(self, path, batch, seq, seed):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.batch, self.seq = batch, seq
        self.rng = np.random.default_rng(seed)

    def next(self):
        ix = self.rng.integers(0, len(self.data) - self.seq - 1, self.batch)
        x = np.stack([self.data[i:i + self.seq + 1] for i in ix]).astype(np.int64)
        x = torch.from_numpy(x).pin_memory().cuda(non_blocking=True)
        return x[:, :-1], x[:, 1:]


@torch.no_grad()
def evaluate(model, path, seq, n_tokens, batch, amp):
    data = np.memmap(path, dtype=np.uint16, mode="r")
    n_win = min(int(n_tokens) // seq, (len(data) - 1) // seq)
    model.eval()
    tot = 0.0
    for s in range(0, n_win, batch):
        ix = range(s, min(s + batch, n_win))
        x = np.stack([data[i * seq:i * seq + seq + 1] for i in ix]).astype(np.int64)
        x = torch.from_numpy(x).cuda()
        with amp:
            tot += model(x[:, :-1], x[:, 1:]).item() * len(ix)
    model.train()
    return tot / n_win


def lr_at(step, total, peak, warmup):
    w = max(1, int(warmup * total))
    if step < w:
        return peak * (step + 1) / w
    p = (step - w) / max(1, total - w)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


def train(args):
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    h = args.h if args.h is not None else default_h(args.trunk)
    cfg = GPTConfig(seq_len=args.seq, trunk=args.trunk, h=h, stream_dtype=args.stream)
    model = GPT(cfg).cuda()
    n_params = model.num_params()

    lr = args.lr * (min(2.0, math.sqrt(args.batch / 32)) if args.scale_lr else 1.0)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=lr, betas=(0.9, 0.95), fused=True)
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    amp = torch.autocast("cuda", dtype=dt)
    scaler = torch.amp.GradScaler("cuda", enabled=args.dtype == "fp16")

    tok_per_step = args.batch * args.seq
    steps = math.ceil(args.tokens / tok_per_step)
    loader = Loader(os.path.join(args.data, "train.bin"), args.batch, args.seq, args.seed)
    os.makedirs(args.out, exist_ok=True)
    log_f = open(os.path.join(args.out, f"{args.name}.csv"), "w", newline="")
    log = csv.writer(log_f)
    log.writerow(["step", "tokens", "loss", "lr", "tok_per_s", "elapsed_s"])
    print(f"[{args.name}] trunk={args.trunk} h={h} stream={args.stream} params={n_params / 1e6:.2f}M "
          f"batch={args.batch}x{args.seq} steps={steps} lr={lr:.2e}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    ema, diverged, warm_t, warm_step = None, False, None, 20
    torch.cuda.synchronize()
    t0 = t_log = time.perf_counter()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, lr, args.warmup)
        x, y = loader.next()
        with amp:
            loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

        if step == warm_step - 1:
            torch.cuda.synchronize()
            warm_t = time.perf_counter()
        if (step + 1) % args.log_every == 0 or step == steps - 1:
            li = loss.item()
            if not math.isfinite(li):
                diverged = True
                print(f"  step {step + 1}: loss is {li}, stopping")
                break
            ema = li if ema is None else 0.9 * ema + 0.1 * li
            now = time.perf_counter()
            tps = args.log_every * tok_per_step / (now - t_log)
            t_log = now
            log.writerow([step + 1, (step + 1) * tok_per_step, f"{li:.4f}", f"{opt.param_groups[0]['lr']:.3e}",
                          f"{tps:.0f}", f"{now - t0:.1f}"])
            if (step + 1) % (args.log_every * 8) == 0 or step == steps - 1:
                log_f.flush()
                print(f"  step {step + 1}/{steps} loss {li:.4f} ema {ema:.4f} tok/s {tps:,.0f} "
                      f"mem {torch.cuda.max_memory_allocated() / 2**30:.2f}GiB")
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    log_f.close()
    steps_done = step + 1
    peak_alloc = torch.cuda.max_memory_allocated() / 2**30
    peak_res = torch.cuda.max_memory_reserved() / 2**30

    val = float("nan") if diverged else evaluate(model, os.path.join(args.data, "val.bin"), args.seq,
                                                 args.val_tokens, 32, amp)
    rec = None
    if args.trunk.endswith("_rev"):
        with amp:
            rec = reconstruction_error(model, x[:4])
    res = dict(
        name=args.name, trunk=args.trunk, h=h, stream=args.stream, batch=args.batch, seq=args.seq, params=n_params,
        lr=lr, steps=steps_done, tokens=steps_done * tok_per_step,
        final_train_loss=ema, val_loss=val, diverged=diverged,
        tokens_per_s=steps_done * tok_per_step / (t1 - t0),
        tokens_per_s_steady=((steps_done - warm_step) * tok_per_step / (t1 - warm_t)) if warm_t else None,
        wall_s=t1 - t0, peak_mem_gib=peak_alloc, peak_reserved_gib=peak_res, recon_error=rec,
        gpu=torch.cuda.get_device_name(), dtype=args.dtype,
    )
    with open(os.path.join(args.out, f"{args.name}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    train(get_args())
