# ERA V5 Session 13 — Reversible LLM training: report

**Task (study guide §10):** train a ~20M-parameter LLM for 50M tokens three ways — (1) baseline at a fixed batch,
(2) reversible at the same batch, (3) reversible at the largest batch the freed memory allows — and report
final loss, tokens/s, peak memory and findings.

## TL;DR

| run | integrator | batch | steps | final train loss | **val loss** | **tokens/s** | **peak mem** |
|---|---|---|---|---|---|---|---|
| **Run 1** baseline (standard residual) | — | 32 | 3,052 | 1.761 | **1.849** | **243,278** | **3.22 GiB** |
| **Run 2** reversible, same batch | rev. Euler, h=0.5 | 32 | 3,052 | 1.804 | **1.890** | **156,965** | **1.16 GiB** |
| **Run 3** reversible, max batch | rev. Euler, h=0.5 | 432 | 227 | 4.760 | **3.970** | **158,349** | **10.51 GiB** |
| Run 1b (extra) baseline, max batch | — | 152 | 643 | 3.768 | 3.442 | 247,824 | 11.35 GiB |

All runs: 50.0M tokens, seq 512, bf16 autocast, AdamW, one RTX 5070 Ti (16 GB).
Tokens/s is steady-state (first 20 steps excluded). Peak memory is `torch.cuda.max_memory_allocated()`.

> **Interpretation.** Reversibility cut peak memory by **64%** (3.22 → 1.16 GiB) and cost **35%** throughput at equal
> batch, with nearly the same loss (val 1.890 vs 1.849). It let the batch grow **2.8×** (432 vs 152 for the baseline under the
> same memory budget). But the larger batch in Run 3 **did not recover throughput** (158K tok/s = 65% of baseline).
> At this scale the GPU is **compute-bound** already at batch 32, so the recompute is a pure tax. The trade becomes
> favourable only when memory, not compute, is the bottleneck (study guide §9.1).

![loss curves](results/plots/main_loss.png)
![throughput and memory](results/plots/main_bars.png)

---

## 1. Setup

| | |
|---|---|
| Data | TinyStories. One train parquet shard, 55.1M tokens written, 50.0M consumed per run. Val: 1.2M tokens, 0.5M used for eval |
| Tokenizer | 8,192-vocab byte-level BPE trained on 200K stories |
| Model | GPT, pre-LN, d=384, 10 layers, 6 heads, MLP 4×, context 512, learned positions, tied embeddings, no dropout. **21.05M params** |
| Optimizer | AdamW (0.9, 0.95), wd 0.1, peak lr 1e-3 (max-batch runs: lr·√(B/32), capped at 2×), 2% warmup, cosine to 10%, clip 1.0 |
| Precision | bf16 autocast; residual stream fp32 (baseline) or **fp64** (reversible, see §4.1) |
| Loss | Checkpointed, chunked cross-entropy (8,192 tokens/chunk), the **same for all runs** (see §4.3) |
| Hardware | RTX 5070 Ti 16 GB, PyTorch 2.7.0+cu128, Windows 11 |

### Reversible implementation
`reversible.py` has custom `autograd.Function`s. The forward runs under `no_grad` and keeps **only the final state(s)**.
The backward walks the layers in reverse. For each layer it (a) reconstructs the layer's input from its output,
(b) recomputes that one layer with grad enabled, and (c) back-propagates through it. Activation memory is therefore
one layer, independent of depth.

`tests/test_reversible.py` checks three things, and all 6 tests pass:
- The loss and every parameter gradient equal ordinary autograd on the same network, in fp64.
- Reconstruction recovers the input.
- Activation memory stays roughly flat from 4 to 16 layers, where the stored version grows linearly.

## 2. Which integrator worked (screening: 5M tokens, batch 32)

| variant | update rule | h | val loss | tokens/s | peak mem |
|---|---|---|---|---|---|
| baseline residual | x ← x + f(x) | — | 3.624 | 244,633 | 3.22 GiB |
| midpoint / leapfrog | x₊ = x₋ + 2h·f(x) | 0.5 | 4.379 | 159,520 | 1.38 GiB |
| midpoint / leapfrog | 〃 | 0.25 | 4.132 | 159,536 | 1.38 GiB |
| midpoint, *stored activations* (control) | 〃 | 0.5 | 4.210 | 243,082 | 3.22 GiB |
| **reversible (symplectic) Euler** | y ← y + h·Attn(z); z ← z + h·MLP(y) | **0.5** | **3.601** | 155,296 | **1.16 GiB** |
| reversible (symplectic) Euler | 〃 | 1.0 | 3.912 | 156,191 | 1.16 GiB |

![screening](results/plots/screen_loss.png)

- **Reversible (symplectic) Euler with h = 0.5 trained stably and reached the best loss.** It matched and slightly beat
  the baseline (3.601 vs 3.624), so it was used for Runs 2 and 3.
- **Midpoint/leapfrog trained stably (no divergence) but learned more slowly.** A smaller step helped (h=0.25 beat
  h=0.5). The stored-activation control reaches the same loss, so the gap comes from the **leapfrog architecture**
  (its two interleaved odd/even streams), not from the reversible backward.
- Neither variant diverged.

## 3. Findings

1. **Memory: −64% at equal batch.** The remaining 1.16 GiB is weights + grads + AdamW state (21M × 16 B ≈ 0.31 GiB), embeddings,
   one loss chunk and one layer's activations. In the baseline, stored activations were ~2 GiB of its 3.22 GiB.
2. **Throughput: −35% at equal batch.** The backward does one extra forward per layer (theory ≈ +33% compute). About
   20 points of the 35% come from the fp64 residual stream. With an fp32 stream the reversible model ran at 195K tok/s
   (−20%), but its gradients are approximate (§4.1).
3. **Loss: essentially unchanged at equal batch** (val 1.890 vs 1.849, +2%). The two curves overlap for the whole run.
4. **The bigger batch did not buy speed.** Reversible throughput is flat in batch size:
   157K (B=32) → 160K (B=128) → 160K (B=256) → 158K (B=432). A 21M-param model already saturates the GPU at
   B=32, so extra memory has nothing to convert into throughput. **This is the compute-bound case, where reversibility
   is a tax.** The 2.8× larger batch would pay off only if the baseline could not reach an efficient batch at all, e.g.
   a bigger model, longer context, or a smaller GPU.
5. **A big batch hurts loss under a fixed token budget.** Run 3 took only 227 optimizer steps (val 3.970), against
   3,052 for Run 1 (val 1.849). The baseline at its own max batch shows the same effect (Run 1b: 643 steps, val 3.442).
   Batch 152–432 is far above the useful ("critical") batch size for this model and budget. Freed memory is worth more
   as longer context or a bigger model than as more tokens per step.

## 4. Implementation lessons (things that went wrong first)

### 4.1 Residual-stream precision decides whether reversibility is exact
Reconstruction `x₋ = x₊ − 2h·f(x)` is exact only in exact arithmetic. With an **fp32** residual stream, rounding
errors are amplified as the backward walks down the layers:

| | fp32 stream | fp64 stream |
|---|---|---|
| midpoint: relative reconstruction error of x₀ | 0.17–0.19 | **0** |
| rev. Euler: relative reconstruction error of x₀ | 0.1–4 | **0** |
| rev. Euler (h=1.0): gradient relative error vs. stored autograd | **5.9%** | 0.08% (bf16 noise) |
| rev. Euler throughput (B=32) | 195K tok/s | 157K tok/s |

Only 2 state tensors per sequence are kept, so an fp64 stream costs little memory. It still costs ~20% speed on a
GeForce card, which has weak fp64. The blocks themselves still compute in bf16. The main runs use fp64 so the
gradients are exact.

### 4.2 Call `backward()` outside `autocast`
Calling `loss.backward()` inside the autocast context made the custom backward's recomputation disagree with the
forward: gradient cosine similarity was 0.92 against ordinary autograd. Outside autocast, as `train.py` does, it is 1.0000.

### 4.3 The LM head hid the savings until the loss was chunked
Even with an 8K vocab, the fp32 logits (and their CE copies) grew with batch and dominated peak memory.
Before chunking, the max batch was 104 (baseline) vs 176 (reversible), 1.7×. After switching to a checkpointed,
chunked cross-entropy for **all** runs, it was 152 vs 432, 2.8×. With GPT-2's 50K vocab this effect would be far worse.

### 4.4 "Max batch" on Windows: leave VRAM headroom
The first max-batch attempt used a 92%-of-free-memory budget: batch 496 for reversible, 12.0 GiB allocated, 15.5 of 16 GB
of device memory in use. It ran at only **66–127K tok/s**, against 157K when measured in isolation. Windows (WDDM) spills
GPU memory to host RAM instead of raising OOM when the card is nearly full, and other desktop apps' VRAM use varied.
GPU clocks and temperature were normal (~2.8 GHz, ≤75 °C, no throttle flags). The budget was lowered to 80% of free
memory (11.7 GiB) and Runs 3/1b were re-run: batch 432 and 152, with normal speed.

| discarded attempt (near-full VRAM) | batch | tokens/s | val loss |
|---|---|---|---|
| Run 3, try 1 | 496 | 89,715 | 4.047 |
| Run 3, try 2 | 496 | 113,348 | 4.030 |
| Run 1b | 176 | 251,102 | 3.454 |
| Run 3b, fp32 stream | 744 | 187,346 | 4.680 (recon. error 45 → gradients unreliable) |

<!-- COLAB_NOTE:START -->
The planned re-run of Run 3b (fp32 stream at its safe max batch of 640) was **stopped by request** after a machine
warning and has no final result. The max-batch probe alone shows the fp32 stream fits **640** vs 432 for fp64.
<!-- COLAB_NOTE:END -->

## 5. Conclusion
The reversible (symplectic Euler) network was a drop-in replacement. Its loss was within 2% of the baseline, its
activation memory was independent of depth, and peak memory fell by 64%. The price was a 35% throughput loss (≈20%
with an fp32 stream, at the cost of exact gradients). For a 20M model on a 16 GB GPU this is not worth paying: the
baseline is compute-bound and fits comfortably, and the extra batch that reversibility unlocks neither speeds up
training nor helps loss under a 50M-token budget. The technique earns its keep when activations, not compute, are the
binding constraint: long context, deep models, or GPUs too small to hold an efficient batch.

<!-- COLAB_SECTION:START -->
## 6. Run 3b on Google Colab
*Pending: run `colab_run3b.ipynb` in Colab, then locally `python colab_summary.py --update REPORT.md`.*
<!-- COLAB_SECTION:END -->

## Reproduce
```bash
python prepare_data.py
python -m pytest tests -q
bash run_all.sh screen
bash run_all.sh main
python make_report.py        # regenerates REPORT_results.md and results/plots/ (CPU only)
```
Raw data: `results/*.json|csv` (main runs), `results/screen/` (screening), `results/tp/` (throughput sweep and discarded attempts).
