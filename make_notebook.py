import json
import os

OUT = "Reversible_Assignment.ipynb"


def src(t):
    lines = t.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": src(t.strip("\n"))}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src(t.strip("\n"))}


def wf(path, title):
    body = open(path, encoding="utf-8").read().rstrip("\n")
    return [md(title), code(f"%%writefile {path}\n{body}")]


report = open("REPORT.md", encoding="utf-8").read()
tldr = report[report.index("| run | integrator"):report.index("All runs: 50.0M tokens")].strip()
if "† Run 3b" in report:
    tldr += "\n\n" + report[report.index("† Run 3b"):].split("\n", 1)[0]

cells = [
    md("""
# ERA V5 — Session 13 Assignment: Reversible LLM training

**Task (study guide §10):** train a ~20M-parameter LLM for 50M tokens, three runs:
1. **Baseline** — standard residual transformer at a fixed batch (32 × 512).
2. **Reversible, same batch** — and report which integrator variant worked (midpoint/leapfrog vs reversible Euler).
3. **Reversible, max batch** — push the batch as large as the freed memory allows.

Report: final loss, speed (tokens/s), peak memory, findings.

| Part | What |
|---|---|
| 0 | Setup (GPU, precision, config) |
| 1 | Data: TinyStories → 8K BPE → `data/train.bin`, `data/val.bin` |
| 2 | Code: `reversible.py`, `model.py`, `train.py`, `find_max_batch.py`, `make_report.py`, written with `%%writefile`. These are the exact files that were tested |
| 3 | Tests: reversible gradients == autograd, reconstruction, memory flat in depth |
| 4 | Integrator screening (short runs) |
| 5 | The three assignment runs (+ optional extras) |
| 6 | Report: tables + plots |
| 7 | Results obtained on RTX 5070 Ti + findings |

Run top to bottom on **Google Colab (Runtime → Change runtime type → GPU)** or on a local CUDA machine.
On a T4 the full notebook takes roughly 1.5–2.5 h; turn off the optional parts in the config cell to save time.
"""),
    md("## 0. Setup"),
    code("""
import os, sys, json, glob, torch
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    !pip -q install tokenizers datasets
    os.makedirs("/content/era13", exist_ok=True)
    %cd /content/era13
os.makedirs("tests", exist_ok=True)
assert torch.cuda.is_available(), "Needs a CUDA GPU (Colab: Runtime > Change runtime type > GPU)"
DT = "bf16" if torch.cuda.get_device_capability()[0] >= 8 else "fp16"      # native bf16 needs Ampere+; T4 would emulate it (very slow) -> fp16 + GradScaler
print(torch.__version__, "|", torch.cuda.get_device_name(), "| autocast dtype:", DT)

# ---- experiment config ----
B0 = 32                 # baseline batch (x 512 tokens)
TOKENS = 50e6           # token budget per run
RUN_SCREENING = True    # 5M-token integrator screening (7 short runs)
RUN_EXTRAS = True       # Run 1b (baseline at max batch) and Run 3b (fp32 stream at max batch)
HEADROOM = 0.90 if IN_COLAB else 0.80   # fraction of free VRAM for the max-batch probe (Windows spills to host RAM near full)
"""),
    md("""
## 1. Data
TinyStories, tokenised with an **8,192-vocab byte-level BPE**. A small vocab keeps the embedding/logits from dominating
parameters and peak memory, so the activation savings being measured stay visible. If `data/raw/*.parquet` exists
(curl-downloaded), it is used; otherwise the script streams from the Hugging Face Hub.
"""),
    *wf("prepare_data.py", "### `prepare_data.py`"),
    code("""
if not os.path.exists("data/train.bin"):
    !python prepare_data.py
!ls -la data
"""),
    md("""
## 2. Code
### `reversible.py`: memory-free backward
The forward runs under `no_grad` and keeps only the final state(s). The backward walks the layers in reverse,
**reconstructs** each layer's input from its output, recomputes that single layer with grad enabled, and back-propagates
through it. Activation memory is therefore one layer, independent of depth.

* **Midpoint / leapfrog:** `x_{l+1} = x_{l-1} + 2h f_l(x_l)`; inverse `x_{l-1} = x_{l+1} - 2h f_l(x_l)`
* **Reversible (symplectic) Euler / RevNet coupling:** `y' = y + h·Attn(z)`, `z' = z + h·MLP(y')`; inverse `z = z' - h·MLP(y')`, `y = y' - h·Attn(z)`
"""),
    code("%%writefile reversible.py\n" + open("reversible.py", encoding="utf-8").read().rstrip("\n")),
    *wf("model.py", """### `model.py`: GPT with interchangeable trunks
d=384, 10 layers, 6 heads, context 512, tied embeddings, no dropout (the backward must recompute the same function) → **21.05M params**.
`residual` is the baseline; `midpoint` / `reveuler` are the same architecture with stored activations; `*_rev` use the memory-free backward.
`stream_dtype="fp64"` makes reconstruction exact. The loss is a checkpointed, chunked cross-entropy, the same for all runs."""),
    *wf("train.py", "### `train.py`: token-budgeted training; records val loss, tokens/s, peak memory"),
    *wf("find_max_batch.py", "### `find_max_batch.py`: largest batch that trains (doubling + binary search with real optimizer steps)"),
    *wf("make_report.py", "### `make_report.py`: tables + plots"),
    md("## 3. Tests\nThe reversible backward must give the **same gradients** as ordinary autograd before any result means anything."),
    *wf("tests/test_reversible.py", "### `tests/test_reversible.py`"),
    code("!python -m pytest tests -v"),
    md("""
## 4. Integrator screening (5M tokens, batch 32)
Pick the integrator (and step size h) that trains stably and reaches the best loss. The stored-activation `midpoint` run is a
control: it separates architecture effects from backward effects.
"""),
    code("""
S = f"--tokens 5e6 --batch {B0} --dtype {DT} --out results/screen"
if RUN_SCREENING:
    !python train.py --name residual             --trunk residual $S
    !python train.py --name midpoint_h0.5        --trunk midpoint_rev --h 0.5  --stream fp64 $S
    !python train.py --name midpoint_h0.25       --trunk midpoint_rev --h 0.25 --stream fp64 $S
    !python train.py --name midpoint_h0.5_stored --trunk midpoint     --h 0.5 $S
    !python train.py --name reveuler_h1.0        --trunk reveuler_rev --h 1.0  --stream fp64 $S
    !python train.py --name reveuler_h0.5        --trunk reveuler_rev --h 0.5  --stream fp64 $S
    !python train.py --name reveuler_h0.5_fp32stream --trunk reveuler_rev --h 0.5 --stream fp32 $S
"""),
    code("""
# Choose the winner: best val loss among exact (fp64-stream) reversible variants.
# Default = the local screening result (reversible Euler, h = 0.5).
REV, H = "reveuler_rev", 0.5
cands = [json.load(open(p)) for p in glob.glob("results/screen/*.json")]
cands = [c for c in cands if c["trunk"].endswith("_rev") and c.get("stream") == "fp64" and not c["diverged"]]
for c in sorted(cands, key=lambda c: c["val_loss"]):
    print(f"{c['name']:22s} val {c['val_loss']:.3f}  tok/s {c['tokens_per_s_steady']:,.0f}  mem {c['peak_mem_gib']:.2f} GiB")
if cands:
    best = min(cands, key=lambda c: c["val_loss"])
    REV, H = best["trunk"], best["h"]
print("using:", REV, "h =", H)
"""),
    md("## 5. The assignment runs (50M tokens each)"),
    code("""
# Run 1: baseline at fixed batch
!python train.py --name run1_baseline --trunk residual --batch $B0 --tokens $TOKENS --dtype $DT
"""),
    code("""
# Run 2: reversible at the same batch (fp64 residual stream -> exact reconstruction, exact gradients)
!python train.py --name run2_rev_same --trunk $REV --h $H --stream fp64 --batch $B0 --tokens $TOKENS --dtype $DT
"""),
    code("""
# Run 3: reversible at max batch
!python find_max_batch.py --trunk $REV --h $H --stream fp64 --start 64 --headroom $HEADROOM --dtype $DT
BMAX = json.load(open(f"results/maxbatch_{REV}_fp64.json"))["max_batch"]; print("max batch =", BMAX)
!python train.py --name run3_rev_max --trunk $REV --h $H --stream fp64 --batch $BMAX --scale_lr --tokens $TOKENS --dtype $DT
"""),
    code("""
# Optional extras: Run 1b = baseline at ITS max batch (fair payoff check); Run 3b = fp32 stream (faster, approximate)
if RUN_EXTRAS:
    !python find_max_batch.py --trunk residual --headroom $HEADROOM --dtype $DT
    B1 = json.load(open("results/maxbatch_residual_fp32.json"))["max_batch"]
    !python train.py --name run1b_baseline_max --trunk residual --batch $B1 --scale_lr --tokens $TOKENS --dtype $DT
    !python find_max_batch.py --trunk $REV --h $H --stream fp32 --start 64 --headroom $HEADROOM --dtype $DT
    B3 = json.load(open(f"results/maxbatch_{REV}_fp32.json"))["max_batch"]
    !python train.py --name run3b_rev_max_fp32stream --trunk $REV --h $H --stream fp32 --batch $B3 --scale_lr --tokens $TOKENS --dtype $DT
"""),
    md("## 6. Report"),
    code("""
!python make_report.py
from IPython.display import Markdown, Image, display
display(Markdown(open("REPORT_results.md", encoding="utf-8").read().split("![")[0]))
for p in ["results/plots/screen_loss.png", "results/plots/main_loss.png", "results/plots/main_bars.png"]:
    if os.path.exists(p):
        display(Image(p))
"""),
    md("## 7. Results obtained (RTX 5070 Ti 16 GB, bf16, 50M tokens per run)\n\n" + tldr + """

**Which integrator worked:** **reversible (symplectic) Euler, h = 0.5** trained stably and matched the baseline in screening
(val 3.601 vs 3.624). Midpoint/leapfrog was stable but learned more slowly (val 4.13–4.38). Its stored-activation control was
equally slow (4.21), so the gap is an architecture effect, not a backward bug.

**Findings**
* **Memory −64% at equal batch** (3.22 → 1.16 GiB). What remains is weights/grads/Adam, embeddings, one loss chunk and one layer's activations.
* **Throughput −35% at equal batch.** The backward does one extra forward per layer (≈ +33% compute in theory); about 20 points of the 35% come from
  the fp64 residual stream. An fp32 stream is faster (−20%), but its reconstruction error corrupts gradients (5.9% error for reversible Euler with h=1.0).
* **Loss ≈ unchanged** at equal batch (val 1.890 vs 1.849).
* **Max batch 2.8× larger** (432 vs 152 under the same memory budget), **but throughput was not recovered** (158K tok/s = 65% of baseline).
  The 21M model already saturates the GPU at batch 32, so this is the *compute-bound* case where reversibility is a tax (§9.1).
* Under a fixed 50M-token budget a huge batch means few optimizer steps (227 vs 3,052), so Run 3's loss is much worse. That comes from the
  step count, not from reversibility; the baseline at its max batch shows the same effect.

> *Reversibility cut peak memory ~64%, cost ~35% throughput at equal batch, and the 2.8× larger batch in Run 3 did **not** recover
> throughput (65% of baseline). The model is compute-bound, so the trade only pays off when memory is the binding constraint
> (long context, deep models, small GPUs).*

Full write-up: `REPORT.md`.
"""),
]

# Colab results (REPORT.md section 6), once merged with `colab_summary.py --update REPORT.md`
S0, S1 = "<!-- COLAB_SECTION:START -->", "<!-- COLAB_SECTION:END -->"
if S0 in report:
    rest = report[report.index(S1) + len(S1):]
    interp = rest[:rest.index("\n## ")].strip() if "\n## " in rest else ""  # hand-written notes after the auto block
    colab = report[report.index(S0) + len(S0):report.index(S1)].strip()
    if "*Pending" not in colab:
        cells.append(md(colab.replace("## 6. ", "## 8. ", 1) + ("\n\n" + interp if interp else "")))

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(OUT, len(cells), "cells")
