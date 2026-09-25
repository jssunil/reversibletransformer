# Reversible LLM Training & Benchmark

Train a ~20M-param GPT for 50M tokens three ways (baseline / reversible same batch / reversible max batch)
and measure loss, tokens/s and peak memory. See `REPORT.md` for results and findings.
See `explainer.md` for a plain-English walkthrough; its §9 compares reversible training with how frontier labs handle
activation memory (checkpointing, TP/SP/PP/CP) and explains where this approach wins.

## Files
| File | Purpose |
|---|---|
| `prepare_data.py` | TinyStories → 8K byte-level BPE → `data/train.bin` (55M tok), `data/val.bin` (1.2M tok) |
| `model.py` | GPT (d=384, L=10, 6 heads, T=512, tied emb, 21.05M params) with trunks `residual`, `midpoint[_rev]`, `reveuler[_rev]` |
| `reversible.py` | Memory-free backward (`RevMidpointFn`, `RevEulerFn`) + `reconstruction_error` |
| `train.py` | Token-budgeted training, logs CSV + JSON (loss, val loss, tokens/s, peak memory) |
| `find_max_batch.py` | Largest batch that fits (doubling + binary search, real optimizer steps) |
| `tests/test_reversible.py` | Grad equivalence rev vs. stored (fp64), reconstruction, memory flat in depth |
| `run_all.sh` | `screen` (integrator screening, 5M tok) and `main` (the assignment runs) |
| `make_report.py` | Tables + plots → `REPORT_results.md`, `results/plots/` |
| `colab_summary.py` | Colab: writes `REPORT_colab.md`; locally: merges Colab results into `REPORT.md` §6 |
| `Reversible_Assignment.ipynb` | **Assignment notebook**: all tested code embedded (`%%writefile`), tests, screening, Runs 1/2/3, report, results + findings |
| `colab_run3b.ipynb` | Colab notebook doing the whole Run 3b flow (alternative to pasting cells) |
| `colab_bundle.zip` | Code + tokenizer + `train.bin`/`val.bin` for Colab (same data as the local runs) |

## Reproduce
```bash
python prepare_data.py          # if the HF hub SSL fails, curl the parquet files into data/raw/ (see script)
python -m pytest tests -q
bash run_all.sh screen          # pick integrator
bash run_all.sh main            # Run 1/2/3 (+ 1b, 3b)
```
Colab (T4): add `--dtype fp16` to `train.py` calls (no bf16 on T4) and expect ~4-5x lower tokens/s.

## Tests
`python -m pytest tests -q` → expect **6 passed** (runs in seconds; the last test needs a CUDA GPU and is skipped otherwise).

| Test | What it checks |
|---|---|
| `test_grad_equivalence[midpoint-0.5]`, `[midpoint-0.25]`, `[reveuler-1.0]` | Memory-free reversible backward gives the **same loss and every parameter gradient** as ordinary autograd on the same network (fp64, tolerance 1e-6) |
| `test_reconstruction[midpoint_rev]`, `[reveuler_rev]` | Running the trunk forward then inverting it recovers the input: relative error < 1e-5 |
| `test_memory_flat_in_depth` | Going 4 → 16 layers, the reversible trunk's activation memory grows < 20% of the stored trunk's growth |

## Running Run 3b on Google Colab
Run 3b (reversible, max batch, fp32 residual stream) is run on Colab, and its report is merged back here.

**Before you start:** put `colab_bundle.zip` in Google Drive → My Drive (or upload it when Cell 1 asks).
In Colab: New notebook → Runtime → Change runtime type → **T4 GPU** → Save. Paste each cell and run it in order.
(Or upload `colab_run3b.ipynb` via File → Upload notebook and do Runtime → Run all.)

**Cell 1: setup and unzip**
```python
import os, json, torch
assert torch.cuda.is_available(), "No GPU: Runtime > Change runtime type > T4 GPU"
DT = "bf16" if torch.cuda.get_device_capability()[0] >= 8 else "fp16"
print(torch.cuda.get_device_name(), "| dtype:", DT)

from google.colab import drive, files
drive.mount("/content/drive")
ZIP = "/content/drive/MyDrive/colab_bundle.zip"
if not os.path.exists(ZIP):
    ZIP = "/content/" + list(files.upload().keys())[0]
!unzip -o -q "$ZIP" -d /content/revtransformer
%cd /content/revtransformer

!ls data
```

**Cell 2: tests** (expect `6 passed`)
```python
!python -m pytest tests -q
```

**Cell 3: Run 3b** (~20-30 min on a T4)
```python
!python find_max_batch.py --trunk reveuler_rev --h 0.5 --stream fp32 --start 64 --headroom 0.90 --dtype $DT --out results_colab
B = json.load(open("results_colab/maxbatch_reveuler_rev_fp32.json"))["max_batch"]; print("batch =", B)
!python train.py --name run3b_rev_max_fp32stream --trunk reveuler_rev --h 0.5 --stream fp32 --batch $B --scale_lr --dtype $DT --out results_colab
```

**Cell 4: same-GPU fp64 reference** (optional; tokens/s is only comparable on the same GPU)
```python
!python find_max_batch.py --trunk reveuler_rev --h 0.5 --stream fp64 --start 64 --headroom 0.90 --dtype $DT --out results_colab
B64 = json.load(open("results_colab/maxbatch_reveuler_rev_fp64.json"))["max_batch"]; print("batch =", B64)
!python train.py --name colab_run3_rev_max_fp64 --trunk reveuler_rev --h 0.5 --stream fp64 --batch $B64 --scale_lr --dtype $DT --out results_colab
```

**Cell 5: report in Colab + download**
```python
!python colab_summary.py --results results_colab --write REPORT_colab.md
from IPython.display import Markdown, display
display(Markdown(open("REPORT_colab.md").read()))
!zip -q -r colab_results.zip results_colab REPORT_colab.md
!cp colab_results.zip /content/drive/MyDrive/
files.download("colab_results.zip")
```

**Back on the PC (CPU only):** unzip `colab_results.zip` into this folder (giving `results_colab\`), then
```
python colab_summary.py --results results_colab --update REPORT.md
python make_report.py
```
This replaces the "stopped by request" note in `REPORT.md`, fills in §6 (table + findings), and copies the Colab runs into
`results/` so they appear in `REPORT_results.md` with a GPU column.

Notes: if a run prints `loss is nan, stopping` under fp16, rerun that cell without `--scale_lr`. Check `recon_error` in
the Run 3b JSON: above ~1e-2 means the fp32-stream reconstruction (and so the gradients) is unreliable.

## Integrators
* **midpoint / leapfrog** — `x_{l+1} = x_{l-1} + 2h f_l(x_l)`, inverse `x_{l-1} = x_{l+1} - 2h f_l(x_l)` (Euler starter step for x_1).
* **reversible (symplectic) Euler / RevNet coupling** — two streams, `y' = y + h Attn(z)`, `z' = z + h MLP(y')`;
  exact inverse `z = z' - h MLP(y')`, `y = y' - h Attn(z)`. Output `(y+z)/2`.

Both keep only the final state(s) after the forward; backward reconstructs each layer's input and recomputes one layer at a time.

## Implementation notes (things that bit)
* **Residual stream precision.** With an fp32 stream, `(a+b)-b != a` rounding errors are amplified layer by layer on
  reconstruction (rel. error 0.1–4 on x_0 for trained weights; reveuler grads ~6% off). A **float64 residual stream**
  (the stored states are only 2 tensors, the blocks still compute in bf16) makes reconstruction exact and grads match
  stored-activation autograd to bf16 noise. Costs ~20% throughput on a GeForce card (weak fp64).
* **Call `loss.backward()` outside `autocast`** (as `train.py` does). Calling it inside skews the recomputation in the
  custom backward.
* **LM-head memory.** With an 8K vocab the fp32 logits still dominate at large batch, so all runs use a checkpointed,
  chunked cross-entropy (8,192 tokens/chunk). Without it, max batch was 104 (baseline) vs 176 (reversible); with it, 152 vs 432 (reveuler, fp64 stream).
* `find_max_batch.py` budgets 80% of free VRAM: on Windows a nearly full card spills to host RAM and training slows 20-60% instead of OOM-ing.
* No dropout inside blocks: the backward recomputes `f(x)` and must see the same function.
