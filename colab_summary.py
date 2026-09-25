"""Summarise the Colab runs, and merge them into REPORT.md locally.

In Colab:   python colab_summary.py --results results_colab --write REPORT_colab.md
Locally:    python colab_summary.py --results results_colab --update REPORT.md
            (also copies the Colab run JSON/CSV into results/ so make_report.py includes them)
"""
import argparse
import glob
import json
import os
import re
import shutil

SEC_START, SEC_END = "<!-- COLAB_SECTION:START -->", "<!-- COLAB_SECTION:END -->"
NOTE_START, NOTE_END = "<!-- COLAB_NOTE:START -->", "<!-- COLAB_NOTE:END -->"
RUN3B = "run3b_rev_max_fp32stream"
REF = "colab_run3_rev_max_fp64"


def load(results):
    runs = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(os.path.join(results, "*.json"))
            if not os.path.basename(p).startswith("maxbatch_")}
    probes = [json.load(open(p)) for p in glob.glob(os.path.join(results, "maxbatch_*.json"))]
    return runs, probes


def f(v, spec="{:.3f}"):
    return "—" if v is None else spec.format(v)


def section(runs, probes, local_run3=None):
    if RUN3B not in runs:
        return None
    r = runs[RUN3B]
    out = ["## 6. Run 3b on Google Colab (reversible, max batch, fp32 residual stream)\n",
           f"Run on **{r['gpu']}** with **{r['dtype']}** autocast, using the same tokenizer and token files "
           "as the local runs (`colab_bundle.zip`), so loss is comparable across machines. "
           "Tokens/s is comparable only between rows on the same GPU.\n",
           "| run | GPU | dtype | stream | batch | steps | final train loss | val loss | tokens/s (steady) "
           "| peak mem (GiB) | recon err |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in [RUN3B, REF]:
        if name in runs:
            x = runs[name]
            out.append(f"| {name} | {x['gpu']} | {x['dtype']} | {x['stream']} | {x['batch']} | {x['steps']} "
                       f"| {f(x['final_train_loss'])} | {f(x['val_loss'])} "
                       f"| {f(x['tokens_per_s_steady'] or x['tokens_per_s'], '{:,.0f}')} "
                       f"| {x['peak_mem_gib']:.2f} | {f(x['recon_error'], '{:.1e}')} |")
    if probes:
        out.append("\nMax-batch probes on this GPU: " + ", ".join(
            f"{p['trunk']} stream {p['stream']} → **{p['max_batch']}** (budget {p['budget_gib']:.1f} GiB)"
            for p in sorted(probes, key=lambda p: p["stream"])))

    out.append("\n**Findings (auto-generated from the numbers above):**\n")
    if r["diverged"]:
        out.append(f"- Run 3b **diverged** (non-finite loss) at step {r['steps']}; the fp32-stream variant is not "
                   "trainable at this batch/lr on this GPU.")
    rec = r.get("recon_error")
    if rec is not None:
        verdict = ("reconstruction is **unreliable** (gradients only approximate)" if rec > 1e-2 else
                   "reconstruction is accurate enough for training")
        out.append(f"- fp32-stream reconstruction error at the end of training: **{rec:.2e}** → {verdict}.")
    if REF in runs:
        g = runs[REF]
        k = lambda x: x["tokens_per_s_steady"] or x["tokens_per_s"]
        out.append(f"- Same GPU, fp32 vs fp64 stream: batch {r['batch']} vs {g['batch']} "
                   f"(**{r['batch'] / g['batch']:.2f}×**), throughput {k(r):,.0f} vs {k(g):,.0f} tok/s "
                   f"(**{100 * (k(r) / k(g) - 1):+.0f}%**), val loss {f(r['val_loss'])} vs {f(g['val_loss'])}.")
    if local_run3:
        out.append(f"- Local Run 3 (fp64 stream, {local_run3['gpu']}, batch {local_run3['batch']}): val loss "
                   f"{local_run3['val_loss']:.3f}; Colab Run 3b val loss {f(r['val_loss'])} at batch {r['batch']} "
                   f"({r['steps']} steps). Both are limited mainly by the small number of optimizer steps "
                   "under the fixed 50M-token budget.")
    return "\n".join(out) + "\n"


def replace_block(text, start, end, body):
    block = f"{start}\n{body}{end}"
    if start in text:
        return re.sub(re.escape(start) + ".*?" + re.escape(end), lambda _: block, text, flags=re.S)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_colab")
    ap.add_argument("--write", help="write a standalone markdown summary (Colab side)")
    ap.add_argument("--update", help="update this REPORT.md in place (local side)")
    args = ap.parse_args()
    runs, probes = load(args.results)
    local = json.load(open("results/run3_rev_max.json")) if os.path.exists("results/run3_rev_max.json") else None
    body = section(runs, probes, local)
    if body is None:
        raise SystemExit(f"no {RUN3B}.json in {args.results} — did the Colab run finish?")

    if args.write:
        open(args.write, "w", encoding="utf-8").write(body)
        print(f"wrote {args.write}")
    if args.update:
        for name in runs:  # make the main table/plots pick the Colab runs up
            for ext in ("json", "csv"):
                src = os.path.join(args.results, f"{name}.{ext}")
                if os.path.exists(src):
                    shutil.copy(src, os.path.join("results", f"{name}.{ext}"))
        text = open(args.update, encoding="utf-8").read()
        note = (f"Run 3b was re-run on Google Colab ({runs[RUN3B]['gpu']}); results are in §6.\n")
        text = replace_block(text, NOTE_START, NOTE_END, note) or text
        new = replace_block(text, SEC_START, SEC_END, body)
        if new is None:
            new = text.replace("## Reproduce", f"{SEC_START}\n{body}{SEC_END}\n\n## Reproduce")
        open(args.update, "w", encoding="utf-8").write(new)
        print(f"updated {args.update}; now run: python make_report.py")
    if not (args.write or args.update):
        print(body)


if __name__ == "__main__":
    main()
