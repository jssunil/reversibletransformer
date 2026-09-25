"""Download TinyStories, train an 8K byte-level BPE, and write uint16 token files.

Outputs (in data/):
  tokenizer.json   - the trained BPE
  train.bin        - >= --train_tokens tokens (uint16)
  val.bin          - --val_tokens tokens from the validation split (uint16)
"""
import argparse
import os

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

EOS = "<|endoftext|>"
RAW = {"train": "data/raw/train0.parquet", "validation": "data/raw/val.parquet"}


def stories(split):
    """Stream TinyStories; prefer local parquet (curl-downloaded) over the Hub."""
    if os.path.exists(RAW[split]):
        return load_dataset("parquet", data_files=RAW[split], split="train", streaming=True)
    return load_dataset("roneneldan/TinyStories", split=split, streaming=True)


def train_tokenizer(texts, vocab_size):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOS],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(texts, trainer=trainer)
    return tok


def write_tokens(tok, stream, path, target, batch=2000):
    eos_id = tok.token_to_id(EOS)
    buf = np.empty(target + 1_000_000, dtype=np.uint16)
    n = 0
    texts = []

    def flush():
        nonlocal n
        for enc in tok.encode_batch(texts):
            ids = enc.ids + [eos_id]
            k = min(len(ids), len(buf) - n)
            buf[n:n + k] = ids[:k]
            n += k
        texts.clear()

    for ex in stream:
        texts.append(ex["text"])
        if len(texts) == batch:
            flush()
            print(f"\r{path}: {n / 1e6:.1f}M tokens", end="", flush=True)
            if n >= target:
                break
    if texts and n < target:
        flush()
    print()
    buf[:n].tofile(path)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--tok_docs", type=int, default=200_000)
    ap.add_argument("--train_tokens", type=int, default=55_000_000)
    ap.add_argument("--val_tokens", type=int, default=1_000_000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tok_path = os.path.join(args.out, "tokenizer.json")
    if os.path.exists(tok_path):
        tok = Tokenizer.from_file(tok_path)
    else:
        ds = stories("train")
        texts = [ex["text"] for _, ex in zip(range(args.tok_docs), ds)]
        tok = train_tokenizer(texts, args.vocab)
        tok.save(tok_path)
    print("vocab size:", tok.get_vocab_size())

    train = stories("train")
    n = write_tokens(tok, train, os.path.join(args.out, "train.bin"), args.train_tokens)
    print(f"train tokens: {n:,}")
    val = stories("validation")
    n = write_tokens(tok, val, os.path.join(args.out, "val.bin"), args.val_tokens)
    print(f"val tokens: {n:,}")


if __name__ == "__main__":
    main()
