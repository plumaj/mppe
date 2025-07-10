#!/usr/bin/env python3
"""
Evaluate Word2Vec models with a pseudo masked-token task.

For each token w in each test sentence:
  • build a context vector = mean(embeddings of all other words)
  • ask the model for its top-k nearest neighbours to that vector
  • hit if w is in that top-k list
Aggregate hits / totals → accuracy@k  (higher = better LB coverage)

Usage
-----
python scripts/pseudo_mlm_eval.py \
       --sentences data/lb_holdout.txt \
       --models models/lb.kv models/lb_de_fr.kv \
       --k 10 \
       --csv results/pseudo_mlm.csv
"""
import argparse, logging, csv
from pathlib import Path
from collections import defaultdict

import numpy as np
from gensim.models import KeyedVectors
from tqdm import tqdm   # progress bar

def load_sentences(path, max_sent=None):
    with Path(path).open(encoding="utf-8") as fh:
        for idx, line in enumerate(fh, 1):
            line = line.strip()
            if line:
                yield line.split()
            if max_sent and idx >= max_sent:
                break

def sent2ctx(sent, wv):
    """Return matrix (len-1, dim) & index map of in-vocab tokens."""
    rows = [wv[tok] for tok in sent if tok in wv]
    return np.array(rows) if rows else None

def accuracy_at_k(model_path, sents, k):
    wv = KeyedVectors.load(model_path, mmap="r")
    hit, total = 0, 0
    dim = wv.vector_size

    for sent in sents:
        # collect in-vocab tokens once per sentence
        iv_tok = [tok for tok in sent if tok in wv]
        if len(iv_tok) < 2:
            continue
        vecs = np.vstack([wv[t] for t in iv_tok])          # (n, dim)
        for i, w in enumerate(iv_tok):
            # context = mean of all other word vectors
            ctx_vec = vecs[np.arange(len(iv_tok)) != i].mean(axis=0)
            # gensim 4.x: can query directly with a vector
            nn = wv.similar_by_vector(ctx_vec, topn=k)
            total += 1
            if any(w == nn_word for nn_word, _ in nn):
                hit += 1
    return hit / total if total else 0.0

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sentences", required=True,
                   help="clean LB sentences, one per line")
    p.add_argument("--models", nargs="+", required=True,
                   help="list of *.kv models to test")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--max_sent", type=int,
                   help="optional cap on #sentences for speed")
    p.add_argument("--csv", help="optional CSV output")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    all_sents = list(load_sentences(args.sentences, args.max_sent))
    logging.info("Loaded %d test sentences", len(all_sents))

    results = []
    for model in tqdm(args.models, desc="Models"):
        acc = accuracy_at_k(model, all_sents, args.k)
        logging.info("%s  accuracy@%d = %.4f", Path(model).stem, args.k, acc)
        results.append((Path(model).stem, acc))

    # nice table
    print("\naccuracy@%d" % args.k)
    print("-------------------------------")
    for name, acc in results:
        print(f"{name:<20s} {acc:.4f}")
    print("-------------------------------")

    # optional CSV
    if args.csv:
        with Path(args.csv).open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model", f"acc@{args.k}"])
            writer.writerows(results)
        logging.info("CSV written → %s", args.csv)

if __name__ == "__main__":
    main()
