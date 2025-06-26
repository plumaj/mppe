import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from gensim.models import KeyedVectors

def word_sample(kv, size, min_count=5):
    vocab = [w for w in kv.index_to_key if kv.get_vecattr(w, "count") >= min_count]
    return np.random.choice(vocab, size=size, replace=False)

def cohesion_metrics(kv, n_clusters=20, sample_size=10000):
    words = word_sample(kv, sample_size)
    X = kv[words]                   # shape (sample_size, dim)
    km = KMeans(n_clusters, n_init="auto").fit(X)
    sil = silhouette_score(X, km.labels_)
    db  = davies_bouldin_score(X, km.labels_)
    return {"silhouette": sil, "davies_bouldin": db}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)  # list of .kv files
    args = parser.parse_args()

    rows = []
    for mfile in args.models:
        kv = KeyedVectors.load(mfile, mmap="r")
        res = cohesion_metrics(kv)
        rows.append({"model": Path(mfile).stem} | res)

    df = pd.DataFrame(rows)
    print(df.to_markdown(index=False))
    df.to_csv("cohesion_summary.csv", index=False)

if __name__ == "__main__":
    main()

# silhoutte + is better
# davies_bouldin - is better
# python scripts/analyse_clusters.py models/*.kv
