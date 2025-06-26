import json
import argparse
import itertools
import yaml
from pathlib import Path
from gensim.models import Word2Vec

def filter_docs(docs, languages, domains):
    return [d["tokens"] for d in docs
            if d["lang"] in languages and d["domain"] in domains]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens_json", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="models")
    args = parser.parse_args()

    conf = yaml.safe_load(Path(args.config).read_text())
    docs = json.loads(Path(args.tokens_json).read_text())
    corpus = filter_docs(docs, conf["languages"], conf["domains"])
    
    model = Word2Vec(corpus, **conf["w2v"])
    out_path = Path(args.out_dir)/f"{conf['exp_name']}.kv"
    model.wv.save(str(out_path))

