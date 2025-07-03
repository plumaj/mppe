#!/usr/bin/env python3
import json
import yaml
import logging
import argparse
from pathlib import Path

from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec


class EpochLogger(CallbackAny2Vec):
    """Log loss and epoch progress."""
    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        logging.info(f"Epoch {self.epoch} start")

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        logging.info(f"Epoch {self.epoch} end – cumulative loss: {loss:.2f}")
        self.epoch += 1


def filter_docs(docs, languages, domains):
    """
    Yield sentence lists for the languages / domains requested.
    Works with either format:
      {…, "sentences": [[tok1, tok2], ...]}
      {…, "tokens":    [tok1, tok2, tok3, ...]}
    """
    for d in docs:
        if d["lang"] not in languages:
            continue
        if domains and d["domain"] not in domains:
            continue

        if "sentences" in d:
            for sent in d["sentences"]:
                yield sent
        elif "tokens" in d:
            yield d["tokens"]
        else:
            # skip silently or raise an error if you prefer
            continue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens_json", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_dir", default="models")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg_path = Path(args.config)
    conf = yaml.safe_load(cfg_path.read_text())

    logging.info("Loaded config %s", cfg_path.name)
    logging.info("Experiment name: %s", conf["exp_name"])

    docs = json.loads(Path(args.tokens_json).read_text())
    corpus = list(filter_docs(docs,
                              conf["languages"],
                              conf.get("domains")))

    if not corpus:
        raise ValueError(
            "Filtered corpus is empty – check that `languages` / `domains` in "
            f"{cfg_path.name} match what's in {args.tokens_json}."
        )

    logging.info("Sentences in corpus: %d", len(corpus))

    w2v_params = dict(conf["w2v"])
    w2v_params.setdefault("workers", 4)
    w2v_params.setdefault("compute_loss", True)

    model = Word2Vec(
        corpus,
        callbacks=[EpochLogger()],
        **w2v_params,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{conf['exp_name']}.kv"
    model.wv.save(str(out_path))
    logging.info("Model saved to %s", out_path.resolve())


if __name__ == "__main__":
    main()
