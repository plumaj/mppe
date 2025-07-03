#!/usr/bin/env python3
import json
import yaml
import logging
import argparse
from pathlib import Path
from collections import defaultdict
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec


class EpochLogger(CallbackAny2Vec):
    def __init__(self): self.epoch = 0
    def on_epoch_begin(self, model):
        logging.info("Epoch %d start", self.epoch)
    def on_epoch_end(self, model):
        logging.info("Epoch %d end – cumulative loss: %.2f",
                     self.epoch,
                     model.get_latest_training_loss())
        self.epoch += 1


def filter_docs(docs, languages, domains, limits):
    """Yield sentences while respecting per-language token caps."""
    used_tokens = defaultdict(int)
    for d in docs:
        lang, dom = d["lang"], d["domain"]
        if lang not in languages:           # language filter
            continue
        if domains and dom not in domains:  # domain filter
            continue

        # Choose the iterable of token lists
        sents = d.get("sentences") or [d.get("tokens", [])]

        for sent in sents:
            cap = limits.get(lang)          # None == unlimited
            if cap is not None and used_tokens[lang] >= cap:
                continue                    # already hit the limit
            sent_len = len(sent)
            if cap is not None:
                if used_tokens[lang] + sent_len > cap:
                    # trim the sentence to fit exactly into the budget
                    sent = sent[: cap - used_tokens[lang]]
                    sent_len = len(sent)
                used_tokens[lang] += sent_len
            yield sent
    logging.info("Token usage per language: %s",
                 {k: v for k, v in used_tokens.items()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens_json", required=True)
    parser.add_argument("--config",      required=True)
    parser.add_argument("--out_dir",     default="models")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    conf = yaml.safe_load(Path(args.config).read_text())
    logging.info("Experiment name: %s", conf["exp_name"])

    docs = json.loads(Path(args.tokens_json).read_text())

    token_limits = conf.get("token_limits", {})      # may be empty
    corpus = list(
        filter_docs(docs,
                    conf["languages"],
                    conf.get("domains"),
                    token_limits)
    )

    if not corpus:
        raise ValueError("Corpus empty after filtering / throttling.")

    logging.info("Total sentences kept: %d", len(corpus))

    w2v_params = dict(conf["w2v"])
    w2v_params.setdefault("workers", 4)
    w2v_params.setdefault("compute_loss", True)

    model = Word2Vec(corpus, callbacks=[EpochLogger()], **w2v_params)

    out_path = Path(args.out_dir) / f"{conf['exp_name']}.kv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.wv.save(str(out_path))
    logging.info("Model saved to %s", out_path.resolve())


if __name__ == "__main__":
    main()
