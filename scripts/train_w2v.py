#!/usr/bin/env python3
import json, yaml, logging, argparse, random
from pathlib import Path
from collections import defaultdict

from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec


class EpochLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
    def on_epoch_begin(self, model):
        logging.info("Epoch %d start", self.epoch)
    def on_epoch_end(self, model):
        logging.info("Epoch %d end – cumulative loss: %.2f",
                     self.epoch,
                     model.get_latest_training_loss())
        self.epoch += 1


def build_corpus(docs, languages, domains, limits, targets, seed):
    """
    • Apply language/domain filters.
    • Apply per-language MAX cap (limits).
    • Apply per-language MIN requirement (targets) by sampling
      sentences *with replacement* until the target is met.
    Returns: list of sentences (list[str])
    """
    rng = random.Random(seed)
    lang2sent = defaultdict(list)
    lang2tok  = defaultdict(int)

    # pass 1 – collect / cap
    for d in docs:
        lang, dom = d["lang"], d["domain"]
        if lang not in languages:
            continue
        if domains and dom not in domains:
            continue

        sentences = d.get("sentences") or [d.get("tokens", [])]
        cap = limits.get(lang)         # None = unlimited
        for sent in sentences:
            sent_len = len(sent)
            if cap is not None and lang2tok[lang] >= cap:
                break
            if cap is not None and lang2tok[lang] + sent_len > cap:
                sent = sent[: cap - lang2tok[lang]]
                sent_len = len(sent)
            lang2sent[lang].append(sent)
            lang2tok[lang] += sent_len

    # pass 2 – upsample if below target
    for lang, target in targets.items():
        current = lang2tok.get(lang, 0)
        if current >= target:
            continue
        if lang not in lang2sent or not lang2sent[lang]:
            logging.warning("Upsample requested but no data for language %s", lang)
            continue
        logging.info("Upsampling %s: %d → %d tokens", lang, current, target)
        while current < target:
            sent = rng.choice(lang2sent[lang])
            needed = target - current
            if len(sent) > needed:
                sent = sent[:needed]
            lang2sent[lang].append(sent)
            current += len(sent)
        lang2tok[lang] = current

    logging.info("Final token counts: %s", dict(lang2tok))

    # flatten into one big corpus list
    corpus = [s for lang in lang2sent for s in lang2sent[lang]]
    rng.shuffle(corpus)                      # better mixing
    return corpus


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens_json", required=True)
    p.add_argument("--config",      required=True)
    p.add_argument("--out_dir",     default="models")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = yaml.safe_load(Path(args.config).read_text())
    seed = cfg.get("random_seed", 12345)
    random.seed(seed)

    docs = json.loads(Path(args.tokens_json).read_text())

    corpus = build_corpus(
        docs,
        languages = cfg["languages"],
        domains   = cfg.get("domains"),
        limits    = cfg.get("token_limits", {}),
        targets   = cfg.get("token_targets", {}),
        seed      = seed,
    )

    if not corpus:
        raise ValueError("Corpus empty after filtering / throttling / upsampling.")

    logging.info("Total sentences in training corpus: %d", len(corpus))

    w2v_params = dict(cfg["w2v"])
    w2v_params.setdefault("workers", 4)
    w2v_params.setdefault("compute_loss", True)

    model = Word2Vec(corpus, callbacks=[EpochLogger()], **w2v_params)

    out_path = Path(args.out_dir) / f"{cfg['exp_name']}.kv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.wv.save(str(out_path))
    logging.info("Model saved → %s", out_path.resolve())


if __name__ == "__main__":
    main()
