#!/usr/bin/env python3
"""
preprocess.py  —  streaming, line-wise sentences with verbose logging
"""
import argparse, json, logging, sys
from pathlib import Path
import spacy
from spacy.lang.lb import Luxembourgish

LANG_MODEL = {
    "lb": "lb",                     # handled separately
    "de": "de_core_news_sm",
    "fr": "fr_core_news_sm",
}

# -------------------------------------------------------------------- #
#  spaCy loader (cached)
# -------------------------------------------------------------------- #
def get_nlp(lang: str, cache: dict[str, spacy.Language]) -> spacy.Language:
    if lang in cache:
        return cache[lang]
    if lang == "lb":
        nlp = Luxembourgish()
    else:
        nlp = spacy.load(
            LANG_MODEL.get(lang, "xx_ent_wiki_sm"),
            disable=["ner", "tagger", "lemmatizer"],
        )
    cache[lang] = nlp
    logging.debug("Loaded spaCy model for %s", lang)
    return nlp


# -------------------------------------------------------------------- #
#  per-file processing (one line == one sentence)
# -------------------------------------------------------------------- #
def process_file(fp: Path, nlp: spacy.Language, log_sentences: bool):
    sentences = []
    with fp.open(encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            doc = nlp(line)
            toks = [tok.text for tok in doc if tok.is_alpha]
            if toks:
                sentences.append(toks)
    if log_sentences:
        logging.debug("  %s → %d sentences", fp.name, len(sentences))

    lang, domain, *_ = fp.stem.split("_", 2)
    return {"lang": lang, "domain": domain, "sentences": sentences}


# -------------------------------------------------------------------- #
#  main driver
# -------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",  required=True)
    ap.add_argument("--out_json",  required=True)
    ap.add_argument("--log_every", type=int, default=1,
                    help="progress line every N files")
    ap.add_argument("--sentence_log", action="store_true",
                    help="log each file's sentence count (verbose)")
    ap.add_argument("--loglevel", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    args = ap.parse_args()

    # ---------- logging ----------
    logging.basicConfig(
        level=getattr(logging, args.loglevel),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    files = sorted(Path(args.data_dir).glob("*.txt"))
    if not files:
        logging.error("No .txt files found in %s", args.data_dir)
        sys.exit(1)

    logging.info("Starting preprocessing")
    logging.info("  data dir : %s", args.data_dir)
    logging.info("  out file : %s", args.out_json)
    logging.info("  files    : %d", len(files))
    logging.info("  log freq : every %d files", args.log_every)

    nlp_cache: dict[str, spacy.Language] = {}
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    processed, total_sent = 0, 0
    first_doc = True

    with out_path.open("w", encoding="utf-8") as out:
        out.write("[\n")
        for idx, fp in enumerate(files, 1):
            lang = fp.stem.split("_", 1)[0]
            record = process_file(fp,
                                  get_nlp(lang, nlp_cache),
                                  args.sentence_log)
            total_sent += len(record["sentences"])

            if not first_doc:
                out.write(",\n")
            first_doc = False
            json.dump(record, out, ensure_ascii=False, indent=2)

            processed += 1
            if processed % args.log_every == 0:
                logging.info("processed %d / %d files (sentences so far: %d)",
                             processed, len(files), total_sent)

        out.write("\n]\n")

    logging.info("Finished: %d files → %d sentences", processed, total_sent)
    logging.info("Output written to %s", out_path.resolve())


if __name__ == "__main__":
    main()
