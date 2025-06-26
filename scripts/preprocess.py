#!/usr/bin/env python
from __future__ import annotations
from pathlib import Path
import argparse, json
import spacy
from spacy.lang.lb import Luxembourgish

LANG_MAP = {"lb": "lb", "de": "de_core_news_sm", "fr": "fr_core_news_sm"}


def make_nlp(lang: str):
    """Tokenizer-only pipeline for a given language."""
    if lang == "lb":
        nlp = Luxembourgish()
        nlp.add_pipe("sentencizer")
    else:
        nlp = spacy.load(
            LANG_MAP.get(lang, "xx_ent_wiki_sm"),
            disable=[
                "parser",
                "ner",
                "tagger",
                "attribute_ruler",
                "lemmatizer",
            ],
        )
    return nlp


def tokens_from_file(path: Path, nlp, *, batch_size: int = 1000):
    """
    Yield lowercase alphabetic tokens from *one text file*.
    spaCy sees at most `batch_size` lines per call, so memory stays tiny.
    """
    with path.open() as f:
        for doc in nlp.pipe(f, batch_size=batch_size):
            for tok in doc:
                if tok.is_alpha:
                    yield tok.text


def parse_fname(fname: str):
    lang, domain, *_ = Path(fname).stem.split("_")
    return lang, domain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data"),
        help="directory full of *.txt files",
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output JSON file (give a filename, NOT a dir!)",
    )
    args = ap.parse_args()

    files = sorted(args.data_dir.glob("*.txt"))
    if not files:
        raise SystemExit(f"No .txt files in {args.data_dir}")

    # one spaCy object per language, reused across files
    nlp_cache: dict[str, spacy.language.Language] = {}

    with args.out.open("w") as out_f:
        out_f.write("[\n")
        first = True

        for path in files:
            print(f"Processing {path.name}")
            lang, domain = parse_fname(path.name)
            nlp = nlp_cache.setdefault(lang, make_nlp(lang))

            tokens = list(tokens_from_file(path, nlp))
            doc_obj = {"lang": lang, "domain": domain, "tokens": tokens}

            if not first:
                out_f.write(",\n")
            json.dump(doc_obj, out_f)
            first = False

        out_f.write("\n]\n")

    print(f"Processed {len(files)} files → {args.out}")


if __name__ == "__main__":
    main()
