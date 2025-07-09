#!/usr/bin/env python3
"""
count_tokens.py
==============#
Usage:
    python scripts/count_tokens.py --json data/all_tokens.json

• Works with either JSON array ([{…}, …]) or JSON-Lines (one object per line).
• Handles both `"sentences": [[...], ...]` and `"tokens": [...]` layouts.
• Streams the file → constant memory.
• Prints a table and writes CSV if --csv is given.
"""

import argparse, json, sys, csv
from pathlib import Path
from collections import defaultdict

def token_total(obj):
    """Return number of tokens in one document object."""
    if "sentences" in obj:
        return sum(len(s) for s in obj["sentences"])
    if "tokens" in obj:
        return len(obj["tokens"])
    return 0

def read_json_stream(fp):
    """Yield objects from either a JSON array or JSONL file."""
    first = fp.read(1)
    fp.seek(0)
    if first == "[":                               # JSON array
        data = json.load(fp)
        yield from data
    else:                                          # JSON-Lines
        for line in fp:
            if line.strip():
                yield json.loads(line)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", required=True, help="all_tokens.json or .jsonl")
    p.add_argument("--csv", help="optional output CSV path")
    args = p.parse_args()

    path = Path(args.json)
    if not path.exists():
        sys.exit(f"File not found: {path}")

    counts_lang  = defaultdict(int)
    counts_ld    = defaultdict(lambda: defaultdict(int))
    total_tokens = 0

    with path.open(encoding="utf-8") as fh:
        for rec in read_json_stream(fh):
            tok = token_total(rec)
            lang, dom = rec["lang"], rec["domain"]
            counts_lang[lang]         += tok
            counts_ld[lang][dom]      += tok
            total_tokens              += tok

    # ---- print summary ----
    print("\nToken counts by language and domain\n" + "-"*40)
    for lang, tok in sorted(counts_lang.items(), key=lambda x: -x[1]):
        print(f"{lang:<8} {tok:>12,}")
        for dom, t in sorted(counts_ld[lang].items(), key=lambda x: -x[1]):
            print(f"  • {dom:<12} {t:>12,}")
    print("-"*40)
    print(f"TOTAL      {total_tokens:>12,}")

    # ---- optional CSV ----
    if args.csv:
        csv_path = Path(args.csv)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["lang", "domain", "tokens"])
            for lang, doms in counts_ld.items():
                for dom, t in doms.items():
                    writer.writerow([lang, dom, t])
        print(f"\nCSV written → {csv_path.resolve()}")

if __name__ == "__main__":
    main()
