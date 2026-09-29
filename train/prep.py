"""Build training/validation manifests (JSON lines) from the downloaded MDC datasets.

    python train/prep.py dev --tsv DIR/metadata.tsv --clips DIR/clips --track nh|jv --out DIR

Rows: key, audio, ref (the raw reference), text (the official normalisation of ref: the CTC target),
lang, convo. Speaker IDs are not carried. Folds are fixed by FINETUNE.md §2 (pre-registered):
  nh: V = 2 conversations per variety, random.Random(0).sample over the sorted convo_ids of each variety
  jv: V = the conversation with fewer clips (a tie goes to the lexically first convo_id)
Writes OUT/{track}_all.jsonl, OUT/{track}_T.jsonl, OUT/{track}_V.jsonl and prints the split.
"""

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import normalize, write_manifest  # noqa: E402


def dev(a) -> int:
    rows = []
    with open(a.tsv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            ref = (r.get("transcript") or "").strip()
            audio = Path(a.clips) / r["audio_filename"]
            rows.append({"key": Path(r["audio_filename"]).stem, "audio": str(audio.resolve()), "ref": ref,
                         "text": normalize(ref), "lang": (r.get("language") or "").strip(),
                         "convo": (r.get("convo_id") or "").strip()})
    missing = [r["key"] for r in rows if not Path(r["audio"]).exists()]
    empty = [r["key"] for r in rows if not r["text"]]
    by_convo = defaultdict(list)
    for r in rows:
        by_convo[r["convo"]].append(r)
    if a.track == "nh":
        by_var = defaultdict(set)
        for r in rows:
            by_var[r["lang"]].add(r["convo"])
        V = set()
        for var in sorted(by_var):
            V |= set(random.Random(0).sample(sorted(by_var[var]), 2))
    else:
        V = {min(by_convo, key=lambda c: (len(by_convo[c]), c))}
    T = [r for r in rows if r["convo"] not in V]
    Vr = [r for r in rows if r["convo"] in V]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    write_manifest(out / f"{a.track}_all.jsonl", rows)
    write_manifest(out / f"{a.track}_T.jsonl", T)
    write_manifest(out / f"{a.track}_V.jsonl", Vr)
    words = lambda rs: sum(len(r["text"].split()) for r in rs)  # noqa: E731
    print(f"{a.track}: {len(rows)} clips, {len(by_convo)} convos, langs {dict(Counter(r['lang'] for r in rows))}; "
          f"missing audio {len(missing)}, empty ref {len(empty)}")
    print(f"V = {sorted(V)}: {len(Vr)} clips, {words(Vr)} words, langs {dict(Counter(r['lang'] for r in Vr))}")
    print(f"T: {len(T)} clips, {words(T)} words, langs {dict(Counter(r['lang'] for r in T))}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dev")
    d.add_argument("--tsv", required=True)
    d.add_argument("--clips", required=True)
    d.add_argument("--track", choices=["nh", "jv"], required=True)
    d.add_argument("--out", required=True)
    a = ap.parse_args()
    return {"dev": dev}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
