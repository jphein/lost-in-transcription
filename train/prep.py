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


def csvset(a) -> int:
    """A proxy set in the runtime layout: clips/, test_metadata.csv and a ground-truth CSV."""
    lang = {}
    with open(Path(a.dir) / "test_metadata.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            lang[r["audio_filename"]] = (r.get("language") or "").strip()
    rows = []
    with open(a.gt, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ref = (r.get("transcript") or "").strip()
            rows.append({"key": Path(r["audio_filename"]).stem, "audio": str((Path(a.dir) / "clips" / r["audio_filename"]).resolve()),
                         "ref": ref, "text": normalize(ref), "lang": lang.get(r["audio_filename"], ""), "convo": ""})
    write_manifest(a.out, rows)
    print(f"{a.out}: {len(rows)} rows, langs {dict(Counter(r['lang'] for r in rows))}")
    return 0


MAP_TAGS = {"nhi", "mixed", "asl", "adapted-spanish-loan", "adapted_spanish_loan"}


def tet_text(sentence: str, lid: str) -> str:
    """Tetelancingo SMT orthography -> the competition's (u/k/s/j), per FINETUNE.md §9 09:10: map words
    tagged Nahuatl / mixed / adapted loan with nahuatl.to_reference; keep Spanish and names as written."""
    from nahuatl import to_reference

    toks = [t for t in (lid or "").split() if "//" in t]
    if toks:
        out = []
        for t in toks:
            w, _, tag = t.rpartition("//")
            out.append(to_reference(w) if tag.strip().lower() in MAP_TAGS else w)
        return " ".join(out)
    return to_reference(sentence)


def tetelancingo(a) -> int:
    rows = {"train": [], "test": []}
    base = Path(a.tsv).parent
    with open(a.tsv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            audio = Path(a.audio_dir) / r["audio_file"] if a.audio_dir else base / r["audio_file"]
            ref = tet_text(r.get("sentence", ""), r.get("lid_tokens", ""))
            split = (r.get("split") or "train").strip().lower()
            rows.setdefault(split, []).append({"key": "tet_" + Path(r["audio_file"]).stem, "audio": str(audio.resolve()),
                                               "ref": ref, "text": normalize(ref), "lang": "nhi", "convo": "tet"})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for split, rs in rows.items():
        if rs:
            write_manifest(out / f"tet_{split}.jsonl", rs)
            miss = sum(not Path(r["audio"]).exists() for r in rs)
            print(f"tet_{split}: {len(rs)} rows, {sum(len(r['text'].split()) for r in rs)} words, missing audio {miss}")
    return 0


def _secs(t: str) -> float:
    parts = [float(x) for x in str(t).strip().split(":")]
    v = 0.0
    for x in parts:
        v = v * 60 + x
    return v


def jember(a) -> int:
    """Jember sessions: cut utterances at the TSV timestamps, padded by PAD s each side (FINETUNE.md §9:
    the timestamps are whole seconds, so a word at an edge would otherwise be cut); > 30 s or empty -> drop."""
    rows, drop = [], Counter()
    for tsv in sorted(Path(a.root).rglob("*.tsv")):
        with open(tsv, newline="", encoding="utf-8") as f:
            rd = csv.DictReader(f, delimiter="\t")
            cols = {c.lower().strip(): c for c in rd.fieldnames or []}
            c_file = next((cols[k] for k in cols if "file" in k or "audio" in k), None)
            c_s, c_e = cols.get("start"), cols.get("end")
            c_t = cols.get("text") or cols.get("transcript") or cols.get("sentence")
            if not (c_file and c_s and c_e and c_t):
                print(f"skip {tsv.name}: columns {rd.fieldnames}")
                continue
            for i, r in enumerate(rd):
                ref = (r[c_t] or "").strip()
                s0, e0 = _secs(r[c_s]), _secs(r[c_e])
                if not ref:
                    drop["empty"] += 1
                    continue
                if e0 - s0 > 30 or e0 <= s0:
                    drop["len"] += 1
                    continue
                name = str(r[c_file]).strip()
                cands = list(Path(a.root).rglob(name if name.endswith(".mp3") else name + ".mp3"))
                if not cands:
                    drop["no_audio"] += 1
                    continue
                rows.append({"key": f"jem_{Path(name).stem}_{i:04d}", "audio": str(cands[0].resolve()),
                             "start": max(0.0, s0 - a.pad), "end": e0 + a.pad, "ref": ref, "text": normalize(ref),
                             "lang": "jav", "convo": "jem_" + Path(name).stem})
    rows.sort(key=lambda r: (r["audio"], r["start"]))  # cache.py decodes each session once
    write_manifest(a.out, rows)
    print(f"{a.out}: {len(rows)} utterances, {sum(r['end'] - r['start'] for r in rows) / 3600:.2f} h, "
          f"{len({r['audio'] for r in rows})} sessions; dropped {dict(drop)}")
    return 0


def runtime_dir(a) -> int:
    """A manifest -> the competition's /code_execution/data layout (clips/ as hard links, test_metadata.csv)
    plus gt.csv, so a zip can be scored on exactly these clips in the official image."""
    import os

    from common import decode_audio, read_manifest

    rows = read_manifest(a.manifest)
    out = Path(a.out)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    with open(out / "test_metadata.csv", "w", newline="", encoding="utf-8") as fm, \
         open(out / "gt.csv", "w", newline="", encoding="utf-8") as fg:
        wm, wg = csv.writer(fm), csv.writer(fg)
        wm.writerow(["audio_filename", "file_duration_seconds", "language"])
        wg.writerow(["audio_filename", "transcript"])
        for r in rows:
            src = Path(r["audio"])
            dst = out / "clips" / src.name
            if not dst.exists():
                os.link(src, dst)
            try:
                dur = round(len(decode_audio(src)) / 16000)
            except Exception:
                dur = 0
            wm.writerow([src.name, dur, r["lang"]])
            wg.writerow([src.name, r["ref"]])
    print(f"{out}: {len(rows)} clips")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    rd = sub.add_parser("runtime-dir")
    rd.add_argument("--manifest", required=True)
    rd.add_argument("--out", required=True)
    j = sub.add_parser("jember")
    j.add_argument("--root", required=True)
    j.add_argument("--pad", type=float, default=0.25)
    j.add_argument("--out", required=True)
    t = sub.add_parser("tetelancingo")
    t.add_argument("--tsv", required=True)
    t.add_argument("--audio-dir", default="")
    t.add_argument("--out", required=True)
    c = sub.add_parser("csvset")
    c.add_argument("--dir", required=True)
    c.add_argument("--gt", required=True)
    c.add_argument("--out", required=True)
    d = sub.add_parser("dev")
    d.add_argument("--tsv", required=True)
    d.add_argument("--clips", required=True)
    d.add_argument("--track", choices=["nh", "jv"], required=True)
    d.add_argument("--out", required=True)
    a = ap.parse_args()
    return {"dev": dev, "csvset": csvset, "tetelancingo": tetelancingo, "jember": jember,
            "runtime-dir": runtime_dir}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
