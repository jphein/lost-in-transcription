"""Build src/nah_vocab.tsv for segment.py from reference transcripts, normalized exactly as the official scorer does.

The vocabulary is derived from the Lost in Transcription dev set (Mozilla Data Collective, CC-BY-SA-4.0), so it is
generated locally and never committed (see .gitignore).

Usage (repo root): uv run --no-project -q --with jiwer --with pandas --with typer \
    python scripts/build_nah_vocab.py <gt.csv> [<gt.csv> ...]
"""
import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
def mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return m
score = mod("score", ROOT / "runtime" / "score.py")
seg = mod("segment", ROOT / "src" / "segment.py")
texts = []
for p in sys.argv[1:]:
    texts += [score.normalize_text(t) for t in pd.read_csv(p)["transcript"].fillna("").tolist()]
uni, bi = seg.learn(texts)
out = ROOT / "src" / "nah_vocab.tsv"
seg.save(out, uni, bi)
nb = sum(1 for l in out.read_text(encoding="utf-8").splitlines() if l.startswith("b\t"))
print(f"{len(texts)} transcripts -> {len(uni)} word types ({sum(uni.values())} tokens), {nb} join bigrams -> {out} "
      f"({out.stat().st_size} bytes)")
