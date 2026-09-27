"""Join per-piece predictions (<rec>_NNN.mp3) into per-recording text, write rec-level CSV."""
import csv, sys
from collections import defaultdict
pred, out = sys.argv[1], sys.argv[2]
parts = defaultdict(list)
for r in csv.DictReader(open(pred, encoding="utf-8")):
    rec, idx = r["audio_filename"].rsplit("_", 1)
    parts[rec + ".mp3"].append((idx, r["transcript"]))
w = csv.writer(open(out, "w", newline="", encoding="utf-8")); w.writerow(["audio_filename", "transcript"])
for k, v in parts.items():
    w.writerow([k, " ".join(t for _, t in sorted(v))])
