"""Turn an MDC dev set (clips/ + metadata.tsv) into the runtime layout.

Usage: python3 scripts/prep_dev.py <dev_dir> <out_dir>
Writes <out_dir>/clips -> symlinked mp3s, test_metadata.csv, submission_format.csv, gt.csv
"""
import csv, subprocess, sys
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
(out / "clips").mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader((src / "metadata.tsv").open(encoding="utf-8"), delimiter="\t"))
meta, gt = [], []
for r in rows:
    fn = r["audio_filename"]
    dst = out / "clips" / fn
    if not dst.exists():
        dst.write_bytes((src / "clips" / fn).read_bytes())
    d = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(dst)]))
    meta.append([fn, round(d), r["language"]])
    gt.append([fn, r["transcript"]])
def w(name, header, data):
    with (out / name).open("w", newline="", encoding="utf-8") as f:
        cw = csv.writer(f); cw.writerow(header); cw.writerows(data)
w("test_metadata.csv", ["audio_filename", "file_duration_seconds", "language"], meta)
w("submission_format.csv", ["audio_filename", "transcript"], [[m[0], ""] for m in meta])
w("gt.csv", ["audio_filename", "transcript"], gt)
print(f"{len(rows)} clips, {sum(m[1] for m in meta)/60:.1f} min -> {out}")
