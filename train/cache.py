"""Cache the frozen bottom of Omni CTC: the hidden states entering layer UPTO, one clip at a time.

    python train/cache.py --model DIR --manifest rows.jsonl --out DIR [--upto 52] [--chunk]

Manifest rows (JSON lines): key, audio (path), text (the scoring target), lang, convo; optional
start/end (seconds) to cut an utterance out of a longer recording. --chunk splits each clip the way the
submission does (omni._split, ≤ 30 s at the quietest point), for validation sets.

Output: OUT/feats.f16 (all frames back to back, fp16, D columns) and OUT/index.jsonl (one line per
cached piece: key, clip, part, off, n and the row's fields). Resumable: keys already in the index are
skipped, and SIGTERM stops after the current clip with everything written so far kept.
"""

import argparse
import json
import resource
import signal
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SR, bottom_features, decode_audio, load_fp16_upcast, read_manifest  # noqa: E402

STOP = False


def _term(signum, frame):
    global STOP
    STOP = True
    print(f"SIGTERM at {time.strftime('%H:%M:%S')}: stopping after the current clip", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--upto", type=int, default=52)
    ap.add_argument("--chunk", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    signal.signal(signal.SIGTERM, _term)

    import torch
    from transformers import AutoProcessor

    from omni import _split

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    idx_path, feat_path = out / "index.jsonl", out / "feats.f16"
    done, off = set(), 0
    if idx_path.exists():
        for line in idx_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.add(r["clip"])
            off = max(off, r["off"] + r["n"])
    D = json.loads((Path(a.model) / "config.json").read_text())["hidden_size"]
    if feat_path.exists():  # drop a half-written tail from an interrupted run
        with open(feat_path, "r+b") as f:
            f.truncate(off * 2 * D)

    rows = read_manifest(a.manifest)
    if a.limit:
        rows = rows[: a.limit]
    todo = [r for r in rows if r["key"] not in done]
    print(f"{len(rows)} rows, {len(done)} cached, {len(todo)} to do; upto={a.upto} chunk={a.chunk}", flush=True)
    if not todo:
        return 0

    t0 = time.time()
    proc = AutoProcessor.from_pretrained(a.model)
    m = load_fp16_upcast(a.model, keep_layers=a.upto)
    print(f"model loaded in {time.time() - t0:.0f}s; gpu mem {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)

    cur_audio, cur_wav = None, None
    secs = 0.0
    t1 = time.time()
    with open(feat_path, "ab") as ff, open(idx_path, "a", encoding="utf-8") as fi:
        for n_done, r in enumerate(todo, 1):
            if STOP:
                break
            if r["audio"] != cur_audio:
                try:
                    cur_audio, cur_wav = r["audio"], decode_audio(r["audio"])
                except Exception as e:  # a stub or corrupt file: skip it, counted
                    cur_audio, cur_wav = r["audio"], None
                    print(f"decode failed {r['key']}: {type(e).__name__}", flush=True)
            if cur_wav is None:
                continue
            wav = cur_wav
            if "start" in r:
                wav = wav[int(r["start"] * SR) : int(r["end"] * SR)]
            if len(wav) < SR // 10:
                print(f"skip {r['key']}: {len(wav) / SR:.2f}s", flush=True)
                continue
            parts = _split(wav) if a.chunk else [wav]
            lines = []
            with torch.inference_mode():
                for j, p in enumerate(parts):
                    h = bottom_features(m, proc, p, a.upto).to(torch.float16).cpu().numpy()
                    ff.write(h.tobytes())
                    meta = {k: v for k, v in r.items() if k not in ("start", "end")}
                    lines.append(dict(meta, key=f"{r['key']}#{j}", clip=r["key"], part=j, off=off, n=int(h.shape[0])))
                    off += int(h.shape[0])
            ff.flush()
            for l in lines:
                fi.write(json.dumps(l, ensure_ascii=False) + "\n")
            fi.flush()
            secs += len(wav) / SR
            if n_done % 50 == 0 or n_done == len(todo):
                el = time.time() - t1
                print(f"{n_done}/{len(todo)} clips, {secs:.0f}s audio in {el:.0f}s ({secs / max(el, 1e-9):.1f}x), "
                      f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.2f} GiB, "
                      f"gpu peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
    print(f"CACHE_DONE {out} frames={off} audio={secs:.0f}s peak_rss_gib="
          f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.2f} stopped={STOP}", flush=True)
    return 0 if not STOP else 75


if __name__ == "__main__":
    sys.exit(main())
