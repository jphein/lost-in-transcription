"""Batched vs batch-1 Omni inference, reproducing the submission's batching exactly on a small card.

    python train/batch_ab.py --model DIR --manifest dev.jsonl --out DIR [--budgets 240 1]

Without an attention mask, a clip's output in a batch depends only on its own samples zero-padded to
the batch's longest chunk (the processor pads, then normalises each padded row over its full length;
the model and the decoder see every padded frame). So each chunk is run alone, padded to the length its
batch would have had under main.run_omni's groups of 64 (longest first) and omni.OmniCTC's budget rule
(bs = budget // longest chunk). That is the platform's computation on an A100 without needing its memory,
and budget 1 is plain batch-1. fp32 compute from the fp16 weights (Pascal hosts).
Writes OUT/pred_b<budget>.csv per arm and prints corpus WER overall and per language.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEVICE, SR, decode_audio, load_fp16_upcast, normalize, read_manifest  # noqa: E402


def plan_lengths(wavs, budget, split):
    """Per (clip index, chunk index): the padded length (samples) its batch would have."""
    pieces = []
    for i, w in enumerate(wavs):
        for j, c in enumerate(split(w)):
            pieces.append((i, j, c))
    pieces.sort(key=lambda p: -len(p[2]))  # as in OmniCTC.transcribe_many
    target, k = {}, 0
    while k < len(pieces):
        longest = max(len(pieces[k][2]) / SR, 1.0)
        bs = max(1, int(budget // longest))
        group = pieces[k : k + bs]
        tmax = max(len(g[2]) for g in group)
        for i, j, c in group:
            target[(i, j)] = tmax
        k += len(group)
    return pieces, target


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budgets", type=float, nargs="+", default=[240.0, 1.0])
    a = ap.parse_args()

    import jiwer
    import torch
    from transformers import AutoProcessor

    import main as sub
    from omni import OmniCTC, _split

    rows = read_manifest(a.manifest)
    wav = {}
    for r in rows:
        try:
            wav[r["key"]] = decode_audio(r["audio"])
        except Exception:
            wav[r["key"]] = None
    # main() sorts the manifest by file_duration_seconds (integer seconds), longest first; stable sort
    order = sorted(rows, key=lambda r: -round(len(wav[r["key"]]) / SR if wav[r["key"]] is not None else 0))
    m = OmniCTC.__new__(OmniCTC)
    m.device, m.dtype = DEVICE, torch.float32
    m.proc = AutoProcessor.from_pretrained(a.model)
    m.model = load_fp16_upcast(a.model)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for budget in a.budgets:
        t0, padded_frac = time.time(), []
        preds = {}
        for k in range(0, len(order), 64):  # run_omni's groups of 64
            group = [r for r in order[k : k + 64] if wav[r["key"]] is not None]
            wavs = [wav[r["key"]] for r in group]
            pieces, target = plan_lengths(wavs, budget, _split)
            texts = {}
            for i, j, c in pieces:
                tl = target[(i, j)]
                x = np.pad(c, (0, tl - len(c))) if tl > len(c) else c
                padded_frac.append(1 - len(c) / tl)
                texts[(i, j)] = m._batch([x])[0]
            for i, r in enumerate(group):
                parts = [texts[(i, j)] for j in range(len(_split(wavs[i])))]
                t = " ".join(p.strip() for p in parts if p.strip())
                t = sub.nahuatl_orthography(t) if r["lang"] in sub.NAHUATL else t
                preds[r["key"]] = sub.clean_transcript(t, r["lang"])
        for r in rows:
            preds.setdefault(r["key"], sub.clean_transcript("", r["lang"]))
        secs = time.time() - t0
        with open(out / f"pred_b{budget:g}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["audio_filename", "transcript"])
            for r in rows:
                w.writerow([Path(r["audio"]).name, preds[r["key"]]])
        refs = [normalize(r["ref"]) for r in rows]
        hyp = [normalize(preds[r["key"]]) for r in rows]
        by = {}
        for r, x, y in zip(rows, refs, hyp):
            g = by.setdefault(r["lang"], ([], []))
            g[0].append(x), g[1].append(y)
        per = {k: round(jiwer.wer(x, y), 4) for k, (x, y) in sorted(by.items())}
        print(f"AB budget={budget:g}s: WER {jiwer.wer(refs, hyp):.4f} on {len(rows)} clips; by lang {per}; "
              f"mean padding {100 * np.mean(padded_frac):.1f}% (max {100 * np.max(padded_frac):.0f}%); {secs:.0f}s; "
              f"gpu peak {(torch.cuda.max_memory_allocated() if DEVICE == 'cuda' else 0) / 2**30:.2f} GiB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
