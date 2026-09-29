"""Batched vs batch-1 inference through the submission's own code (omni.OmniCTC.transcribe_many and
main.run_omni's grouping), with fp32 compute from the fp16 weights (Pascal hosts), so the only thing
that changes between the arms is the batch budget.

    python train/batch_ab.py --model DIR --manifest dev.jsonl --out DIR [--budgets 240 1]

Writes OUT/pred_b<budget>.csv (audio_filename, transcript) per arm and prints corpus WER overall and
per language with the official normalisation.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEVICE, SR, decode_audio, load_fp16_upcast, normalize, read_manifest  # noqa: E402


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
    from omni import OmniCTC

    rows = read_manifest(a.manifest)
    wav = {}
    for r in rows:
        try:
            wav[r["key"]] = decode_audio(r["audio"])
        except Exception:
            wav[r["key"]] = None
    # main() sorts the manifest by file_duration_seconds (integer seconds), longest first
    order = sorted(rows, key=lambda r: -round(len(wav[r["key"]]) / SR if wav[r["key"]] is not None else 0))
    m = OmniCTC.__new__(OmniCTC)
    m.device, m.dtype = DEVICE, torch.float32
    m.proc = AutoProcessor.from_pretrained(a.model)
    m.model = load_fp16_upcast(a.model)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for budget in a.budgets:
        m.budget_s, m.oom_retries = None, 0
        t0 = time.time()
        preds = {}
        for k in range(0, len(order), 64):  # run_omni's groups of 64
            group = order[k : k + 64]
            ok = [r for r in group if wav[r["key"]] is not None]
            texts = m.transcribe_many([wav[r["key"]] for r in ok], max_batch_s=budget)
            for r, t in zip(ok, texts):
                t = sub.nahuatl_orthography(t) if r["lang"] in sub.NAHUATL else t
                preds[r["key"]] = sub.clean_transcript(t, r["lang"])
            for r in group:
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
              f"{secs:.0f}s; oom_retries={m.oom_retries}; gpu peak "
              f"{(torch.cuda.max_memory_allocated() if DEVICE == 'cuda' else 0) / 2**30:.2f} GiB",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
