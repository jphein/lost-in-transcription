"""Advance a feature cache from layer A to layer B by running the frozen layers A..B-1 once.

    python train/advance.py --model DIR --inp CACHE_A --a 44 --b 52 --out CACHE_B

Each piece runs alone (no padding), in fp32 from the fp16 checkpoint, exactly as cache.py would have
computed layer B directly (up to the fp16 rounding of the stored layer-A states). Resumable like cache.py.
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
from common import DEVICE  # noqa: E402

STOP = False


def _term(signum, frame):
    global STOP
    STOP = True
    print(f"SIGTERM at {time.strftime('%H:%M:%S')}: stopping after the current piece", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--inp", required=True)
    ap.add_argument("--a", type=int, required=True)
    ap.add_argument("--b", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    signal.signal(signal.SIGTERM, _term)

    import torch
    from safetensors import safe_open
    from transformers import Wav2Vec2Config
    from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2EncoderLayerStableLayerNorm

    cfg = Wav2Vec2Config.from_pretrained(a.model)
    cfg._attn_implementation = "sdpa"
    D = cfg.hidden_size
    with torch.device(DEVICE):  # fp16 storage, each layer upcast to fp32 only while it runs (small cards)
        layers = torch.nn.ModuleList([Wav2Vec2EncoderLayerStableLayerNorm(cfg) for _ in range(a.b - a.a)]).eval().half()

    def up(mod, args):
        for p in mod.parameters():
            p.data = p.data.float()

    def down(mod, args, out):
        for p in mod.parameters():
            p.data = p.data.half()

    for layer in layers:
        layer.register_forward_pre_hook(up)
        layer.register_forward_hook(down)
    idx = Path(a.model) / "model.safetensors.index.json"
    wm = json.loads(idx.read_text())["weight_map"] if idx.exists() else None
    sd = layers.state_dict()
    with torch.no_grad():
        for name, t in sd.items():
            i, rest = name.split(".", 1)
            src = f"wav2vec2.encoder.layers.{int(i) + a.a}.{rest}"
            with safe_open(str(Path(a.model) / (wm[src] if wm else "model.safetensors")), framework="pt", device="cpu") as f:
                t.copy_(f.get_tensor(src))

    inp, out = Path(a.inp), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in (inp / "index.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    done, off = set(), 0
    if (out / "index.jsonl").exists():
        for l in (out / "index.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            done.add(r["key"])
            off = max(off, r["off"] + r["n"])
    if (out / "feats.f16").exists():
        with open(out / "feats.f16", "r+b") as f:
            f.truncate(off * 2 * D)
    fin = open(inp / "feats.f16", "rb")
    t0, n_done = time.time(), 0
    with open(out / "feats.f16", "ab") as ff, open(out / "index.jsonl", "a", encoding="utf-8") as fi, torch.inference_mode():
        for r in rows:
            if STOP:
                break
            if r["key"] in done:
                continue
            fin.seek(r["off"] * D * 2)
            h = torch.from_numpy(np.fromfile(fin, dtype=np.float16, count=r["n"] * D).reshape(1, r["n"], D)).to(DEVICE).float()
            for layer in layers:
                h = layer(h, attention_mask=None)[0]
            ff.write(h[0].to(torch.float16).cpu().numpy().tobytes())
            ff.flush()
            fi.write(json.dumps(dict(r, off=off), ensure_ascii=False) + "\n")
            fi.flush()
            off += r["n"]
            n_done += 1
            if n_done % 500 == 0:
                print(f"{n_done} pieces in {time.time() - t0:.0f}s", flush=True)
    gpu = torch.cuda.max_memory_allocated() / 2**30 if DEVICE == "cuda" else 0
    print(f"ADVANCE_DONE {out} pieces={n_done} frames={off} peak_rss_gib="
          f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.2f} gpu_peak_gib={gpu:.2f} stopped={STOP}", flush=True)
    return 0 if not STOP else 75


if __name__ == "__main__":
    sys.exit(main())
