"""Merge fine-tuned top layers into a copy of the fp16 Omni checkpoint (same shards, same names).

    python train/export.py --model SRC_DIR --top RUN/best.pt --upto 52 --out DST_DIR [--nah-ortho 0|1]

Every tensor the trainer did not touch is copied bit for bit; the fine-tuned ones are cast to fp16.
Writes DST_DIR/lit.json (provenance + the nahj switch read by src/main.py) and prints what changed.
Needs RAM for the largest shard (~5 GB for the 3B).
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--top", required=True)
    ap.add_argument("--upto", type=int, default=52)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nah-ortho", type=int, choices=[0, 1], default=1)
    ap.add_argument("--omni-batch-s", type=float, default=None,
                    help="write lit.json omni_batch_s (omit: the stock batching, FINETUNE.md 11:05)")
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    import torch
    from safetensors.torch import load_file, save_file

    src, dst = Path(a.model), Path(a.out)
    dst.mkdir(parents=True, exist_ok=False)
    top = torch.load(a.top, map_location="cpu", weights_only=True)
    if isinstance(top, dict) and "top" in top and "opt" in top:  # a last.pt (F ships its final epoch)
        top = top["top"]
    ren = {}
    for k, v in top.items():
        if k.startswith("layers."):
            i, rest = k.split(".", 2)[1:]
            ren[f"wav2vec2.encoder.layers.{int(i) + a.upto}.{rest}"] = v
        elif k.startswith("layer_norm."):
            ren["wav2vec2.encoder." + k] = v
        else:
            ren[k] = v
    for f in src.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors") and f.name != "lit.json":
            shutil.copy2(f, dst / f.name)
    idx = src / "model.safetensors.index.json"
    shards = sorted(set(json.loads(idx.read_text())["weight_map"].values())) if idx.exists() else ["model.safetensors"]
    used, changed = set(), 0
    for sh in shards:
        t = load_file(str(src / sh))
        for k in list(t):
            if k in ren:
                new = ren[k].to(torch.float16)
                assert new.shape == t[k].shape, (k, new.shape, t[k].shape)
                changed += int(not torch.equal(new, t[k]))
                t[k] = new.contiguous()
                used.add(k)
        save_file(t, str(dst / sh), metadata={"format": "pt"})
        del t
    unused = set(ren) - used
    assert not unused, f"fine-tuned tensors not found in the checkpoint: {sorted(unused)[:5]}"
    meta = {"base": src.name, "finetuned_from_layer": a.upto, "tensors_replaced": len(used), "tensors_changed": changed,
            "nah_ortho": bool(a.nah_ortho), "note": a.note}
    if a.omni_batch_s is not None:
        meta["omni_batch_s"] = a.omni_batch_s
    (dst / "lit.json").write_text(json.dumps(meta, indent=1) + "\n")
    for sh in shards:  # streamed: read_bytes() of a 5 GB shard blew a 6G cap (familiar, 13:39:53)
        h = hashlib.sha256()
        with open(dst / sh, "rb") as f:
            for chunk in iter(lambda: f.read(16 << 20), b""):
                h.update(chunk)
        print(f"{sh} sha256 {h.hexdigest()}")
    print(f"EXPORT_DONE {dst}: {json.dumps(meta)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
