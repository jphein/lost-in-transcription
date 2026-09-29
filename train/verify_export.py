"""Check an export tensor by tensor with little memory: every tensor outside the fine-tuned set is
bit-identical to the base checkpoint, and every fine-tuned one equals the checkpoint's top cast to fp16.
    python train/verify_export.py --base DIR --export DIR --top PT --upto 52      (prints counts only)
"""
import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--export", required=True)
    ap.add_argument("--top", required=True)
    ap.add_argument("--upto", type=int, default=52)
    a = ap.parse_args()
    import torch
    from safetensors import safe_open

    top = torch.load(a.top, map_location="cpu", weights_only=True)
    if "top" in top and "opt" in top:
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
    wm_b = json.loads((Path(a.base) / "model.safetensors.index.json").read_text())["weight_map"]
    wm_e = json.loads((Path(a.export) / "model.safetensors.index.json").read_text())["weight_map"]
    assert wm_b == wm_e, "shard layout differs"
    same = tuned_ok = bad = 0
    for sh in sorted(set(wm_b.values())):
        with safe_open(str(Path(a.base) / sh), "pt") as fb, safe_open(str(Path(a.export) / sh), "pt") as fe:
            assert set(fb.keys()) == set(fe.keys())
            for k in fe.keys():
                te = fe.get_tensor(k)
                if k in ren:
                    tuned_ok += int(torch.equal(te, ren[k].to(torch.float16)))
                    bad += int(not torch.equal(te, ren[k].to(torch.float16)))
                else:
                    ok = torch.equal(te, fb.get_tensor(k))
                    same += int(ok)
                    bad += int(not ok)
    print(f"VERIFY {'OK' if bad == 0 and tuned_ok == len(ren) else 'FAIL'}: {same} untouched tensors identical, "
          f"{tuned_ok}/{len(ren)} fine-tuned tensors equal the checkpoint (fp16), {bad} mismatches")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
