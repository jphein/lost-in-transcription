"""Parity checks before any training (GPU, ~2 min):
1. bottom_features(upto) + the trainer's Top(upto) == the full HF model (same logits, same text);
2. the CTC loss on a known target is finite;
3. speed of the frozen bottom, for the cache-time estimate.

    python train/parity.py --model DIR --clips DIR [--upto 52] [--n 4]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEVICE, SR, Vocab, bottom_features, decode_audio, load_fp16_upcast  # noqa: E402
from train import build_top  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--upto", type=int, default=52)
    ap.add_argument("--n", type=int, default=4)
    a = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(a.model)
    t0 = time.time()
    m = load_fp16_upcast(a.model)
    print(f"full model (fp16 storage, fp32 compute) loaded in {time.time() - t0:.0f}s; "
          f"gpu {(torch.cuda.memory_allocated() if DEVICE == 'cuda' else 0) / 2**30:.2f} GiB", flush=True)
    top, cfg = build_top(a.model, a.upto, DEVICE)
    top.eval()
    print(f"top ({cfg.num_hidden_layers - a.upto} layers) loaded", flush=True)
    vocab = Vocab(a.model)
    clips = sorted(Path(a.clips).glob("*.mp3"))[: a.n]
    worst, agree_all, secs, tb = 0.0, True, 0.0, 0.0
    for c in clips:
        wav = decode_audio(c)
        secs += len(wav) / SR
        with torch.inference_mode():
            x = proc(wav, sampling_rate=SR, return_tensors="pt").input_values.to(DEVICE, torch.float32)
            ref = m(x).logits[0].float()
            t1 = time.time()
            h = bottom_features(m, proc, wav, a.upto)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            tb += time.time() - t1
            mask = torch.ones(1, h.shape[0], dtype=torch.bool, device=DEVICE)
            got = top(h[None], mask)[0]
            # the cache stores fp16: check that rounding does not change the text either
            got16 = top(h.half().float()[None], mask)[0]
        d = (ref - got).abs().max().item()
        worst = max(worst, d)
        t_ref = proc.batch_decode([ref.argmax(-1).cpu().numpy()])[0]
        t_got = proc.batch_decode([got.argmax(-1).cpu().numpy()])[0]
        t_16 = proc.batch_decode([got16.argmax(-1).cpu().numpy()])[0]
        agree = t_ref == t_got
        agree_all &= agree
        ids, dropped = vocab.encode(t_ref)
        top.train()
        lg = top(h[None], mask)
        lp = F.log_softmax(lg, -1, dtype=torch.float32).transpose(0, 1)
        loss = F.ctc_loss(lp, torch.tensor(ids, device=DEVICE), torch.tensor([h.shape[0]], device=DEVICE),
                          torch.tensor([len(ids)], device=DEVICE), blank=cfg.pad_token_id, reduction="mean", zero_infinity=True)
        top.eval()
        print(f"{c.name}: {len(wav) / SR:.1f}s T={h.shape[0]} max|dlogit|={d:.2e} text_same={agree} "
              f"fp16cache_same={t_16 == t_ref} ctc_loss(own text, train mode)={loss.item():.3f} dropped={dropped}\n"
              f"   {t_ref[:120]!r}", flush=True)
    print(f"PARITY {'OK' if agree_all and worst < 1e-2 else 'FAIL'}: worst max|dlogit| {worst:.2e}; "
          f"bottom speed {secs / max(tb, 1e-9):.1f}x realtime; gpu peak {(torch.cuda.max_memory_allocated() if DEVICE == 'cuda' else 0) / 2**30:.2f} GiB",
          flush=True)
    return 0 if agree_all and worst < 1e-2 else 1


if __name__ == "__main__":
    sys.exit(main())
