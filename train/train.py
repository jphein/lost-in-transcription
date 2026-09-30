"""Fine-tune the top layers of Omni CTC (+ final LayerNorm + lm_head) on cached layer-UPTO features.

    python train/train.py --model DIR --upto 52 --train CACHE[:REPEAT] ... --val CACHE --out RUN
        [--val2 CACHE] [--epochs 12] [--eval-only]

Every epoch: CTC training, then corpus WER on --val with the official score.py normalisation (raw and,
for Nahuatl clips, after the repo's orthography map nahuatl.to_reference). Epoch 0 is the untouched
model, i.e. the baseline. Writes RUN/log.jsonl, RUN/best.pt (top weights only, fp32, lowest val WER),
RUN/last.pt (weights + optimiser + position, for --resume). SIGTERM saves last.pt after the current
optimiser step and exits 75.
"""

import argparse
import json
import math
import os
import random
import resource
import signal
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEVICE, NAHUATL, REPO, normalize  # noqa: E402

STOP = False
# fragmentation-proof allocator (mz-C-jv OOMed at 13:31:30 with 1.93 GiB reserved but unallocated)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _term(signum, frame):
    global STOP
    STOP = True
    print(f"SIGTERM at {time.strftime('%H:%M:%S')}: saving after the current step", flush=True)


class Cache:
    """One cache dir from cache.py. Pieces are read from disk on demand (np.fromfile), so RSS stays small."""

    def __init__(self, path, D, vocab=None):
        self.path = Path(path)
        self.D = D
        self.rows = [json.loads(l) for l in (self.path / "index.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        if vocab is not None:  # training caches are built without --chunk: one piece per clip
            assert all(r["part"] == 0 for r in self.rows), f"{path}: training cache must not be chunked"
        self.f = open(self.path / "feats.f16", "rb")
        self.dropped = 0
        if vocab is not None:
            keep = []
            for r in self.rows:
                ids, d = vocab.encode(r["text"])
                self.dropped += d
                # CTC needs T >= len(target) + number of repeated neighbours
                reps = sum(1 for a, b in zip(ids, ids[1:]) if a == b)
                if ids and r["n"] >= len(ids) + reps:
                    r["ids"] = ids
                    keep.append(r)
            self.skipped = len(self.rows) - len(keep)
            self.rows = keep

    def feats(self, r):
        # np.fromfile's offset is relative to the file position, so seek to the absolute offset first
        self.f.seek(r["off"] * self.D * 2)
        a = np.fromfile(self.f, dtype=np.float16, count=r["n"] * self.D)
        assert a.size == r["n"] * self.D, (self.path, r["key"], a.size)
        return a.reshape(r["n"], self.D)


def build_top(model_dir, upto, device):
    import torch
    from safetensors import safe_open
    from torch import nn
    from transformers import Wav2Vec2Config
    from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2EncoderLayerStableLayerNorm

    cfg = Wav2Vec2Config.from_pretrained(model_dir)
    cfg._attn_implementation = "sdpa"
    n_top = cfg.num_hidden_layers - upto

    class Top(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Wav2Vec2EncoderLayerStableLayerNorm(cfg) for _ in range(n_top)])
            self.layer_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
            self.dropout = nn.Dropout(cfg.final_dropout)
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size)
            self.layerdrop = cfg.layerdrop
            self.ckpt = True

        def forward(self, h, mask):
            from torch.utils.checkpoint import checkpoint
            from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa

            h = h.masked_fill(~mask[..., None], 0.0)
            am = _prepare_4d_attention_mask_for_sdpa(mask.long(), h.dtype)
            for layer in self.layers:
                if self.training and torch.rand([]).item() < self.layerdrop:
                    continue
                if self.training and self.ckpt:
                    h = checkpoint(lambda x, l=layer: l(x, attention_mask=am)[0], h, use_reentrant=False)
                else:
                    h = layer(h, attention_mask=am)[0]
            h = self.layer_norm(h)
            return self.lm_head(self.dropout(h))

    with torch.device(device):  # build on the card directly: no CPU copy of the 1.7 GB fp32 top
        top = Top()
    # load exactly the tensors we train from the (sharded) fp16 checkpoint, as fp32, one at a time
    idx = Path(model_dir) / "model.safetensors.index.json"
    files = json.loads(idx.read_text())["weight_map"] if idx.exists() else None
    want = {}
    for name in top.state_dict():
        if name.startswith("layers."):
            i, rest = name.split(".", 2)[1:]
            src = f"wav2vec2.encoder.layers.{int(i) + upto}.{rest}"
        elif name.startswith("layer_norm."):
            src = "wav2vec2.encoder." + name
        else:
            src = name
        want[name] = src
    by_file = {}
    for dst, src in want.items():
        fn = files[src] if files else "model.safetensors"
        by_file.setdefault(fn, []).append((dst, src))
    dst_sd = top.state_dict()  # tensors share storage with the parameters
    loaded = set()
    with torch.no_grad():
        for fn, pairs in by_file.items():
            with safe_open(str(Path(model_dir) / fn), framework="pt", device="cpu") as f:
                for dst, src in pairs:
                    t = f.get_tensor(src)
                    assert t.shape == dst_sd[dst].shape, (dst, t.shape, dst_sd[dst].shape)
                    dst_sd[dst].copy_(t.float())
                    loaded.add(dst)
                    del t
    missing = set(dst_sd) - loaded
    assert not missing, f"top tensors not in the checkpoint: {sorted(missing)[:5]}"
    return top, cfg


def time_mask(h, p=0.05, span=10, rng=None):
    """HF-style time masking on (T, D) features; masked frames become 0."""
    T = h.shape[0]
    n = int(p * T / span + rng.random())
    for _ in range(n):
        s = rng.randrange(0, max(1, T - span))
        h[s : s + span] = 0
    return h


def make_batches(items, max_frames, rng):
    """items: list of (cache, row). Sorted by length, cut into batches of ≤ max_frames padded frames."""
    items = sorted(items, key=lambda it: it[1]["n"])
    batches, cur, longest = [], [], 0
    for it in items:
        n = it[1]["n"]
        if cur and max(longest, n) * (len(cur) + 1) > max_frames:
            batches.append(cur)
            cur, longest = [], 0
        cur.append(it)
        longest = max(longest, n)
    if cur:
        batches.append(cur)
    rng.shuffle(batches)
    return batches


def collate(batch, device, train, rng):
    import torch

    T = max(r["n"] for _, r in batch)
    D = batch[0][0].D
    x = np.zeros((len(batch), T, D), dtype=np.float32)
    mask = np.zeros((len(batch), T), dtype=bool)
    for b, (c, r) in enumerate(batch):
        f = c.feats(r).astype(np.float32)
        if train:
            f = time_mask(f, rng=rng)
        x[b, : r["n"]] = f
        mask[b, : r["n"]] = True
    x = torch.from_numpy(x).to(device)
    mask = torch.from_numpy(mask).to(device)
    if not train:
        return x, mask, None, None
    tg = torch.tensor([i for _, r in batch for i in r["ids"]], dtype=torch.long)
    tl = torch.tensor([len(r["ids"]) for _, r in batch], dtype=torch.long)
    return x, mask, tg, tl


def evaluate(top, cache, proc, device, max_frames=12000):
    """Corpus WER on a chunked validation cache, the way the submission joins chunk transcripts."""
    import jiwer
    import torch

    from main import clean_transcript
    from nahuatl import to_reference

    top.eval()
    pieces = {}
    items = sorted(((cache, r) for r in cache.rows), key=lambda it: it[1]["n"])
    batches, cur, longest = [], [], 0
    for it in items:
        n = it[1]["n"]
        if cur and max(longest, n) * (len(cur) + 1) > max_frames:
            batches.append(cur)
            cur, longest = [], 0
        cur.append(it)
        longest = max(longest, n)
    if cur:
        batches.append(cur)
    with torch.inference_mode():
        for b in batches:
            x, mask, _, _ = collate(b, device, False, None)
            ids = top(x, mask).argmax(-1).cpu().numpy()
            texts = proc.batch_decode([ids[k, : r["n"]] for k, (_, r) in enumerate(b)])
            for (_, r), t in zip(b, texts):
                pieces[(r["clip"], r["part"])] = t
    clips = {}
    for (clip, part), t in sorted(pieces.items()):
        clips.setdefault(clip, []).append(t)
    meta = {r["clip"]: r for r in cache.rows}
    import main as sub

    refs, raw, mapped, seg, post, langs = [], [], [], [], [], []
    for clip, parts in clips.items():
        r = meta[clip]
        t = " ".join(p.strip() for p in parts if p.strip())
        code = r.get("lang", "")
        langs.append(code)
        refs.append(normalize(r["ref"]))
        raw.append(normalize(clean_transcript(t, code)))
        m = to_reference(t) if code in NAHUATL else t
        mapped.append(normalize(clean_transcript(m, code)))
        # luna-refurb's post-processing (main.reference_style): post = map + join fix / Indonesian rules;
        # seg = join fix without the map (what a fine-tuned model with lit.json nah_ortho=false ships)
        post.append(normalize(clean_transcript(sub.reference_style(t, code), code)))
        if code in NAHUATL:
            from segment import fix_boundaries

            seg.append(normalize(clean_transcript(fix_boundaries(t), code)))
        else:
            seg.append(post[-1])
    top.train()
    out = {"wer_raw": jiwer.wer(refs, raw), "wer_map": jiwer.wer(refs, mapped), "wer_seg": jiwer.wer(refs, seg),
           "wer_post": jiwer.wer(refs, post), "n_clips": len(refs)}
    by = {}
    for code, ref_, raw_, m_ in zip(langs, refs, raw, mapped):
        g = by.setdefault(code or "?", ([], [], []))
        g[0].append(ref_), g[1].append(raw_), g[2].append(m_)
    out["by_lang"] = {k: {"raw": round(jiwer.wer(a, b), 4), "map": round(jiwer.wer(a, c), 4), "n": len(a)}
                      for k, (a, b, c) in by.items()}
    return out, {c: {"raw": x, "map": y} for c, x, y in zip(clips, raw, mapped)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--upto", type=int, default=52)
    ap.add_argument("--train", nargs="*", default=[])
    ap.add_argument("--val", required=True)
    ap.add_argument("--val2", default="")
    ap.add_argument("--val-keys", default="", help="restrict --val to the clips of this manifest (LOCO folds)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.10)
    ap.add_argument("--micro-frames", type=int, default=6000)  # 120 s
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--init", default="", help="load the top's weights from this .pt (best.pt) before anything else")
    ap.add_argument("--seg-vocab", default="", help="segment.load() this vocab first (e.g. a V-excluded nah_vocab)")
    ap.add_argument("--stop-after", type=int, default=0,
                    help="keep the --epochs LR schedule but stop after this epoch (F = C's recipe up to C's pick)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N optimiser steps (smoke tests)")
    ap.add_argument("--amp", choices=["none", "bf16"], default="none",
                    help="bf16 autocast for the trainable forward (B60/XPU); master weights and Adam stay fp32")
    a = ap.parse_args()
    signal.signal(signal.SIGTERM, _term)

    import torch
    import torch.nn.functional as F
    from transformers import AutoProcessor

    from common import Vocab

    sys.path.insert(0, str(REPO / "src"))
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.jsonl", "a", encoding="utf-8")

    def log(**kw):
        kw["t"] = time.strftime("%Y-%m-%d %H:%M:%S")
        kw["rss_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
        if device == "cuda":
            kw["gpu_peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        elif device == "xpu":
            kw["gpu_peak_gib"] = round(torch.xpu.max_memory_allocated() / 2**30, 2)
        print(json.dumps(kw, ensure_ascii=False), flush=True)
        logf.write(json.dumps(kw, ensure_ascii=False) + "\n")
        logf.flush()

    device = DEVICE
    if a.seg_vocab:
        import segment

        n = segment.load(a.seg_vocab)
        log(event="seg_vocab", path=a.seg_vocab, loaded=n)
    D = json.loads((Path(a.model) / "config.json").read_text())["hidden_size"]
    vocab = Vocab(a.model)
    proc = AutoProcessor.from_pretrained(a.model)
    top, cfg = build_top(a.model, a.upto, device)
    if a.init:
        top.load_state_dict(torch.load(a.init, map_location=device, weights_only=True))
        log(event="init", path=a.init)
    val = Cache(a.val, D)
    if a.val_keys:
        want = {json.loads(l)["key"] for l in Path(a.val_keys).read_text(encoding="utf-8").splitlines() if l.strip()}
        val.rows = [r for r in val.rows if r["clip"] in want]
        log(event="val_keys", path=a.val_keys, pieces=len(val.rows))
    val2 = Cache(a.val2, D) if a.val2 else None

    def run_eval(epoch, tag=""):
        res, preds = evaluate(top, val, proc, device)
        extra = {}
        if val2 is not None:
            r2, _ = evaluate(top, val2, proc, device)
            extra = {"val2_wer_raw": round(r2["wer_raw"], 4), "val2_wer_map": round(r2["wer_map"], 4)}
        log(event="eval", epoch=epoch, tag=tag, wer_raw=round(res["wer_raw"], 4), wer_map=round(res["wer_map"], 4),
            wer_seg=round(res["wer_seg"], 4), wer_post=round(res["wer_post"], 4),
            by_lang=res["by_lang"], n_clips=res["n_clips"], **extra)
        (out / f"val_pred_ep{epoch:02d}.json").write_text(json.dumps(preds, ensure_ascii=False, indent=0), encoding="utf-8")
        return min(res["wer_raw"], res["wer_map"])

    if a.eval_only:
        run_eval(0, "eval-only")
        return 0

    caches, items = [], []
    for spec in a.train:  # PATH[:REPEAT[:KEYS.jsonl]]: KEYS restricts the cache to that manifest's clips
        path, rep, keys = (spec.split(":") + ["", ""])[:3]
        c = Cache(path, D, vocab=vocab)
        if keys:
            want = {json.loads(l)["key"] for l in Path(keys).read_text(encoding="utf-8").splitlines() if l.strip()}
            c.rows = [r for r in c.rows if r["clip"] in want]
        caches.append(c)
        items += [(c, r) for r in c.rows] * int(rep or 1)
        log(event="train_cache", path=path, keys=keys, repeat=int(rep or 1), pieces=len(c.rows), skipped=c.skipped,
            dropped_chars=c.dropped, audio_s=round(sum(r["n"] for r in c.rows) / 50))

    head = [p for n, p in top.named_parameters() if n.startswith(("lm_head", "layer_norm"))]
    body = [p for n, p in top.named_parameters() if not n.startswith(("lm_head", "layer_norm"))]

    def groups(ps, lr):
        dec = [p for p in ps if p.ndim >= 2]
        nodec = [p for p in ps if p.ndim < 2]
        return [{"params": dec, "lr": lr, "weight_decay": a.wd}, {"params": nodec, "lr": lr, "weight_decay": 0.0}]

    opt = torch.optim.AdamW(groups(body, a.lr) + groups(head, a.lr_head), betas=(0.9, 0.98), eps=1e-8, foreach=False)
    batches0 = make_batches(items, a.micro_frames, random.Random(a.seed))
    steps_per_epoch = math.ceil(len(batches0) / a.accum)
    total = steps_per_epoch * a.epochs
    warm = max(1, int(a.warmup * total))
    def lr_at(s):  # linear warm-up, then linear decay to 0
        if s < warm:
            return (s + 1) / warm
        return max(0.0, (total - s) / max(1, total - warm))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    log(event="setup", amp=a.amp, upto=a.upto, micro_batches=len(batches0), steps_per_epoch=steps_per_epoch, total_steps=total, warmup=warm,
        trainable_m=round(sum(p.numel() for p in top.parameters()) / 1e6, 1), epochs=a.epochs)

    start_epoch, skip_micro, step, best = 1, 0, 0, float("inf")
    if a.resume and (out / "last.pt").exists():
        st = torch.load(out / "last.pt", map_location=device, weights_only=True)
        top.load_state_dict(st["top"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start_epoch, skip_micro, step, best = st["epoch"], st["micro"], st["step"], st["best"]
        rng.setstate(st["rng"])
        log(event="resume", epoch=start_epoch, micro=skip_micro, step=step, best=best)
    else:
        best = run_eval(0, "baseline")

    def save_last(epoch, micro):
        tmp = out / "last.pt.tmp"
        torch.save({"top": top.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": epoch,
                    "micro": micro, "step": step, "best": best, "rng": rng.getstate()}, tmp)
        tmp.replace(out / "last.pt")

    top.train()
    for epoch in range(start_epoch, a.epochs + 1):
        batches = make_batches(items, a.micro_frames, random.Random(a.seed * 1000 + epoch))
        t0, tot, cnt = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        for k, b in enumerate(batches):
            if k < skip_micro:
                continue
            x, mask, tg, tl = collate(b, device, True, rng)
            if a.amp == "bf16":
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits = top(x, mask)
            else:
                logits = top(x, mask)
            lp = F.log_softmax(logits.float(), dim=-1, dtype=torch.float32).transpose(0, 1)
            il = mask.sum(-1)
            loss = F.ctc_loss(lp, tg.to(device), il, tl.to(device), blank=cfg.pad_token_id, reduction="mean",
                              zero_infinity=True)
            (loss / a.accum).backward()
            tot += loss.item()
            cnt += 1
            if (k + 1) % a.accum == 0 or k + 1 == len(batches):
                torch.nn.utils.clip_grad_norm_(top.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 20 == 0:
                    log(event="step", epoch=epoch, step=step, loss=round(tot / max(cnt, 1), 4),
                        lr=sched.get_last_lr()[0], s_per_step=round((time.time() - t0) / max(1, (k + 1 - skip_micro) / a.accum), 2))
                if STOP:
                    save_last(epoch, k + 1)
                    log(event="sigterm_saved", epoch=epoch, micro=k + 1, step=step)
                    return 75
                if a.max_steps and step >= a.max_steps:
                    log(event="max_steps", step=step)
                    run_eval(epoch, "max-steps")
                    return 0
        skip_micro = 0
        log(event="epoch", epoch=epoch, loss=round(tot / max(cnt, 1), 4), secs=round(time.time() - t0))
        w = run_eval(epoch)
        if w < best:
            best = w
            torch.save(top.state_dict(), out / "best.pt.tmp")
            (out / "best.pt.tmp").replace(out / "best.pt")
            log(event="best", epoch=epoch, wer=round(w, 4))
        save_last(epoch + 1, 0)
        if a.stop_after and epoch >= a.stop_after:
            log(event="stop_after", epoch=epoch)
            break
    log(event="TRAIN_DONE", best=round(best, 4))
    return 0


if __name__ == "__main__":
    sys.exit(main())
