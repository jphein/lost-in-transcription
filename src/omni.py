"""Omnilingual ASR CTC (wav2vec2, transformers port) batched greedy decoding.

Language-agnostic: no language token. Long clips are split into <=CHUNK_S
windows at the quietest point near each boundary.
"""

from pathlib import Path

import numpy as np
import torch

SR = 16000
CHUNK_S = 30.0


def _split(wav: np.ndarray) -> list[np.ndarray]:
    n = int(CHUNK_S * SR)
    if len(wav) <= n:
        return [wav]
    out, start = [], 0
    win = int(0.02 * SR)
    while len(wav) - start > n:
        lo, hi = start + int(0.7 * n), start + n
        seg = wav[lo:hi]
        e = np.convolve(seg**2, np.ones(win) / win, mode="same")
        cut = lo + int(np.argmin(e))
        out.append(wav[start:cut])
        start = cut
    out.append(wav[start:])
    return out


def _placement(device: str) -> dict:
    """Local-testing hook: LIT_GPU_MEM=9GiB splits a too-big model across GPU+CPU."""
    import os

    if device == "cuda" and os.environ.get("LIT_GPU_MEM"):
        return {"device_map": "auto", "max_memory": {0: os.environ["LIT_GPU_MEM"], "cpu": "24GiB"}}
    return {"device_map": device}


class BatchOOM(RuntimeError):
    """A batch of one chunk still ran out of GPU memory."""


def is_oom(e: BaseException) -> bool:
    return isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e).lower()


class OmniCTC:
    budget_s: float | None = None  # current seconds-of-audio per forward pass
    oom_retries: int = 0

    def __init__(self, model_dir: Path, device: str = "cuda"):
        from transformers import AutoProcessor, Wav2Vec2ForCTC

        self.device = device
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        self.proc = AutoProcessor.from_pretrained(str(model_dir))
        self.model = (
            Wav2Vec2ForCTC.from_pretrained(
                str(model_dir), dtype=self.dtype, low_cpu_mem_usage=True, **_placement(device)
            ).eval()
        )

    @torch.inference_mode()
    def _batch(self, chunks: list[np.ndarray]) -> list[str]:
        feats = self.proc(chunks, sampling_rate=SR, return_tensors="pt", padding=True)
        x = feats.input_values.to(self.device, self.dtype)
        mask = feats.get("attention_mask")
        kw = {"attention_mask": mask.to(self.device)} if mask is not None else {}
        logits = self.model(x, **kw).logits
        ids = logits.argmax(-1).cpu()
        return self.proc.batch_decode(ids)

    def _free(self) -> None:
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def transcribe_many(self, wavs: list[np.ndarray], max_batch_s: float = 240.0) -> list[str]:
        """wavs: 16 kHz float32 mono. Returns one string per input.

        On CUDA OOM the batch budget (seconds of audio per forward pass) is
        halved and the same chunks retried; the reduced budget sticks for later
        calls. A single chunk that still OOMs raises BatchOOM -- never "".
        """
        if self.budget_s is None or self.budget_s > max_batch_s:
            self.budget_s = max_batch_s
        pieces = []  # (owner, order, chunk)
        counts = []
        for i, w in enumerate(wavs):
            cs = _split(w)
            counts.append(len(cs))
            for j, c in enumerate(cs):
                pieces.append((i, j, c))
        pieces.sort(key=lambda p: -len(p[2]))
        texts: dict[tuple[int, int], str] = {}
        k = 0
        while k < len(pieces):
            longest = max(len(pieces[k][2]) / SR, 1.0)
            bs = max(1, int(self.budget_s // longest))
            group = pieces[k : k + bs]
            oom = False
            try:
                outs = self._batch([g[2] for g in group])
            except Exception as e:
                if not is_oom(e):
                    raise
                oom = True
            if oom:  # outside the except: the traceback no longer pins activations
                self._free()
                if len(group) == 1:
                    raise BatchOOM(f"single {longest:.0f}s chunk OOM at budget {self.budget_s:.0f}s") from None
                self.budget_s = max(longest, self.budget_s / 2)
                if int(self.budget_s // longest) >= bs:  # guarantee progress toward bs=1
                    self.budget_s = longest * (bs // 2)
                self.oom_retries += 1
                continue
            for (i, j, _), t in zip(group, outs):
                texts[(i, j)] = t
            k += len(group)
        out = []
        for i in range(len(wavs)):
            parts = [texts[(i, j)] for j in range(counts[i])]
            out.append(" ".join(p.strip() for p in parts if p.strip()))
        return out
