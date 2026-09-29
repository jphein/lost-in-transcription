"""Shared helpers for fine-tuning the top layers of Omnilingual CTC (the HF Wav2Vec2ForCTC port).

This runs on the training host, not in the competition image. The host is a Pascal P102-100: it has no
bf16 and runs fp16 at 1/64 of its fp32 rate, so all maths is fp32, and the frozen layers are kept in fp16
and upcast one layer at a time.
"""

import gc
import io
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))  # omni._split, nahuatl.to_reference
sys.path.insert(0, str(REPO / "runtime"))  # the official score.py

SR = 16000
DEVICE = os.environ.get("LIT_DEVICE", "cuda")  # "cpu" only for logic tests on a small model
HOP = 320  # product of the conv strides: 50 frames per second
NAHUATL = {"azz", "nhw", "nhi"}


# --- audio -----------------------------------------------------------------------------------------
# faster_whisper 1.2.1 audio.decode_audio (MIT), vendored so that hosts where ctranslate2 cannot be
# imported (no AVX) decode exactly as the submission does.
def decode_audio(input_file, sampling_rate: int = SR) -> np.ndarray:
    import av

    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=sampling_rate)
    raw_buffer = io.BytesIO()
    dtype = None
    with av.open(str(input_file), mode="r", metadata_errors="ignore") as container:
        frames = container.decode(audio=0)
        frames = _ignore_invalid_frames(frames)
        frames = _group_frames(frames, 500000)
        frames = _resample_frames(frames, resampler)
        for frame in frames:
            array = frame.to_ndarray()
            dtype = array.dtype
            raw_buffer.write(array)
    del resampler
    gc.collect()
    audio = np.frombuffer(raw_buffer.getbuffer(), dtype=dtype)
    return audio.astype(np.float32) / 32768.0


def _ignore_invalid_frames(frames):
    import av

    iterator = iter(frames)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            break
        except av.error.InvalidDataError:
            continue


def _group_frames(frames, num_samples=None):
    import av

    fifo = av.audio.fifo.AudioFifo()
    for frame in frames:
        frame.pts = None
        fifo.write(frame)
        if num_samples is not None and fifo.samples >= num_samples:
            yield fifo.read()
    if fifo.samples > 0:
        yield fifo.read()


def _resample_frames(frames, resampler):
    for frame in itertools.chain(frames, [None]):
        yield from resampler.resample(frame)


# --- text ------------------------------------------------------------------------------------------
def normalize(text: str) -> str:
    """The official scorer's normalisation (runtime/score.py), stripped."""
    from score import normalize_text

    return normalize_text(text or "").strip()


class Vocab:
    """Character-level CTC vocab of the Omni port. The tokenizer's word delimiter '▁' is not in
    vocab.json; the model writes word breaks as ' ' (id 4), so targets are encoded by hand."""

    def __init__(self, model_dir):
        self.tok = json.loads((Path(model_dir) / "vocab.json").read_text(encoding="utf-8"))
        self.space = self.tok[" "]

    def encode(self, text: str) -> tuple[list[int], int]:
        ids, dropped = [], 0
        for ch in text:
            i = self.tok.get(ch)
            if i is None or (i < 4):
                dropped += 1
                continue
            ids.append(i)
        # no leading/trailing/double spaces
        out = []
        for i in ids:
            if i == self.space and (not out or out[-1] == self.space):
                continue
            out.append(i)
        while out and out[-1] == self.space:
            out.pop()
        return out, dropped


def n_frames(n_samples: int) -> int:
    """Output length of the wav2vec2 conv stack (kernels 10,3,3,3,3,2,2; strides 5,2,2,2,2,2,2)."""
    L = n_samples
    for k, s in zip((10, 3, 3, 3, 3, 2, 2), (5, 2, 2, 2, 2, 2, 2)):
        L = (L - k) // s + 1
    return max(L, 0)


# --- model -----------------------------------------------------------------------------------------
def load_fp16_upcast(model_dir, device=None, attn="sdpa", keep_layers=None):
    """Full model with fp16 storage and fp32 compute: the small modules become fp32 for good, and each
    transformer layer is cast to fp32 just before it runs and back to fp16 afterwards (≤ 0.2 GB extra)."""
    import torch
    from transformers import Wav2Vec2ForCTC

    device = device or DEVICE
    kw = {"device_map": device} if device != "cpu" else {}  # device_map streams shards straight to the GPU
    if keep_layers is not None:  # build only the frozen bottom, so the rest never reaches the card
        kw["num_hidden_layers"] = keep_layers
    m = Wav2Vec2ForCTC.from_pretrained(str(model_dir), dtype=torch.float16, attn_implementation=attn, **kw).eval()
    w = m.wav2vec2
    assert keep_layers is None or len(w.encoder.layers) == keep_layers
    for mod in (w.feature_extractor, w.feature_projection, w.encoder.pos_conv_embed, w.encoder.layer_norm, m.lm_head):
        mod.float()

    def up(mod, args):
        for p in mod.parameters():
            p.data = p.data.float()

    def down(mod, args, out):
        for p in mod.parameters():
            p.data = p.data.half()

    for layer in w.encoder.layers:
        layer.register_forward_pre_hook(up)
        layer.register_forward_hook(down)
    return m


def bottom_features(m, proc, wav: np.ndarray, upto: int):
    """Hidden states entering layer `upto` (fp32, shape (T, D)), exactly as Wav2Vec2ForCTC computes them
    in eval mode with no attention mask (what omni.OmniCTC does for a batch of one)."""
    import torch

    x = proc(wav, sampling_rate=SR, return_tensors="pt").input_values.to(DEVICE, torch.float32)
    w = m.wav2vec2
    f = w.feature_extractor(x).transpose(1, 2)
    h, _ = w.feature_projection(f)
    enc = w.encoder
    h = h + enc.pos_conv_embed(h)
    for layer in enc.layers[:upto]:
        h = layer(h, attention_mask=None)[0]
    return h[0]


def read_manifest(path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def write_manifest(path, rows) -> None:
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
