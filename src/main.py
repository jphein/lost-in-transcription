"""Lost in Transcription submission: faster-whisper large-v3 + Omnilingual ASR CTC, routed per clip.

One zip serves all three tracks; the track is inferred from the manifest's
`language` column. Logs only aggregate progress (no test data).
"""

import csv
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

T0 = time.time()
# keep transformers from importing TensorFlow/JAX (slow, and competes for GPU memory)
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_JAX", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
TIME_BUDGET_S = float(os.environ.get("LIT_TIME_BUDGET_S", 105 * 60))  # hard limit is 120 min

ROOT = Path.cwd()  # /code_execution
DATA_DIR = ROOT / "data"
CLIPS_DIR = DATA_DIR / "clips"
MANIFEST = DATA_DIR / "test_metadata.csv"
OUT = ROOT / "submission" / "submission.csv"
SRC = Path(__file__).resolve().parent
MODELS = SRC / "models"
WHISPER_DIR = MODELS / os.environ.get("LIT_WHISPER", "faster-whisper-large-v3")


def _pick_omni() -> Path:
    if os.environ.get("LIT_OMNI"):
        return MODELS / os.environ["LIT_OMNI"]
    for name in ("omni-7b-v2-fp16", "omni-3b-v2-fp16", "omni-1b-v2-fp16"):
        if (MODELS / name).exists():
            return MODELS / name
    return MODELS / "omni-missing"


OMNI_DIR = _pick_omni()
QWEN_DIR = MODELS / os.environ.get("LIT_QWEN", "qwen3-asr-1.7b")
HAVE_W, HAVE_O, HAVE_Q = WHISPER_DIR.exists(), OMNI_DIR.exists(), QWEN_DIR.exists()
# Qwen3-ASR language names (no Javanese / Nahuatl support)
QWEN_LANG = {"spa": "Spanish", "eng": "English", "enspa": None, "ind": "Indonesian"}

# language code -> backend, based on what the zip ships. Whisper wins on
# Spanish/English (casing, punctuation); Omnilingual CTC wins on Javanese
# (FLEURS jv proxy: 25.7% vs 70% WER) and is the only model with Nahuatl.
BACKENDS = {}
for c in ["spa", "enspa", "eng"]:
    BACKENDS[c] = "whisper" if HAVE_W else ("qwen" if HAVE_Q else "omni")
for c in ["ind", "jav", "javind", "azz", "nhw", "nhi"]:
    BACKENDS[c] = "omni" if HAVE_O else "whisper"
for kv in filter(None, os.environ.get("LIT_BACKENDS", "").split(",")):
    k, v = kv.split("=")
    BACKENDS[k] = v

# manifest language code -> whisper language code
LANG_MAP = {
    "spa": "es", "enspa": None, "eng": "en",  # enspa: let whisper pick per chunk
    "ind": "id", "jav": "jw", "javind": "id",
    "azz": "es", "nhw": "es", "nhi": "es",  # no Nahuatl in whisper; es phonetics closest
}
NAHUATL = {"azz", "nhw", "nhi"} if os.environ.get("LIT_NAH_ORTHO", "1") == "1" else set()

# Short vocabulary priming per track (initial_prompt). Kept tiny: long prompts
# make whisper hallucinate on short clips.
PROMPTS = {
    "enspa": "So, like, yo le dije que I was going, pero she didn't want to, ¿sabes?",
    "spa": None,
    "javind": None, "jav": None, "ind": None,
    "azz": None, "nhw": None, "nhi": None,
}


DEBUG = os.environ.get("LIT_DEBUG") == "1"  # local only: error text may echo inputs

for _code in list(PROMPTS):
    if f"LIT_PROMPT_{_code}" in os.environ:
        PROMPTS[_code] = os.environ[f"LIT_PROMPT_{_code}"] or None
for _code in list(LANG_MAP):
    if f"LIT_LANG_{_code}" in os.environ:
        LANG_MAP[_code] = os.environ[f"LIT_LANG_{_code}"] or None


# Every blank transcript is counted here with its cause; nothing is blanked silently.
FAILS: Counter = Counter()
# Exit non-zero when more than this fraction of clips failed in the model (not
# unreadable audio): a failed job is free to retry, a scored blank run is not.
MAX_FAIL_FRAC = float(os.environ.get("LIT_MAX_FAIL_FRAC", 0.02))

# The platform's validator rejects a submission with any empty transcript cell
# (sp-en, 2026-09-29: 592 rows, one empty -> "not a valid submission", no score).
# So an empty whisper result is retried once without VAD, and anything still
# empty is written as one short, common word for its language: at most one
# insertion, and always a valid file. Counted in RESCUE and logged, never silent.
RESCUE: Counter = Counter()
RETRY_MAX_NO_SPEECH = float(os.environ.get("LIT_RETRY_NSP", 0.7))
FALLBACK_WORD = {
    "enspa": "yeah", "eng": "yeah", "spa": "sí",
    "ind": "iya", "jav": "iya", "javind": "iya",
    "azz": "kemaj", "nhw": "kemaj", "nhi": "kemaj",
}
# pandas.read_csv turns these exact cell values into NaN, which a validator
# treats as empty; a trailing period keeps them text (the scorer drops periods).
_PANDAS_NA = {"#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND",
              "1.#QNAN", "<NA>", "N/A", "NA", "NULL", "NaN", "None", "n/a", "nan", "null"}
_CTRL = re.compile(r"[\x00-\x1f\x7f]")


def clean_transcript(text, code: str) -> str:
    """One line of UTF-8 text: never empty, never a value pandas reads as NaN."""
    t = " ".join(_CTRL.sub(" ", text or "").split())
    if not t:
        RESCUE["placeholder"] += 1
        t = FALLBACK_WORD.get(code, "yeah")
    if t in _PANDAS_NA:
        t += "."
    return t


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def nahuatl_orthography(text: str) -> str:
    from nahuatl import to_reference

    return to_reference(text)


def run_whisper(rows, results):
    import torch
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute = "float16" if device == "cuda" else "int8"
    model = WhisperModel(str(WHISPER_DIR), device=device, compute_type=compute)
    pipe = BatchedInferencePipeline(model=model)
    log(f"whisper loaded on {device} ({compute})")
    beam = int(os.environ.get("LIT_BEAM", 5))
    batch = int(os.environ.get("LIT_BATCH", 16))
    for i, r in enumerate(rows):
        name, code = r["audio_filename"], (r.get("language") or "").strip()
        if time.time() - T0 > TIME_BUDGET_S:
            results[name] = ""
            FAILS["time_budget"] += 1
            continue
        try:
            segs, _info = pipe.transcribe(
                str(CLIPS_DIR / name),
                language=LANG_MAP.get(code, None),
                task="transcribe",
                beam_size=beam,
                batch_size=batch,
                initial_prompt=PROMPTS.get(code),
                condition_on_previous_text=False,
                without_timestamps=True,
                vad_filter=True,
            )
            text = " ".join(s.text.strip() for s in segs).strip()
            results[name] = nahuatl_orthography(text) if code in NAHUATL else text
        except Exception as e:  # never let one clip kill the run
            results[name] = ""
            # unreadable audio (PyAV) is "decode", as in run_omni: not a model failure
            cause = "decode" if type(e).__module__.split(".")[0] == "av" else f"whisper_{type(e).__name__}"
            FAILS[cause] += 1
            log(f"whisper clip {i} failed: {type(e).__name__}" + (f" {e}"[:250] if DEBUG else ""))
        if (i + 1) % 50 == 0:
            log(f"whisper done {i + 1}/{len(rows)}")
    # Clips that came back empty (VAD heard no speech, or the clip errored) get one
    # unbatched retry without VAD, keeping only segments whisper itself rates as
    # speech. Measured in the official image (large-v3): 4 s of digital silence
    # comes back as "Teksting av Nicolai Winther" (a known hallucination) with
    # no_speech_prob 0.86; the same speech clip scores 0.17 at full volume and
    # 0.48 at -30 dB. Below the cut the retry keeps it; otherwise the fallback word wins.
    todo = [r for r in rows if not results.get(r["audio_filename"], "").strip()]
    for r in todo:
        if time.time() - T0 > TIME_BUDGET_S:
            break
        name, code = r["audio_filename"], (r.get("language") or "").strip()
        try:
            segs, _info = model.transcribe(
                str(CLIPS_DIR / name),
                language=LANG_MAP.get(code, None),
                task="transcribe",
                beam_size=beam,
                condition_on_previous_text=False,
                without_timestamps=True,
                vad_filter=False,
            )
            text = " ".join(s.text.strip() for s in segs if s.no_speech_prob < RETRY_MAX_NO_SPEECH).strip()
        except Exception as e:
            RESCUE[f"retry_{type(e).__name__}"] += 1
            continue
        if text:
            results[name] = nahuatl_orthography(text) if code in NAHUATL else text
            RESCUE["whisper_novad"] += 1
    if todo:
        log(f"whisper retry: {len(todo)} empty clip(s), {RESCUE['whisper_novad']} recovered")
    del pipe, model


def run_omni(rows, results, model=None, load=None):
    """model/load are injectable for tests (tests/test_oom.py)."""
    if model is None:
        import torch
        from omni import OmniCTC

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = OmniCTC(OMNI_DIR, device=device)
        log(f"omni loaded on {device}")
    if load is None:
        from faster_whisper.audio import decode_audio

        def load(name):
            return decode_audio(str(CLIPS_DIR / name), sampling_rate=16000)

    m = model
    budget = float(os.environ.get("LIT_OMNI_BATCH_S", 240))
    step = 64
    for k in range(0, len(rows), step):
        group = rows[k : k + step]
        if time.time() - T0 > TIME_BUDGET_S:
            for r in group:
                results[r["audio_filename"]] = ""
            FAILS["time_budget"] += len(group)
            continue
        wavs, ok = [], []
        for r in group:
            try:
                wavs.append(load(r["audio_filename"]))
                ok.append(r)
            except Exception as e:
                results[r["audio_filename"]] = ""
                FAILS["decode"] += 1
                log(f"omni decode failed: {type(e).__name__}")
        try:
            texts = m.transcribe_many(wavs, max_batch_s=budget)
        except Exception as e:
            # transcribe_many already backs off on OOM down to one chunk; what
            # reaches here is a single chunk that won't fit or a non-OOM error.
            # Retry clip by clip so one bad clip can't blank the other 63.
            log(f"omni batch failed: {type(e).__name__}; retrying {len(ok)} clips one by one")
            del e
            texts = []
            for w in wavs:
                try:
                    texts.append(m.transcribe_many([w], max_batch_s=budget)[0])
                except Exception as e1:
                    texts.append(None)
                    FAILS[f"omni_{type(e1).__name__}"] += 1
                    log(f"omni clip failed: {type(e1).__name__}")
        for r, t in zip(ok, texts):
            code = (r.get("language") or "").strip()
            t = t or ""
            results[r["audio_filename"]] = nahuatl_orthography(t) if code in NAHUATL else t
        log(f"omni done {min(k + step, len(rows))}/{len(rows)}; budget={getattr(m, 'budget_s', None)}s oom_retries={getattr(m, 'oom_retries', 0)}")
    del m


def run_qwen(rows, results):
    import torch
    from faster_whisper.audio import decode_audio
    from qwen_asr import Qwen3ASRModel

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    m = Qwen3ASRModel.from_pretrained(
        str(QWEN_DIR), dtype=dtype, device_map="cuda:0",
        max_inference_batch_size=int(os.environ.get("LIT_QWEN_BATCH", 32)),
        max_new_tokens=int(os.environ.get("LIT_QWEN_TOKENS", 1024)),
    )
    log(f"qwen loaded ({dtype})")
    step = 32
    for k in range(0, len(rows), step):
        group = rows[k : k + step]
        if time.time() - T0 > TIME_BUDGET_S:
            for r in group:
                results[r["audio_filename"]] = ""
            continue
        auds, langs, ok = [], [], []
        for r in group:
            try:
                auds.append((decode_audio(str(CLIPS_DIR / r["audio_filename"]), sampling_rate=16000), 16000))
                langs.append(QWEN_LANG.get((r.get("language") or "").strip()))
                ok.append(r)
            except Exception as e:
                results[r["audio_filename"]] = ""
                log(f"qwen decode failed: {type(e).__name__}")
        try:
            outs = [o.text for o in m.transcribe(audio=auds, language=langs)]
        except Exception as e:
            log(f"qwen batch failed: {type(e).__name__}")
            FAILS[f"qwen_{type(e).__name__}"] += len(ok)
            outs = [""] * len(ok)
        for r, t in zip(ok, outs):
            results[r["audio_filename"]] = (t or "").strip()
        log(f"qwen done {min(k + step, len(rows))}/{len(rows)}")
    del m


RUNNERS = {"whisper": run_whisper, "omni": run_omni, "qwen": run_qwen}


def main() -> int:
    rows = list(csv.DictReader(MANIFEST.open(newline="", encoding="utf-8")))
    smoke = os.environ.get("LOST_IN_TRANSCRIPTION_IS_SMOKE", "0") == "1"
    log(f"manifest rows: {len(rows)}; smoke={smoke}; whisper={HAVE_W} qwen={HAVE_Q} omni={OMNI_DIR.name if HAVE_O else None}")
    # longest first keeps batches full and surfaces OOM early
    rows.sort(key=lambda r: -float(r.get("file_duration_seconds") or 0))
    results: dict[str, str] = {}
    by_backend: dict[str, list] = {}
    for r in rows:
        by_backend.setdefault(BACKENDS.get((r.get("language") or "").strip(), "omni" if HAVE_O and not HAVE_W else "whisper"), []).append(r)
    for backend, rs in by_backend.items():
        log(f"backend {backend}: {len(rs)} clips")
        RUNNERS[backend](rs, results)

    raw_empty = sum(1 for r in rows if not (results.get(r["audio_filename"]) or "").strip())
    written, empty = write_submission(results)
    log(f"wrote {written} rows; empty={empty}; raw_empty={raw_empty}; rescue={dict(RESCUE) or 0}; failed={dict(FAILS) or 0}")
    return exit_status(len(rows))


def write_submission(results) -> tuple[int, int]:
    """One row per manifest row, in manifest order, every transcript cleaned."""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    written = empty = 0
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(["audio_filename", "transcript"])
        with MANIFEST.open(newline="", encoding="utf-8") as mf:
            for r in csv.DictReader(mf):
                name = r["audio_filename"]
                text = clean_transcript(results.get(name, ""), (r.get("language") or "").strip())
                w.writerow([name, text])
                written += 1
                empty += not text
    return written, empty


def exit_status(n_rows: int) -> int:
    """3 if too many clips failed in the model; unreadable audio and time-budget
    skips are logged above but don't fail the run (a retry wouldn't fix them)."""
    model_fails = sum(v for k, v in FAILS.items() if k not in ("decode", "time_budget"))
    if model_fails > MAX_FAIL_FRAC * max(n_rows, 1):
        log(f"FAILING RUN: {model_fails}/{n_rows} clips failed in the model (limit {MAX_FAIL_FRAC:.0%})")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
