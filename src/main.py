"""Lost in Transcription submission: faster-whisper large-v3, per-clip language forcing.

One zip serves all three tracks; the track is inferred from the manifest's
`language` column. Logs only aggregate progress (no test data).
"""

import csv
import os
import re
import sys
import time
from pathlib import Path

T0 = time.time()
TIME_BUDGET_S = float(os.environ.get("LIT_TIME_BUDGET_S", 105 * 60))  # hard limit is 120 min

ROOT = Path.cwd()  # /code_execution
DATA_DIR = ROOT / "data"
CLIPS_DIR = DATA_DIR / "clips"
MANIFEST = DATA_DIR / "test_metadata.csv"
OUT = ROOT / "submission" / "submission.csv"
SRC = Path(__file__).resolve().parent
MODEL_DIR = SRC / "models" / os.environ.get("LIT_MODEL", "faster-whisper-large-v3")

# manifest language code -> whisper language code
LANG_MAP = {
    "spa": "es", "enspa": None, "eng": "en",  # enspa: let whisper pick per chunk
    "ind": "id", "jav": "jw", "javind": "id",
    "azz": "es", "nhw": "es", "nhi": "es",  # no Nahuatl in whisper; es phonetics closest
}
NAHUATL = {"azz", "nhw", "nhi"}

# Short vocabulary priming per track (initial_prompt). Kept tiny: long prompts
# make whisper hallucinate on short clips.
PROMPTS = {
    "enspa": "So, like, yo le dije que I was going, pero she didn't want to, ¿sabes?",
    "spa": None,
    "javind": None, "jav": None, "ind": None,
    "azz": None, "nhw": None, "nhi": None,
}


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def nahuatl_orthography(text: str) -> str:
    """Map to the reference convention: u for /w/, k for /k/, s for /s/.

    Applied only to words that don't look like Spanish (heuristic: contains
    Nahuatl-typical clusters). Spanish words keep standard orthography.
    """
    def fix(word: str) -> str:
        w = word
        low = w.lower()
        if not re.search(r"(tl|tz|hu|uh|kw|cu[aeio]|x|ts|w)", low):
            return w
        w = re.sub(r"hu|uh|w", "u", w)
        w = re.sub(r"Hu|W", "U", w)
        w = re.sub(r"qu(?=[ei])", "k", w)
        w = re.sub(r"c(?=[aou])|c$|c(?=[^aeiouh])", "k", w)
        w = re.sub(r"z|c(?=[ei])", "s", w)
        w = w.replace("tz", "ts")
        return w

    return " ".join(fix(t) for t in text.split())


def main() -> int:
    import torch  # noqa: F401  (ensures CUDA libs are loaded before ct2)
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    rows = list(csv.DictReader(MANIFEST.open(newline="", encoding="utf-8")))
    log(f"manifest rows: {len(rows)}")
    smoke = os.environ.get("LOST_IN_TRANSCRIPTION_IS_SMOKE", "0") == "1"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute = "float16" if device == "cuda" else "int8"
    model = WhisperModel(str(MODEL_DIR), device=device, compute_type=compute)
    pipe = BatchedInferencePipeline(model=model)
    log(f"model loaded on {device} ({compute}); smoke={smoke}")

    beam = int(os.environ.get("LIT_BEAM", 5))
    batch = int(os.environ.get("LIT_BATCH", 16))
    results: dict[str, str] = {}
    # longest first keeps batches full and surfaces OOM early
    rows.sort(key=lambda r: -float(r.get("file_duration_seconds") or 0))
    for i, r in enumerate(rows):
        name = r["audio_filename"]
        code = (r.get("language") or "").strip()
        if time.time() - T0 > TIME_BUDGET_S:
            results[name] = ""
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
            if code in NAHUATL:
                text = nahuatl_orthography(text)
            results[name] = text
        except Exception as e:  # never let one clip kill the run
            results[name] = ""
            log(f"clip {i} failed: {type(e).__name__}")
        if (i + 1) % 50 == 0:
            log(f"done {i + 1}/{len(rows)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(["audio_filename", "transcript"])
        for r in csv.DictReader(MANIFEST.open(newline="", encoding="utf-8")):
            w.writerow([r["audio_filename"], results.get(r["audio_filename"], "")])
    log(f"wrote {len(results)} rows; empty={sum(1 for v in results.values() if not v)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
