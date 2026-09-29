"""OOM back-off tests. Needs torch: run inside the official runtime image,
see scripts/test_in_image.sh. No GPU and no model weights needed."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import main  # noqa: E402
import omni  # noqa: E402

SR = omni.SR


class FakeOmni(omni.OmniCTC):
    """OmniCTC without weights: _batch OOMs when a batch holds more than
    `fits_s` seconds of audio, or holds any chunk longer than `poison_s`."""

    def __init__(self, fits_s=1e9, poison_s=1e9, error=None):
        self.device = "cpu"
        self.fits_s, self.poison_s, self.error = fits_s, poison_s, error
        self.calls = []

    def _batch(self, chunks):
        secs = [len(c) / SR for c in chunks]
        self.calls.append(len(chunks))
        if self.error:
            raise self.error
        if sum(secs) > self.fits_s or max(secs) > self.poison_s:
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return [f"w{round(s)}" for s in secs]


def wav(seconds):
    return np.random.default_rng(0).standard_normal(int(seconds * SR)).astype(np.float32) * 0.01


class TranscribeManyBackoff(unittest.TestCase):
    def test_backoff_recovers_every_clip(self):
        m = FakeOmni(fits_s=60)
        out = m.transcribe_many([wav(20) for _ in range(10)], max_batch_s=240)
        self.assertEqual(out, ["w20"] * 10)  # nothing blank
        self.assertGreaterEqual(m.oom_retries, 1)
        self.assertLessEqual(m.budget_s, 60)

    def test_reduced_budget_sticks_across_calls(self):
        m = FakeOmni(fits_s=60)
        m.transcribe_many([wav(20) for _ in range(10)], max_batch_s=240)
        r = m.oom_retries
        m.transcribe_many([wav(20) for _ in range(10)], max_batch_s=240)
        self.assertEqual(m.oom_retries, r)  # no re-learning on the next group

    def test_long_clip_chunks_all_transcribed(self):
        m = FakeOmni(fits_s=45)
        (out,) = m.transcribe_many([wav(95)], max_batch_s=240)  # 95 s -> ~4 chunks
        self.assertTrue(out and all(t.startswith("w") for t in out.split()))
        self.assertGreaterEqual(len(out.split()), 3)

    def test_single_chunk_oom_raises_not_blank(self):
        m = FakeOmni(fits_s=0)
        with self.assertRaises(omni.BatchOOM):
            m.transcribe_many([wav(5), wav(5)], max_batch_s=240)

    def test_non_oom_error_propagates_without_retry(self):
        m = FakeOmni(error=ValueError("bad input"))
        with self.assertRaises(ValueError):
            m.transcribe_many([wav(5)] * 3, max_batch_s=240)
        self.assertEqual(m.calls, [3])

    def test_is_oom_matches_runtimeerror_text(self):
        self.assertTrue(omni.is_oom(RuntimeError("CUDA error: out of memory")))
        self.assertFalse(omni.is_oom(RuntimeError("shape mismatch")))


class RunOmniAccounting(unittest.TestCase):
    def setUp(self):
        main.FAILS.clear()

    def rows(self, n):
        return [{"audio_filename": f"c{i}.mp3", "language": "jav"} for i in range(n)]

    def test_one_poison_clip_does_not_blank_the_group(self):
        lens = {f"c{i}.mp3": 5.0 for i in range(10)}
        lens["c3.mp3"] = 28.0  # a single chunk that never fits
        results = {}
        main.run_omni(self.rows(10), results, model=FakeOmni(poison_s=25), load=lambda n: wav(lens[n]))
        self.assertEqual(results["c3.mp3"], "")
        self.assertEqual(sum(1 for v in results.values() if v), 9)
        self.assertEqual(main.FAILS["omni_BatchOOM"], 1)
        self.assertEqual(main.exit_status(10), 3)  # 10% > 2%: fail loudly

    def test_decode_failures_counted_but_do_not_fail_run(self):
        def load(n):
            if n == "c0.mp3":
                raise EOFError("stub file")
            return wav(5)

        results = {}
        main.run_omni(self.rows(10), results, model=FakeOmni(fits_s=20), load=load)
        self.assertEqual(main.FAILS["decode"], 1)
        self.assertEqual(sum(1 for v in results.values() if v), 9)
        self.assertEqual(main.exit_status(10), 0)

    def test_clean_run_exits_zero(self):
        results = {}
        main.run_omni(self.rows(70), results, model=FakeOmni(fits_s=30), load=lambda n: wav(4))
        self.assertTrue(all(results.values()) and len(results) == 70)
        self.assertEqual(set(results.values()), {"wempat"})  # routed through reference_style (jav: digits)
        self.assertEqual(dict(main.FAILS), {})
        self.assertEqual(main.exit_status(70), 0)


if __name__ == "__main__":
    unittest.main()
