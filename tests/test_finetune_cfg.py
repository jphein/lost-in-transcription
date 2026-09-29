"""A fine-tuned Omni model directory (train/export.py) is picked first, and its lit.json can switch the
Nahuatl orthography map off. Run inside the official runtime image (scripts/test_in_image.sh)."""

import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import main  # noqa: E402

NAH = {"azz", "nhw", "nhi"}


def reload_with(model_dir: Path, **env):
    e = {k: v for k, v in os.environ.items() if k not in ("LIT_OMNI", "LIT_NAH_ORTHO")}
    e.update({"LIT_OMNI": str(model_dir)}, **env)
    with mock.patch.dict(os.environ, e, clear=True):
        return importlib.reload(main)


class LitJson(unittest.TestCase):
    def tearDown(self):
        importlib.reload(main)  # leave the module as the other tests expect it

    def test_no_lit_json_keeps_the_map(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(reload_with(Path(d)).NAHUATL, NAH)

    def test_lit_json_can_switch_the_map_off(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "lit.json").write_text(json.dumps({"nah_ortho": False}))
            self.assertEqual(reload_with(Path(d)).NAHUATL, set())

    def test_lit_json_true_keeps_the_map(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "lit.json").write_text(json.dumps({"nah_ortho": True}))
            self.assertEqual(reload_with(Path(d)).NAHUATL, NAH)

    def test_env_still_wins(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "lit.json").write_text(json.dumps({"nah_ortho": False}))
            self.assertEqual(reload_with(Path(d), LIT_NAH_ORTHO="1").NAHUATL, NAH)

    def test_corrupt_or_odd_lit_json_means_defaults(self):
        for body in ["{not json", "[1, 2]", '"nah_ortho"', ""]:
            with tempfile.TemporaryDirectory() as d:
                (Path(d) / "lit.json").write_text(body)
                self.assertEqual(reload_with(Path(d)).NAHUATL, NAH, body)


class BatchBudget(unittest.TestCase):
    def tearDown(self):
        importlib.reload(main)

    def _budget_seen(self, m):
        seen = {}

        class Fake:
            budget_s, oom_retries = None, 0

            def transcribe_many(self, wavs, max_batch_s=240.0):
                seen["b"] = max_batch_s
                return ["kuali"] * len(wavs)

        rows = [{"audio_filename": "a.mp3", "language": "nhi"}]
        m.run_omni(rows, {}, model=Fake(), load=lambda name: [0.0] * 16000)
        return seen["b"]

    def test_lit_json_batch_budget_is_the_default(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "lit.json").write_text(json.dumps({"omni_batch_s": 1}))
            self.assertEqual(self._budget_seen(reload_with(Path(d))), 1.0)

    def test_stock_budget_without_lit_json(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(self._budget_seen(reload_with(Path(d))), 240.0)

    def test_env_budget_wins(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "lit.json").write_text(json.dumps({"omni_batch_s": 1}))
            m = reload_with(Path(d), LIT_OMNI_BATCH_S="60")
            with mock.patch.dict(os.environ, {"LIT_OMNI_BATCH_S": "60"}):
                self.assertEqual(self._budget_seen(m), 60.0)


class PickOmni(unittest.TestCase):
    def test_finetuned_dir_is_preferred(self):
        with tempfile.TemporaryDirectory() as d:
            for n in ("omni-3b-v2-fp16", "omni-3b-ft"):
                (Path(d) / n).mkdir()
            env = {k: v for k, v in os.environ.items() if k != "LIT_OMNI"}
            with mock.patch.object(main, "MODELS", Path(d)), mock.patch.dict(os.environ, env, clear=True):
                self.assertEqual(main._pick_omni().name, "omni-3b-ft")

    def test_stock_dir_when_no_finetune(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "omni-3b-v2-fp16").mkdir()
            env = {k: v for k, v in os.environ.items() if k != "LIT_OMNI"}
            with mock.patch.object(main, "MODELS", Path(d)), mock.patch.dict(os.environ, env, clear=True):
                self.assertEqual(main._pick_omni().name, "omni-3b-v2-fp16")


if __name__ == "__main__":
    unittest.main()


class SegWithoutMap(unittest.TestCase):
    """lit.json can switch the orthography map off; the word-boundary repair must not go with it."""

    def tearDown(self):
        importlib.reload(main)

    def test_seg_runs_when_map_is_off(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "lit.json").write_text(json.dumps({"nah_ortho": False}))
            m = reload_with(Path(d))
            self.assertEqual(m.NAHUATL, set())
            self.assertTrue(m.NAH_SEG)
            with mock.patch("segment.fix_boundaries", side_effect=lambda t: t + "|seg") as fb:
                self.assertEqual(m.reference_style("abc", "nhi"), "abc|seg")
                fb.assert_called_once()

    def test_lit_json_can_switch_seg_off(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "lit.json").write_text(json.dumps({"nah_ortho": False, "nah_seg": False}))
            m = reload_with(Path(d))
            self.assertFalse(m.NAH_SEG)
            self.assertEqual(m.reference_style("abc", "nhi"), "abc")

    def test_non_nahuatl_untouched(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "lit.json").write_text(json.dumps({"nah_ortho": False, "id_post": False}))
            m = reload_with(Path(d))
            self.assertEqual(m.reference_style("abc 12", "jav"), "abc 12")
