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
