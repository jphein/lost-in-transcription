"""Word-boundary repair (segment.py) for Nahuatl output. No GPU and no model weights needed."""

import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import main  # noqa: E402
import segment  # noqa: E402

UNI = Counter({"kikuij": 3, "ya": 5, "no": 9, "uelik": 4, "nochi": 6, "chi": 3, "tel": 2, "chikauak": 2,
               "telchikauak": 3, "se": 2})
BI = Counter({("tel", "chikauak"): 1})


class Split(unittest.TestCase):
    def test_fused_function_word_is_split(self):
        self.assertEqual(segment.split_words("kikuijya nouelik", UNI), "kikuij ya no uelik")

    def test_known_word_is_never_split(self):
        self.assertEqual(segment.split_words("nochi", UNI), "nochi")  # "no" + "chi" are known, "nochi" too

    def test_both_parts_need_minc(self):
        self.assertEqual(segment.split_words("kikuijya", Counter({"kikuij": 1, "ya": 5})), "kikuijya")

    def test_punctuation_and_case_survive(self):
        self.assertEqual(segment.split_words("Kikuijya.", UNI), "Kikuij ya.")


class Join(unittest.TestCase):
    def test_joined_form_wins_when_references_prefer_it(self):
        self.assertEqual(segment.join_words("se tel chikauak", UNI, BI), "se telchikauak")

    def test_apart_form_wins_when_references_prefer_it(self):
        self.assertEqual(segment.join_words("tel chikauak", UNI, Counter({("tel", "chikauak"): 5})), "tel chikauak")

    def test_never_joins_across_punctuation(self):
        self.assertEqual(segment.join_words("tel, chikauak", UNI, BI), "tel, chikauak")


class Wiring(unittest.TestCase):
    def test_empty_vocabulary_changes_nothing(self):
        self.assertEqual(segment.fix_boundaries("kikuijya", Counter(), Counter()), "kikuijya")

    def test_save_load_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "v.tsv"
            segment.save(p, UNI, BI)
            self.assertEqual(segment.load(p), len(UNI))
            self.assertEqual(segment.fix_boundaries("kikuijya tel chikauak"), "kikuijya telchikauak")  # join only
            self.assertEqual(segment.fix_boundaries("kikuijya tel chikauak", split=True), "kikuij ya telchikauak")
        segment.load()  # back to the shipped vocabulary

    def test_shipped_path_joins_but_never_splits(self):
        if not segment.load():
            self.skipTest("nah_vocab.tsv is generated at build time from the dev references (never committed)")
        self.assertTrue(main.NAH_SEG)
        self.assertEqual(main.nahuatl_orthography("nouelik"), "nouelik")  # split is off in the zip
        from nahuatl import to_reference
        pairs = [(a, b) for (a, b), n in segment.BI.items() if segment.UNI[a + b] > n
                 and to_reference(f"{a} {b}") == f"{a} {b}"]
        self.assertTrue(pairs, "the shipped vocabulary has no join pair")
        a, b = pairs[0]
        self.assertEqual(main.nahuatl_orthography(f"{a} {b}"), a + b)


if __name__ == "__main__":
    unittest.main()
