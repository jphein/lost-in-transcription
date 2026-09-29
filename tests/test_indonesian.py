"""Indonesian/Javanese reference conventions (indonesian.py). No GPU and no model weights needed."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import indonesian as ind  # noqa: E402
import main  # noqa: E402


class Numbers(unittest.TestCase):
    def test_standard_forms(self):
        cases = {0: "nol", 7: "tujuh", 10: "sepuluh", 11: "sebelas", 12: "dua belas", 19: "sembilan belas",
                 20: "dua puluh", 21: "dua puluh satu", 100: "seratus", 101: "seratus satu", 110: "seratus sepuluh",
                 111: "seratus sebelas", 200: "dua ratus", 1000: "seribu", 1100: "seribu seratus",
                 2018: "dua ribu delapan belas", 10000: "sepuluh ribu", 11000: "sebelas ribu",
                 100000: "seratus ribu", 1000000: "satu juta", 2500000: "dua juta lima ratus ribu"}
        for n, want in cases.items():
            self.assertEqual(ind.number_words(n), want, n)


class References(unittest.TestCase):
    def test_digits_are_spelled_out(self):
        self.assertEqual(ind.to_reference("tahun 2018 kelas 2"), "tahun dua ribu delapan belas kelas dua")

    def test_acronyms_and_proper_nouns(self):
        self.assertEqual(ind.to_reference("aku sma di jogja pakai hp"), "aku SMA di Jogja pakai HP")

    def test_punctuation_survives_and_unknown_words_stay(self):
        self.assertEqual(ind.to_reference("sma, terus kuliah."), "SMA, terus kuliah.")
        self.assertEqual(ind.to_reference(""), "")


class Dispatch(unittest.TestCase):
    def test_only_indonesian_codes_are_touched(self):
        self.assertIn("javind", main.INDONESIAN)
        self.assertEqual(main.reference_style("kelas 2 sma", "javind"), "kelas dua SMA")
        self.assertEqual(main.reference_style("kelas 2 sma", "ind"), "kelas dua SMA")
        self.assertEqual(main.reference_style("kelas 2 sma", "enspa"), "kelas 2 sma")

    def test_nahuatl_codes_take_the_nahuatl_path(self):
        self.assertEqual(main.reference_style("sma", "nhw"), main.nahuatl_orthography("sma"))


if __name__ == "__main__":
    unittest.main()
