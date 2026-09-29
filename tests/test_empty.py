"""Never-empty submission tests. The platform's validator rejected sp-en on
2026-09-29 because one of 592 transcripts was empty. Run inside the official
runtime image (see scripts/test_in_image.sh); no GPU and no model weights needed."""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import main  # noqa: E402


class CleanTranscript(unittest.TestCase):
    def test_empty_becomes_one_word_for_its_language(self):
        before = main.RESCUE["placeholder"]
        self.assertEqual(main.clean_transcript("", "enspa"), "yeah")
        self.assertEqual(main.clean_transcript("   ", "spa"), "sí")
        self.assertEqual(main.clean_transcript(None, "jav"), "iya")
        self.assertEqual(main.clean_transcript("\n\t", "nhi"), "kemaj")
        self.assertEqual(main.clean_transcript("", "unknown-code"), "yeah")
        self.assertEqual(main.RESCUE["placeholder"] - before, 5)  # counted, never silent

    def test_pandas_na_strings_stay_text(self):
        for s in ["NA", "nan", "null", "None", "N/A", "NULL", "NaN"]:
            self.assertEqual(main.clean_transcript(s, "enspa"), s + ".")

    def test_one_line_no_control_characters(self):
        self.assertEqual(main.clean_transcript(" hola\n que\ttal\x00 ", "spa"), "hola que tal")

    def test_normal_text_unchanged(self):
        t = 'So, like, "yo le dije" que I was going, ¿sabes?'
        self.assertEqual(main.clean_transcript(t, "enspa"), t)


class WriteSubmission(unittest.TestCase):
    def test_written_file_passes_a_pandas_validator(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            names = [f"c{i}.mp3" for i in range(6)]
            with open(tmp / "test_metadata.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["audio_filename", "file_duration_seconds", "language"])
                for n, lang in zip(names, ["enspa", "spa", "jav", "nhi", "enspa", "enspa"]):
                    w.writerow([n, 3, lang])
            results = {"c0.mp3": "", "c1.mp3": "NA", "c2.mp3": "iki, \"kutha\"\nbaris",
                       "c3.mp3": "   ", "c4.mp3": "hello, world"}  # c5 missing entirely
            old = main.MANIFEST, main.OUT
            main.MANIFEST, main.OUT = tmp / "test_metadata.csv", tmp / "sub" / "submission.csv"
            try:
                written, empty = main.write_submission(results)
            finally:
                main.MANIFEST, main.OUT = old
            self.assertEqual((written, empty), (6, 0))
            raw = (tmp / "sub" / "submission.csv").read_bytes()
            raw.decode("utf-8")  # valid UTF-8
            df = pd.read_csv(tmp / "sub" / "submission.csv")  # pandas' default NA parsing
            self.assertEqual(list(df.columns), ["audio_filename", "transcript"])
            self.assertEqual(list(df.audio_filename), names)  # every id, once, in order
            self.assertFalse(df.transcript.isna().any())
            self.assertFalse((df.transcript.str.strip() == "").any())
            self.assertEqual(df.transcript.tolist()[:4], ["yeah", "NA.", 'iki, "kutha" baris', "kemaj"])


if __name__ == "__main__":
    unittest.main()
