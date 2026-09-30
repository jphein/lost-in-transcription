"""Map Nahuatl ASR output to the competition's reference orthography.

Official convention (problem description): u for /w/, k for /k/, s for /s/;
Spanish words keep standard Spanish spelling. The competition's own example
transcript ("tejua", "ijkuak", "nechajsi", "ajko") also writes /h/ as j, so
that is on by default (LIT_NAH_J=0 turns it off).

Spanish words are detected with a small CC-BY lexicon built from FLEURS es_419
transcripts (es_words.txt).
"""

import os
import re
from pathlib import Path

_ES = set()
_p = Path(__file__).with_name("es_words.txt")
if _p.exists():
    _ES = {w.strip() for w in _p.read_text(encoding="utf-8").split() if w.strip()}

J_FOR_H = os.environ.get("LIT_NAH_J", "1") == "1"
_TOKEN = re.compile(r"^(\W*)(.*?)(\W*)$", re.S)


def _fix_word(w: str, j_for_h: bool) -> str:
    low = w.lower()
    if not low or low in _ES or not re.search(r"[a-zñ]", low):
        return w
    cap = w[:1].isupper()
    x = low
    x = x.replace("ch", "\x00").replace("sh", "\x01")  # protect digraphs
    x = x.replace("tz", "ts")
    x = re.sub(r"qu(?=[ei])", "k", x)
    x = re.sub(r"c(?=[ei])", "s", x)
    x = x.replace("c", "k").replace("z", "s").replace("q", "k")
    x = re.sub(r"hu(?=[aeio])", "u", x)  # classical hu = /w/
    x = x.replace("uh", "u")  # classical final uh = /w/
    x = x.replace("w", "u")
    if j_for_h:
        x = x.replace("h", "j")
    x = x.replace("\x00", "ch").replace("\x01", "sh")
    return x[:1].upper() + x[1:] if cap else x


def to_reference(text: str, j_for_h: bool = J_FOR_H) -> str:
    out = []
    for tok in text.split():
        m = _TOKEN.match(tok)
        pre, core, post = m.groups() if m else ("", tok, "")
        out.append(pre + _fix_word(core, j_for_h) + post)
    return " ".join(out)


if __name__ == "__main__":
    for s in ["tehwan tikilwiah artesania", "kikuih ya para yehwan", "ye kualtzin porque ye moskaltia",
              "ihkwak nechahsi", "hasta que el semana", "quemah se chichiltik"]:
        print(f"{s!r:40} -> {to_reference(s)!r}")
