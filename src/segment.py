"""Word-boundary repair for Nahuatl ASR output (sp-nh), from word counts of reference transcripts.

The acoustic model fuses short words onto a neighbour ("kikuij ya" -> "kikuijya", "no uelik" -> "nouelik")
and now and then splits one ("telchikauak" -> "tel chikauak"). Each costs about two word errors. Two rules,
both driven by counts from the references (nah_vocab.tsv):
  join:  two adjacent words become one when the references write them joined (at least MINC times)
         more often than apart. This is the shipped rule: -0.40 WER points [-0.80, -0.10] on the sp-nh
         dev set, cross-validated by conversation (scratch/audit-dev/nahuatl/levers-b1.txt).
  split: a word the references never use becomes two words they use at least MINC times each.
         OFF by default: it made the dev set WORSE (+1.04 [+0.15, +1.88]), because an unseen Nahuatl
         word is usually a real word that happens to decompose, not a fusion.
Nothing changes when the vocabulary is missing.

nah_vocab.tsv is derived from the Lost in Transcription dev set (Mozilla Data Collective, CC-BY-SA-4.0).
It holds counts only: "u<TAB>word<TAB>n" and "b<TAB>word word<TAB>n".
"""
import re
from collections import Counter
from pathlib import Path

MINC = 2
_TOK = re.compile(r"^(\W*)(.*?)(\W*)$", re.S)  # leading punctuation, core, trailing punctuation
UNI: Counter = Counter()
BI: Counter = Counter()


def core(tok: str) -> str:
    return _TOK.match(tok).group(2)


def learn(normalized_texts):
    """Unigram and bigram counts of lowercase word cores, from already-normalized reference texts."""
    uni, bi = Counter(), Counter()
    for t in normalized_texts:
        w = [c for c in (core(x).lower() for x in t.split()) if c]
        uni.update(w)
        bi.update(zip(w, w[1:]))
    return uni, bi


def save(path, uni, bi, minc=MINC):
    """Write the counts; only bigrams the join rule can consult (their joined form is a vocabulary word)."""
    with open(path, "w", encoding="utf-8") as f:
        for w, n in sorted(uni.items()):
            f.write(f"u\t{w}\t{n}\n")
        for (a, b), n in sorted(bi.items()):
            if uni.get(a + b, 0) >= minc:
                f.write(f"b\t{a} {b}\t{n}\n")


def load(path=None) -> int:
    p = Path(path) if path else Path(__file__).with_name("nah_vocab.tsv")
    UNI.clear()
    BI.clear()
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            kind, key, n = line.split("\t")
            if kind == "u":
                UNI[key] += int(n)
            else:
                a, b = key.split(" ")
                BI[(a, b)] += int(n)
    return len(UNI)


def split_words(text: str, uni=None, minc: int = MINC) -> str:
    uni = UNI if uni is None else uni

    def fix(c: str) -> str:
        low = c.lower()
        if not low or uni.get(low, 0) > 0:
            return c
        best = None
        for i in range(2, len(low) - 1):
            ca, cb = uni.get(low[:i], 0), uni.get(low[i:], 0)
            if ca >= minc and cb >= minc and (best is None or ca * cb > best[0]):
                best = (ca * cb, i)
        return c[:best[1]] + " " + c[best[1]:] if best else c

    out = []
    for tok in text.split():
        pre, c, post = _TOK.match(tok).groups()
        out.append(pre + fix(c) + post)
    return " ".join(out)


def join_words(text: str, uni=None, bi=None, minc: int = MINC) -> str:
    uni = UNI if uni is None else uni
    bi = BI if bi is None else bi
    t = text.split()
    out, k = [], 0
    while k < len(t):
        if k + 1 < len(t):
            (_, a, a_post), (b_pre, b, _) = _TOK.match(t[k]).groups(), _TOK.match(t[k + 1]).groups()
            a, b = a.lower(), b.lower()
            # never join across punctuation: "kikuij, ya" -> "kikuij,ya" would be split again by the scorer
            if a and b and not a_post and not b_pre and uni.get(a + b, 0) >= minc and uni[a + b] > bi.get((a, b), 0):
                out.append(t[k] + t[k + 1])
                k += 2
                continue
        out.append(t[k])
        k += 1
    return " ".join(out)


def fix_boundaries(text: str, uni=None, bi=None, split: bool = False) -> str:
    if uni is None and not UNI:
        load()
    if not (UNI if uni is None else uni):
        return text
    if split:
        text = split_words(text, uni)
    return join_words(text, uni, bi)
