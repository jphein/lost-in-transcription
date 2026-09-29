"""Map Indonesian/Javanese ASR output to the reference conventions (in-jv track: codes ind, jav, javind).

The acoustic model writes everything lowercase and sometimes writes numbers as digits. The references
capitalize acronyms and proper nouns and spell numbers out in Indonesian. Three fixed rules, in this order:
  1. digits -> Indonesian number words ("2018" -> "dua ribu delapan belas"), matching num2words(lang="id");
  2. common acronyms -> UPPER ("sma" -> "SMA"); the scorer keeps a capital that is followed by a capital,
     so this is safe even at a sentence start;
  3. common proper nouns (places, countries, religions, holidays, days, months) -> Titlecase.
The lists are general knowledge of Indonesian, not learned from any reference transcripts.
Measured on the in-jv dev set (372 clips, official scorer): -1.53 WER points [-1.94, -1.01], bootstrapped by
speaker (scratch/audit-dev/indonesian/levers-speaker.txt).
"""
import re

ACRONYMS = {"tk", "sd", "smp", "sma", "smk", "mts", "sltp", "slta", "hp", "pr", "sms", "tv", "ktp", "sim", "atm",
            "wa", "pc", "ac", "dvd", "cd", "usb", "pns", "tni", "polri", "dpr", "mpr", "kpk", "bpjs", "pln",
            "pdam", "rt", "rw", "rs", "ukm", "umkm", "osis", "pkl", "kkn", "ipk", "ipa", "ips", "uas", "uts", "ppkn",
            "bk", "ptn", "pts", "snmptn", "sbmptn", "utbk", "cpns", "sdm", "bumn"}
PROPER = {"indonesia", "jawa", "jakarta", "jogja", "yogya", "yogyakarta", "solo", "surakarta", "semarang", "surabaya",
          "bandung", "bali", "malang", "madiun", "kediri", "klaten", "sragen", "boyolali", "wonogiri", "karanganyar",
          "sukoharjo", "madura", "sumatra", "sumatera", "kalimantan", "sulawesi", "papua", "lombok", "jepang", "korea",
          "cina", "china", "amerika", "malaysia", "singapura", "arab", "belanda", "inggris", "eropa", "asia", "afrika",
          "australia", "india", "thailand", "jerman", "perancis", "prancis", "islam", "allah", "kristen", "katolik",
          "hindu", "buddha", "lebaran", "ramadan", "ramadhan", "natal", "senin", "selasa", "rabu", "kamis", "jumat",
          "sabtu", "januari", "februari", "maret", "april", "juni", "juli", "agustus", "september", "oktober",
          "november", "desember"}
_TOK = re.compile(r"^(\W*)(.*?)(\W*)$", re.S)
_ONES = ["nol", "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "delapan", "sembilan"]
_SCALES = [(10 ** 12, "triliun"), (10 ** 9, "miliar"), (10 ** 6, "juta")]


def _below_1000(n: int) -> str:
    parts = []
    h, r = divmod(n, 100)
    if h:
        parts.append("seratus" if h == 1 else _ONES[h] + " ratus")
    if r:
        if r < 10:
            parts.append(_ONES[r])
        elif r == 10:
            parts.append("sepuluh")
        elif r == 11:
            parts.append("sebelas")
        elif r < 20:
            parts.append(_ONES[r - 10] + " belas")
        else:
            t, o = divmod(r, 10)
            parts.append(_ONES[t] + " puluh" + (" " + _ONES[o] if o else ""))
    return " ".join(parts)


def number_words(n: int) -> str:
    """Indonesian cardinal, as num2words(n, lang="id") writes it."""
    if n == 0:
        return "nol"
    parts = []
    for size, name in _SCALES:
        q, n = divmod(n, size)
        if q:
            parts.append(number_words(q) + " " + name)
    q, n = divmod(n, 1000)
    if q:
        parts.append("seribu" if q == 1 else _below_1000(q) + " ribu")
    if n:
        parts.append(_below_1000(n))
    return " ".join(parts)


def _digits(m: re.Match) -> str:
    s = m.group(0)
    return number_words(int(s)) if len(s) <= 15 else s


def _case(core: str) -> str:
    low = core.lower()
    if low in ACRONYMS:
        return core.upper()
    if low in PROPER:
        return core[:1].upper() + core[1:].lower()
    return core


def to_reference(text: str) -> str:
    text = re.sub(r"\d+", _digits, text)
    out = []
    for tok in text.split():
        pre, core, post = _TOK.match(tok).groups()
        out.append(pre + _case(core) + post)
    return " ".join(out)
