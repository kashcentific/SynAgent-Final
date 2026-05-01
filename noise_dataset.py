"""
noise_dataset.py — Take top N rows from APIGen-MT-5k.csv,
corrupt ~half of them in realistically ugly ways, save as noised_APIGen.csv.

Usage:
  python noise_dataset.py              # default: 200 rows, noise 100
  python noise_dataset.py --n 400 --noise 200
"""

import argparse
import random
import re
import string
import pandas as pd
import numpy as np

random.seed(0)
np.random.seed(0)

# ── Noise functions ───────────────────────────────────────────────────────────

def keyboard_smash(text: str) -> str:
    """asdfjkl; type garbage injected mid-text."""
    smash = "".join(random.choices("asdfghjklqwertyuiopzxcvbnm1234567890!@#$", k=random.randint(12, 40)))
    pos = random.randint(0, max(len(text) // 2, 1))
    return text[:pos] + " " + smash + " " + text[pos:]


def random_caps_destroy(text: str) -> str:
    """rAnDoM cApS that make it unreadable."""
    return "".join(c.upper() if random.random() > 0.5 else c.lower() for c in text)


def inject_html_junk(text: str) -> str:
    """Inject raw HTML/XML tags as if copy-pasted from a webpage."""
    tags = [
        "<br/>", "<p>", "</p>", "&nbsp;", "<b>", "</b>", "<div>", "</div>",
        "&#x200B;", "&amp;", "<span style='color:red'>", "</span>",
        "<!-- comment -->", "<script>", "</script>",
    ]
    for _ in range(random.randint(3, 8)):
        pos = random.randint(0, len(text))
        text = text[:pos] + random.choice(tags) + text[pos:]
    return text


def truncate_mid_sentence(text: str) -> str:
    """Cut off the text at a random early point — no ending punctuation."""
    cut = random.randint(10, max(len(text) // 3, 11))
    return text[:cut]


def repeat_spam(text: str) -> str:
    """Pick a word and spam it 10-20 times like a broken keyboard."""
    words = text.split()
    if not words:
        return text
    spam_word = random.choice(words)
    spam = " ".join([spam_word] * random.randint(10, 20))
    pos = random.randint(0, len(words))
    words.insert(pos, spam)
    return " ".join(words)


def encoding_garbage(text: str) -> str:
    """Simulate UTF-8 mojibake / copy-paste encoding errors."""
    replacements = {
        "a": "Ã¡", "e": "Ã©", "i": "Ã­", "o": "Ã³", "u": "Ãº",
        "n": "Ã±", " ": "â€‹",  # zero-width space mojibake
        "s": "ÅŸ", "c": "Ã§",
    }
    out = []
    for ch in text:
        if ch.lower() in replacements and random.random() < 0.3:
            out.append(replacements[ch.lower()])
        else:
            out.append(ch)
    return "".join(out)


def whitespace_explosion(text: str) -> str:
    """Excessive whitespace, tabs, newlines everywhere."""
    words = text.split()
    spacer = lambda: " " * random.randint(2, 12) + "\t" * random.randint(0, 3)
    return spacer().join(words)


def null_literal_injection(text: str) -> str:
    """Sprinkle None/NULL/NaN/undefined as if a serializer broke."""
    junk = random.choices(["None", "NULL", "NaN", "undefined", "null", "#N/A", "N/A"], k=random.randint(2, 5))
    words = text.split()
    for j in junk:
        words.insert(random.randint(0, len(words)), j)
    return " ".join(words)


def number_corruption(text: str) -> str:
    """Replace all numbers with obviously wrong ones or remove them."""
    return re.sub(r"\d+", lambda m: str(random.randint(99999, 9999999)), text)


def mixed_language_paste(text: str) -> str:
    """Insert random foreign-language filler as if copy-pasted from the wrong doc."""
    filler_phrases = [
        "これは間違ったテキストです",
        "este texto está en el lugar equivocado",
        "это неправильный текст",
        "هذا نص خاطئ",
        "texte collé au mauvais endroit",
        "Dies ist ein falscher Text",
        "이것은 잘못된 텍스트입니다",
    ]
    phrase = random.choice(filler_phrases)
    pos = random.randint(0, len(text))
    return text[:pos] + " " + phrase + " " + text[pos:]


def full_replace_gibberish(text: str) -> str:
    """Nuke the whole thing and replace with pure garbage."""
    length = random.randint(20, 80)
    chars  = string.ascii_letters + string.digits + string.punctuation + "     "
    return "".join(random.choices(chars, k=length))


def empty_or_whitespace(text: str) -> str:
    """Just wipe it — empty string or a few spaces."""
    return random.choice(["", "   ", "\t", "\n\n", ".", "???"])


# ── Noise dispatcher ──────────────────────────────────────────────────────────

NOISE_FUNCTIONS = [
    (keyboard_smash,        0.14),
    (random_caps_destroy,   0.08),
    (inject_html_junk,      0.10),
    (truncate_mid_sentence, 0.10),
    (repeat_spam,           0.08),
    (encoding_garbage,      0.10),
    (whitespace_explosion,  0.06),
    (null_literal_injection,0.08),
    (number_corruption,     0.06),
    (mixed_language_paste,  0.08),
    (full_replace_gibberish,0.06),
    (empty_or_whitespace,   0.06),
]

_NAMES   = [f[0].__name__ for f in NOISE_FUNCTIONS]
_WEIGHTS = [f[1]         for f in NOISE_FUNCTIONS]
_FUNCS   = {f[0].__name__: f[0] for f in NOISE_FUNCTIONS}


def apply_noise(text: str, n_corruptions: int = 2) -> tuple[str, list[str]]:
    """Apply 1–3 random noise functions to a text value. Returns (noised_text, applied_names)."""
    if not isinstance(text, str) or not text.strip():
        return full_replace_gibberish(""), ["full_replace_gibberish"]

    chosen = random.choices(_NAMES, weights=_WEIGHTS, k=n_corruptions)
    applied = []
    for name in chosen:
        text = _FUNCS[name](text)
        applied.append(name)
    return text, applied


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="APIGen-MT-5k.csv")
    parser.add_argument("--output", default="noised_APIGen.csv")
    parser.add_argument("--n",      type=int, default=200,
                        help="Total rows to keep (default 200)")
    parser.add_argument("--noise",  type=int, default=None,
                        help="Rows to corrupt (default: half of --n)")
    args = parser.parse_args()

    noise_count = args.noise if args.noise is not None else args.n  # default: all rows

    print(f"[NOISE] Loading {args.input} ...")
    df = pd.read_csv(args.input)
    df = df.head(args.n).copy().reset_index(drop=True)
    print(f"[NOISE] Took top {len(df)} rows  |  will corrupt {noise_count} of them")

    corrupt_idx = set(random.sample(range(len(df)), noise_count))
    text_cols   = ["conversations", "tools", "system"]

    noise_log = []
    for idx in corrupt_idx:
        row_applied = {}
        for col in text_cols:          # ALL 3 columns corrupted per noisy row
            original = str(df.at[idx, col])
            n_hits   = random.randint(3, 5)      # stack 3-5 functions
            noised, applied = apply_noise(original, n_corruptions=n_hits)
            df.at[idx, col] = noised
            row_applied[col] = applied
        noise_log.append({"row": idx, "columns": row_applied})

    df.to_csv(args.output, index=False)

    # Summary
    from collections import Counter
    fn_counts: Counter = Counter()
    for entry in noise_log:
        for col_applied in entry["columns"].values():
            for fn in col_applied:
                fn_counts[fn] += 1

    print(f"\n[NOISE] Saved -> {args.output}  ({len(df)} rows, {noise_count} corrupted, all 3 cols each)\n")
    print(f"  {'Noise type':<30}  {'Times applied':>13}")
    print(f"  {'-'*30}  {'-'*13}")
    for fn, cnt in fn_counts.most_common():
        print(f"  {fn:<30}  {cnt:>13}")

    print(f"\n  Sample (first 3 corrupted rows):")
    for entry in noise_log[:3]:
        print(f"    row {entry['row']:>4}")
        for col, applied in entry["columns"].items():
            print(f"      {col:<16} <- {' + '.join(applied)}")


if __name__ == "__main__":
    main()
