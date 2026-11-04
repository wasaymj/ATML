"""VQA answer normalization and soft consensus accuracy.

Python 3 port of the official VQA evaluation code (`vqaEval.py`, Aishwarya Agrawal, based on Tsung-Yi Lin's
coco-caption code; https://github.com/GT-Vision-Lab/VQA). The contraction table, number map, article list,
punctuation list and both regular expressions are copied unchanged, including the original `periodStrip`
pattern, so normalization matches the reference implementation token for token.

One deliberate difference: the official script normalizes only when the ten human answers are not all identical.
The manual asks us to "apply the same answer normalization throughout", so `normalize_answer` is always applied,
to predictions, references, consensus selection, grounding answers and pair diagnostics alike.
"""
import re
from collections import Counter
from typing import Sequence, Tuple

CONTRACTIONS = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't",
                "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't",
                "dont": "don't", "hadnt": "hadn't", "hadnt've": "hadn't've", "hadn'tve": "hadn't've",
                "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've",
                "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've",
                "I'dve": "I'd've", "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've",
                "it'dve": "it'd've", "itll": "it'll", "let's": "let's", "maam": "ma'am", "mightnt": "mightn't",
                "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
                "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've",
                "oclock": "o'clock", "oughtnt": "oughtn't", "ow's'at": "'ow's'at", "'ows'at": "'ow's'at",
                "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've",
                "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've",
                "shouldn'tve": "shouldn't've", "somebody'd": "somebodyd", "somebodyd've": "somebody'd've",
                "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", "somebodys": "somebody's",
                "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've",
                "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd",
                "somethingd've": "something'd've", "something'dve": "something'd've", "somethingll": "something'll",
                "thats": "that's", "thered": "there'd", "thered've": "there'd've", "there'dve": "there'd've",
                "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've",
                "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've",
                "twas": "'twas", "wasnt": "wasn't", "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've",
                "werent": "weren't", "whatll": "what'll", "whatre": "what're", "whats": "what's",
                "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's",
                "whereve": "where've", "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've",
                "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", "whyre": "why're",
                "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't",
                "wouldnt've": "wouldn't've", "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll",
                "y'allll": "y'all'll", "yall'd've": "y'all'd've", "y'alld've": "y'all'd've",
                "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've",
                "youll": "you'll", "youre": "you're", "youve": "you've"}
MANUAL_MAP = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
              "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
ARTICLES = ["a", "an", "the"]
PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")     # sic: copied verbatim from the official code
COMMA_STRIP = re.compile(r"(\d)(\,)(\d)")
PUNCT = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]


def process_punctuation(in_text: str) -> str:
    out_text = in_text
    for p in PUNCT:
        if (p + " " in in_text or " " + p in in_text) or (re.search(COMMA_STRIP, in_text) is not None):
            out_text = out_text.replace(p, "")
        else:
            out_text = out_text.replace(p, " ")
    return PERIOD_STRIP.sub("", out_text, re.UNICODE)


def process_digit_article(in_text: str) -> str:
    out = []
    for word in in_text.lower().split():
        word = MANUAL_MAP.get(word, word)
        if word not in ARTICLES:
            out.append(word)
    return " ".join(CONTRACTIONS.get(w, w) for w in out)


def normalize_answer(text: str) -> str:
    text = (text or "").replace("\n", " ").replace("\t", " ").strip()
    return process_digit_article(process_punctuation(text))


def soft_accuracy(pred: str, answers: Sequence[str]) -> float:
    """s(a) = 1/10 sum_i min(1, 1/3 sum_{j != i} 1[a = a_j]) over the ten human answers (all normalized)."""
    p = normalize_answer(pred)
    refs = [normalize_answer(a) for a in answers]
    accs = []
    for i in range(len(refs)):
        matches = sum(1 for j, r in enumerate(refs) if j != i and r == p)
        accs.append(min(1.0, matches / 3.0))
    return sum(accs) / len(accs) if accs else 0.0


def consensus(answers: Sequence[str]) -> Tuple[str, int, str]:
    """(normalized consensus answer, its agreement count, an original answer string from the consensus group).

    Ties are broken deterministically (highest count, then alphabetical) so the selection is reproducible. The
    original string is the most frequent raw spelling inside the consensus group."""
    norm = [normalize_answer(a) for a in answers]
    counts = Counter(norm)
    best = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    raw = Counter(a.strip() for a, n in zip(answers, norm) if n == best[0])
    raw_best = sorted(raw.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return best[0], best[1], raw_best


def exact_match(pred: str, ref: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(ref))


def answer_format_ok(pred_norm: str, kind: str) -> bool:
    """Whether a normalized prediction has the expected form (yes/no, or a count) for grounding diagnostics."""
    if kind in ("presence", "yes/no"):
        return pred_norm in ("yes", "no")
    if kind in ("counting", "number"):
        return pred_norm.isdigit()
    return len(pred_norm) > 0
