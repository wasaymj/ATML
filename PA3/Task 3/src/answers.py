"""Answer extraction, normalization, answer groups and grading.

* MATH: answer groups and grading by math-verify mathematical equivalence.
* MathVista: the official answer-type / choice / precision rules (port of `normalize_extracted_answer` from the
  MathVista repository), with multiple-choice answers represented as option letters.
* Missing extractions form one NO_ANSWER group that is always graded incorrect and stays in voting and cost.
* Groups are formed from the 32 extracted answers of a question only (gold is never consulted); grading then
  applies the same equivalence rule to (gold, group representative).
"""
import os
import re
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

NO_ANSWER = "<NO_ANSWER>"
SCORER_VERSION = "task3-scorer-1.1"     # 1.1: math-verify 0.9.0 (as linked by the manual); reported with results


# ----------------------------------------------------------------------------------------------------------
# Extraction (live generation; the stored rollouts use the supplied extracted_answer)
# ----------------------------------------------------------------------------------------------------------
def extract_last_boxed(text: str) -> Optional[str]:
    """README rule: find the last literal '\\boxed{', scan balanced braces, keep nested LaTeX, strip whitespace.
    An absent, empty or unclosed last box gives None; there is no numeric fallback."""
    key = "\\boxed{"
    start = text.rfind(key)
    if start < 0:
        return None
    i = start + len(key)
    depth = 1
    j = i
    while j < len(text):
        ch = text[j]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                inner = text[i:j].strip()
                return inner if inner else None
        j += 1
    return None                                  # unclosed


# ----------------------------------------------------------------------------------------------------------
# MATH: math-verify equivalence
# ----------------------------------------------------------------------------------------------------------
def _with_timeout(fn, seconds: float, default):
    """Run fn() in a worker thread; return (`default`, True) if it does not finish within `seconds`.

    math-verify's own timeouts use signals (main thread, Linux only) or subprocesses, so the same thread-based
    limit is used on every OS. A timed-out call is stopped by raising TimeoutError inside the worker thread
    (sympy is pure Python, so the exception is delivered at the next bytecode boundary)."""
    import ctypes
    import threading
    box = {}

    def run():
        try:
            box["v"] = fn()
        except BaseException:
            box["v"] = default
    th = threading.Thread(target=run, daemon=True)
    th.start()
    th.join(seconds)
    if not th.is_alive():
        return box.get("v", default), False
    for _ in range(50):                                  # stop the stuck computation
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(th.ident), ctypes.py_object(TimeoutError))
        th.join(0.1)
        if not th.is_alive():
            break
    return default, True


class MathEquivalence:
    """math-verify based equivalence with parse and pair caches.

    Answers are parsed as the content of a \\boxed{} (the form they were extracted from). Each parse and each
    verify call is limited to `timeout` seconds (default 5 s, math-verify's own default); a call that times out
    counts as "not equivalent" / "unparseable" and is recorded in `self.timeouts`."""

    def __init__(self, timeout: float = 5.0):
        from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig
        self._parse, self._verify = parse, verify
        self._cfg = [LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig()]
        self.timeout = timeout
        self._cache: Dict[str, Any] = {}
        self._pairs: Dict[Tuple[str, str], bool] = {}
        self.timeouts: List[Tuple[str, str]] = []
        self.skipped: List[str] = []
        logging.getLogger("math_verify").setLevel(logging.ERROR)
        logging.getLogger("latex2sympy2_extended").setLevel(logging.ERROR)

    @staticmethod
    def too_complex(s: str) -> bool:
        """Answers sympy cannot evaluate in bounded time and that a thread exception cannot interrupt (C-level
        big-integer powers): power towers (an exponent containing '^'), numeric exponents with >= 5 digits, or
        very long strings. They are compared as whitespace-normalized strings only (counted in `skipped`)."""
        if len(s) > 300 or re.search(r"\^\s*\{?\s*\d{5,}", s):
            return True
        for m in re.finditer(r"\^\s*\{", s):              # brace-aware: any '^' inside an exponent group
            depth, j = 1, m.end()
            while j < len(s) and depth:
                depth += {"{": 1, "}": -1}.get(s[j], 0)
                j += 1
            if "^" in s[m.end():j]:
                return True
        return False

    def parsed(self, s: str):
        if s not in self._cache and self.too_complex(s):
            self.skipped.append(s)
            self._cache[s] = []
        if s not in self._cache:
            v, timed_out = _with_timeout(lambda: self._parse("\\boxed{" + s + "}", extraction_config=self._cfg,
                                                             parsing_timeout=None), self.timeout, [])
            if timed_out:
                self.timeouts.append(("parse", s))
            self._cache[s] = v
        return self._cache[s]

    @staticmethod
    def _plain(s: str) -> str:
        return re.sub(r"\s+", "", s)

    def equal(self, a: str, b: str) -> bool:
        """Equivalence used for both grouping (a = group representative) and grading (a = gold)."""
        if self._plain(a) == self._plain(b):
            return True
        key = (a, b)
        if key not in self._pairs:
            pa, pb = self.parsed(a), self.parsed(b)
            if not pa or not pb:
                self._pairs[key] = False
            else:
                v, timed_out = _with_timeout(lambda: bool(self._verify(pa, pb, timeout_seconds=None)),
                                             self.timeout, False)
                if timed_out:
                    self.timeouts.append(("verify", f"{a} || {b}"))
                self._pairs[key] = v
        return self._pairs[key]


def group_math_answers(answers: Sequence[Optional[str]], eq: MathEquivalence) -> Tuple[List[int], List[str]]:
    """Answer groups for one question. Unique strings are visited in sorted order (independent of rollout
    order and of gold); each joins the first existing group whose representative is equivalent, otherwise it
    starts a new group. Returns (group id per rollout, representative per group); NO_ANSWER is its own group."""
    reps: List[str] = []
    key_of: Dict[str, int] = {}
    for s in sorted({a for a in answers if a is not None}):
        for gid, r in enumerate(reps):
            if r != NO_ANSWER and eq.equal(r, s):
                key_of[s] = gid
                break
        else:
            key_of[s] = len(reps)
            reps.append(s)
    if any(a is None for a in answers):
        key_of[NO_ANSWER] = len(reps)
        reps.append(NO_ANSWER)
    return [key_of[NO_ANSWER if a is None else a] for a in answers], reps


# ----------------------------------------------------------------------------------------------------------
# MathVista: official normalization rules
# ----------------------------------------------------------------------------------------------------------
def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


_NUM = r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_RANGE = re.compile(r"\d\s*(?:-|--|\\text\{\s*-\s*\}|to)\s*\d")


def _strip_latex_units(s: str) -> str:
    s = re.sub(r"\\(?:text|mathrm|textbf|mbox|operatorname)\s*\{[^{}]*\}", " ", s)
    for tok in ("\\,", "\\;", "\\!", "\\ ", "~", "$", "\\%", "%", "^{\\circ}", "^\\circ", "\\circ", "\\degree",
                "\\left", "\\right"):
        s = s.replace(tok, " ")
    return s.strip()


def _to_number(s: str, eq: Optional[MathEquivalence]) -> Optional[float]:
    """Deterministic LaTeX/plain string -> float (impl). Ranges ('10-15 years') and multi-value strings are
    not numbers. Order: plain float -> leading number followed only by unit words -> math-verify value."""
    s0 = s.strip()
    try:
        return float(s0.replace(",", ""))
    except ValueError:
        pass
    if _RANGE.search(s0) or re.search(r"\band\b|\\text\{\s*and\s*\}", s0):
        return None
    s1 = _strip_latex_units(s0)
    m = re.fullmatch(rf"\s*({_NUM})\s*([A-Za-z][A-Za-z\s/.\^0-9]*)?\s*", s1)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            pass
    if eq is None:
        return None
    for cand in (s1, s0):
        try:
            p = eq.parsed(cand)
            if p:
                v = p[0]
                if getattr(v, "is_number", False) and v.is_real and v.is_finite:
                    return float(v)
        except Exception:
            continue
    return None


def _num_str(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else str(x)


def _to_list(s: str) -> Optional[str]:
    nums = re.findall(_NUM, _strip_latex_units(s))
    if not nums:
        return None
    try:
        return str([int(float(n.replace(",", ""))) if float(n.replace(",", "")).is_integer()
                    else float(n.replace(",", "")) for n in nums])
    except ValueError:
        return None


class MathVistaNormalizer:
    """Port of MathVista's `normalize_extracted_answer` with option letters as the multiple-choice key.

    multi_choice : '(A) text' -> 'A'; a valid letter is kept; choice text maps to its letter; anything else maps
                   to the most similar choice by Levenshtein distance (official rule).
    integer      : str(int(float(x)))           (official; truncation)
    float        : str(round(float(x), precision))
    list         : canonical '[a, b, ...]' of the numbers in the string
    Free-form strings are first converted to a number by `_to_number` (impl: the export stores raw boxed LaTeX,
    while the official pipeline assumes a clean number). A non-null answer that cannot be normalized keeps its
    own group 'UNPARSED:<text>' (never correct); a null extraction is NO_ANSWER."""

    def __init__(self, metadata: Dict[str, Dict[str, Any]], eq: Optional[MathEquivalence] = None):
        self.meta = metadata
        self.eq = eq
        self.stats = {"mc_letter": 0, "mc_paren": 0, "mc_choice_text": 0, "mc_most_similar": 0,
                      "ff_plain": 0, "ff_converted": 0, "ff_unparsed": 0, "no_answer": 0}

    def key(self, pid: str, extraction: Optional[str], count: bool = True) -> str:
        m = self.meta[pid]
        if extraction is None:
            self._bump("no_answer", count)
            return NO_ANSWER
        x = str(extraction).strip()
        if m["question_type"] == "multi_choice":
            choices = list(m["choices"])
            letters = [chr(ord("A") + i) for i in range(len(choices))]
            paren = re.findall(r"\(([a-zA-Z])\)", x)
            if paren:
                x = paren[0].upper()
                self._bump("mc_paren", count)
            if x in letters:
                self._bump("mc_letter", count)
                return x
            if x in choices:
                self._bump("mc_choice_text", count)
                return letters[choices.index(x)]
            d = [_levenshtein(x, c) for c in choices]
            self._bump("mc_most_similar", count)
            return letters[d.index(min(d))]
        at = m["answer_type"]
        if at == "list":
            v = _to_list(x)
            if v is None:
                self._bump("ff_unparsed", count)
                return "UNPARSED:" + re.sub(r"\s+", " ", x)
            self._bump("ff_converted", count)
            return v
        try:
            num = float(x.replace(",", ""))
            self._bump("ff_plain", count)
        except ValueError:
            num = _to_number(x, self.eq)
            if num is None:
                self._bump("ff_unparsed", count)
                return "UNPARSED:" + re.sub(r"\s+", " ", x)
            self._bump("ff_converted", count)
        try:
            if at == "integer":
                return str(int(num))
            if at == "float":
                return str(round(num, int(m["precision"])))
        except (ValueError, OverflowError):
            pass
        return _num_str(num)

    def gold_key(self, pid: str, gold: str) -> str:
        m = self.meta[pid]
        if m["question_type"] == "multi_choice":
            return gold.strip()                    # the export stores the gold option letter
        return self.key(pid, gold, count=False)

    def _bump(self, k: str, count: bool):
        if count:
            self.stats[k] += 1


def load_mathvista_metadata(path: str) -> Dict[str, Dict[str, Any]]:
    """Accepts the released testmini.json ({pid: record}) or the fallback download ({"items": [record, ...]})."""
    from .utils import load_json
    d = load_json(path)
    items = d["items"] if isinstance(d, dict) and "items" in d else (d.values() if isinstance(d, dict) else d)
    return {str(r["pid"]): r for r in items}


def fetch_mathvista_metadata(cfg: Dict[str, Any], out_path: str) -> str:
    """Download the official testmini parquet at the pinned revision and keep only the text columns."""
    import json
    import huggingface_hub as hub
    import pyarrow.parquet as pq
    mv = cfg["mathvista_metadata"]
    p = hub.hf_hub_download(mv["repo"], mv["file"], repo_type="dataset", revision=mv["revision"])
    cols = [c for c in pq.ParquetFile(p).schema_arrow.names if c not in ("image", "decoded_image")]
    items = pq.read_table(p, columns=cols).to_pylist()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"source": mv["repo"], "revision": mv["revision"], "file": mv["file"], "columns": cols,
                   "items": items}, f, ensure_ascii=False, indent=0)
    return out_path
