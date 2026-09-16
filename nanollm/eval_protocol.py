"""Evaluation protocol v2 for the held-out mushroom-safety QA set.

Why v2 exists (audit findings, each fixed here):

- **D2 - substring false hits.** v1 scored ``keyword in answer.lower()``, so
  ``cook`` was "found" in *cooking*, ``no`` in *pois**no**us* and *North
  America*: 3 of the 17 published hits were pure substrings.  v2 matches on
  **word boundaries** (token sequences).  Nothing is stemmed silently: gold may
  declare acceptable surface forms explicitly with ``a|b|c``.
- **D3 - polarity ignored.** Every golden item carries a ``polarity`` field
  (``"yes"``/``"no"``/``None``) and v1 never read it, so an answer that flipped
  the stance ("the death cap is safe to eat") scored the same as a correct one.
  v2 detects the answer's stance (with negation scope, so *"not edible"* is a
  negative stance) and records a violation when it contradicts the item.
- **D5 - no baseline, no interval.** The report always carries the "empty
  answer" floor and a lexical **keyword-lookup baseline** (answer the question
  with the stored answer of the most similar *training* question), plus a
  Wilson 95% confidence interval - with n=44 a point estimate alone is not a
  result.

The v1 metric is still computed on the same answers
(``v1_scoring_on_same_answers``) so old and new numbers can be compared
apples-to-apples instead of being silently replaced.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Iterable, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_JSONL = ROOT / "data" / "qa" / "qa_eval.jsonl"
DEFAULT_TRAIN_JSONL = ROOT / "data" / "qa" / "qa_train.jsonl"

CATEGORIES: list[tuple[str, slice]] = [
    ("edibility", slice(0, 10)),
    ("identification", slice(10, 18)),
    ("habitat", slice(18, 26)),
    ("symptoms", slice(26, 34)),
    ("first-aid/general", slice(34, 44)),
]

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

# ---------------------------------------------------------------------------
# word-boundary keyword matching (D2)
# ---------------------------------------------------------------------------


def tokenize(text: str) -> list[str]:
    """Lowercase word/number tokens (apostrophes kept: ``don't`` -> ``don't``)."""
    return TOKEN_RE.findall(text.lower())


def keyword_hit(answer: str, keywords: Sequence[str]) -> tuple[bool, list[str]]:
    """Word-boundary keyword matching.

    A keyword is a token sequence (``"poison control"`` must appear as those two
    consecutive words).  ``|`` declares explicit acceptable variants
    (``"cook|cooked|cooking"``); the matcher itself never stems or substrings, so
    gold ``cook`` is *not* satisfied by ``cooking``.
    """
    answer_tokens = tokenize(answer)
    hits: list[str] = []
    for keyword in keywords:
        for variant in keyword.split("|"):
            variant_tokens = tokenize(variant)
            if variant_tokens and _contains_sequence(answer_tokens, variant_tokens):
                hits.append(keyword)
                break
    return (len(hits) > 0, hits)


def _contains_sequence(haystack: list[str], needle: list[str]) -> bool:
    n = len(needle)
    if n == 0 or n > len(haystack):
        return False
    first = needle[0]
    for i in range(len(haystack) - n + 1):
        if haystack[i] == first and haystack[i : i + n] == needle:
            return True
    return False


# Regular English inflections, used ONLY by the clearly-labelled sensitivity
# view below (never by the headline number).
INFLECTION_RE = re.compile(r"(?:s|es|ed|d|ing)?$")
MIN_STEM_LEN = 4  # "no" must not match "known"; "raw" must not match "rawhide"


def keyword_hit_inflected(answer: str, keywords: Sequence[str]) -> tuple[bool, list[str]]:
    """**Sensitivity view, not the headline metric.**

    Same word-boundary matcher, but a gold word of at least :data:`MIN_STEM_LEN`
    characters also accepts its regular inflections (``vomit`` -> *vomiting*,
    ``amatoxin`` -> *amatoxins*).  It exists to answer the obvious reviewer
    question - "your strict rule is under-counting inflected answers" - with a
    number, while the reported headline stays strict.  It cannot resurrect the
    v1 errors, because the variant must start with the gold word (``edible``
    does not match *inedible*) and short words are never expanded (``no`` does
    not match *known* or *poisonous*)."""
    answer_tokens = tokenize(answer)
    hits: list[str] = []
    for keyword in keywords:
        for variant in keyword.split("|"):
            variant_tokens = tokenize(variant)
            if not variant_tokens:
                continue
            if _contains_sequence(answer_tokens, variant_tokens):
                hits.append(keyword)
                break
            if len(variant_tokens) == 1 and len(variant_tokens[0]) >= MIN_STEM_LEN:
                stem = variant_tokens[0]
                if INFLECTION_RE.fullmatch(stem):  # already an inflected gold form
                    continue
                if any(
                    tok != stem and tok.startswith(stem) and INFLECTION_RE.fullmatch(tok[len(stem):])
                    for tok in answer_tokens
                ):
                    hits.append(keyword)
                    break
    return (len(hits) > 0, hits)


# ---------------------------------------------------------------------------
# polarity / stance (D3)
# ---------------------------------------------------------------------------

NEGATION_TOKENS = {
    "no", "not", "never", "cannot", "cant", "can't", "dont", "don't",
    "doesnt", "doesn't", "isnt", "isn't", "arent", "aren't", "wasnt", "wasn't",
    "unsafe", "avoid", "without", "nor",
}
AFFIRMATION_PHRASES: list[tuple[str, ...]] = [
    ("yes",), ("safe",), ("safely",), ("edible",), ("choice",),
    ("good", "to", "eat"), ("ok",), ("okay",), ("fine",), ("non-toxic",),
]
NEGATION_SCOPE = 3  # a negation this close before an affirmation flips its stance


def detect_polarity(answer: str) -> Optional[str]:
    """Stance of an answer: ``"yes"`` (affirms), ``"no"`` (denies), or ``None``
    when the answer carries no stance cue (counted as *unverified*, never as a
    violation).  Negation scope is respected: *"not edible"* is a negative
    stance, not a positive one."""
    tokens = tokenize(answer)
    negation_idx = [i for i, tok in enumerate(tokens) if tok in NEGATION_TOKENS]
    affirmation_idx: list[int] = []
    for phrase in AFFIRMATION_PHRASES:
        for i in range(len(tokens) - len(phrase) + 1):
            if tuple(tokens[i : i + len(phrase)]) == phrase:
                # dropped when negated just before it ("not safe", "never edible")
                if not any(n < i <= n + NEGATION_SCOPE for n in negation_idx):
                    affirmation_idx.append(i)

    has_neg = bool(negation_idx)
    has_pos = bool(affirmation_idx)
    if has_neg and not has_pos:
        return "no"
    if has_pos and not has_neg:
        return "yes"
    return None


def polarity_ok(answer: str, expected: Optional[str]) -> Optional[bool]:
    """``True``/``False`` when the stance is both expected and detected,
    ``None`` when the item has no polarity or the answer has no stance cue."""
    if expected is None:
        return None
    detected = detect_polarity(answer)
    if detected is None:
        return None
    return detected == expected


# ---------------------------------------------------------------------------
# per-item scoring
# ---------------------------------------------------------------------------


def score_item(item: dict, answer: str) -> dict:
    """Score one answer under protocol v2 (plus the v1 numbers for comparison)."""
    keywords = list(item.get("keywords", []))
    hit, hits = keyword_hit(answer, keywords)
    hit_infl, hits_infl = keyword_hit_inflected(answer, keywords)
    detected = detect_polarity(answer)
    ok = polarity_ok(answer, item.get("polarity"))

    v1_low = answer.lower()
    v1_hits = [kw for kw in keywords if kw.lower() in v1_low]
    words = tokenize(answer)

    return {
        "question": item.get("question", ""),
        "answer": answer,
        "keywords": keywords,
        "hits": hits,
        "hit": hit,
        "hit_all": bool(keywords) and len(hits) == len(keywords),
        "hit_inflection": hit_infl,
        "hits_inflection": hits_infl,
        "v1_hits": v1_hits,
        "v1_hit": len(v1_hits) > 0,
        "substring_only_hits": [kw for kw in v1_hits if kw not in hits],
        # which surface form made a substring "hit" (evidence for the audit trail)
        "substring_surfaces": {
            kw: sorted({w for w in words if kw.lower() in w and w != kw.lower()})
            for kw in v1_hits if kw not in hits
        },
        "polarity_expected": item.get("polarity"),
        "polarity_detected": detected,
        "polarity_ok": ok,
        # an answer must contain the fact AND not contradict the question
        "correct": bool(hit) and ok is not False,
    }


def score_items(items: Sequence[dict], answers: Sequence[str]) -> list[dict]:
    if len(items) != len(answers):
        raise ValueError(f"{len(items)} items vs {len(answers)} answers")
    return [score_item(item, answer) for item, answer in zip(items, answers)]


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval - the honest interval for small n (n=44 here)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo = centre - half
    return (0.0 if lo < 1e-9 else lo, min(1.0, centre + half))


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


def aggregate(rows: Sequence[dict], categories: Sequence[tuple[str, slice]] = CATEGORIES) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0, "hit_any": 0.0, "hit_all": 0.0, "correct": 0.0}
    hits = sum(r["hit"] for r in rows)
    hits_all = sum(r["hit_all"] for r in rows)
    correct = sum(r["correct"] for r in rows)
    inflected = sum(r["hit_inflection"] for r in rows)
    violations = [
        {"question": r["question"], "expected": r["polarity_expected"],
         "detected": r["polarity_detected"], "answer": r["answer"]}
        for r in rows if r["polarity_ok"] is False
    ]
    unverified = sum(1 for r in rows if r["polarity_ok"] is None and r["polarity_expected"] is not None)
    report = {
        "n": n,
        "hit_any": hits / n,
        "hit_all": hits_all / n,
        "correct": correct / n,
        "hit_any_ci95": list(wilson_ci(hits, n)),
        "correct_ci95": list(wilson_ci(correct, n)),
        "hits": hits,
        # labelled sensitivity: same matcher + regular inflections (NOT the headline)
        "hit_any_morphological_sensitivity": inflected / n,
        "hits_morphological_sensitivity": inflected,
        "polarity_violations": violations,
        "polarity_unverified": unverified,
        "v1_scoring_on_same_answers": {
            "hit_any": sum(r["v1_hit"] for r in rows) / n,
            "hit_all": sum(len(r["v1_hits"]) == len(r["keywords"]) and bool(r["keywords"]) for r in rows) / n,
            "substring_only_hits": [
                {"question": r["question"], "keywords": r["substring_only_hits"],
                 "surfaces": r.get("substring_surfaces", {})}
                for r in rows if r["substring_only_hits"]
            ],
        },
    }
    cat_stats = {}
    for name, sl in categories:
        sub = rows[sl]
        if not sub:
            continue
        cat_stats[name] = {
            "n": len(sub),
            "hit_any": sum(r["hit"] for r in sub) / len(sub),
            "hit_all": sum(r["hit_all"] for r in sub) / len(sub),
            "correct": sum(r["correct"] for r in sub) / len(sub),
        }
    report["categories"] = cat_stats
    return report


# ---------------------------------------------------------------------------
# baselines (D5)
# ---------------------------------------------------------------------------


def empty_baseline(items: Sequence[dict]) -> list[str]:
    """Floor: the model that never answers."""
    return ["" for _ in items]


STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "of", "in", "on", "at",
    "to", "for", "with", "and", "or", "do", "does", "did", "can", "could",
    "what", "which", "who", "why", "how", "when", "where", "it", "its", "you",
    "your", "i", "we", "they", "them", "this", "that", "these", "those", "not",
    "if", "should", "would", "must", "after", "before", "from", "by", "as",
}


def _content_tokens(text: str) -> set[str]:
    return {t for t in tokenize(text) if t not in STOPWORDS}


def keyword_lookup_baseline(items: Sequence[dict], train_jsonl: Path | str = DEFAULT_TRAIN_JSONL) -> list[str]:
    """Non-trivial oracle-free baseline: a **keyword lookup table**.

    Each eval question is answered with the stored answer of the *most similar
    training question* (content-word overlap on the question, ties broken by
    Jaccard, then by file order - fully deterministic).  It uses no model at
    all: if the fine-tuned model cannot beat it, the model is not adding
    knowledge, only phrasing.
    """
    train_path = Path(train_jsonl)
    pairs = [
        json.loads(line)
        for line in train_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train_sets = [_content_tokens(p["question"]) for p in pairs]

    answers: list[str] = []
    for item in items:
        q = _content_tokens(item["question"])
        best_score = (-1.0, -1.0)
        best_answer = ""
        for pair, train_set in zip(pairs, train_sets):
            if not q or not train_set:
                continue
            covered = len(q & train_set) / len(q)
            jaccard = len(q & train_set) / len(q | train_set)
            score = (covered, jaccard)
            if score > best_score:
                best_score, best_answer = score, pair["answer"]
        answers.append(best_answer)
    return answers


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def load_items(path: Path | str = DEFAULT_EVAL_JSONL) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_report(
    items: Sequence[dict],
    answers: Sequence[str],
    train_jsonl: Path | str | None = DEFAULT_TRAIN_JSONL,
    include_baselines: bool = True,
    categories: Sequence[tuple[str, slice]] = CATEGORIES,
) -> dict:
    """Full protocol-v2 report for one set of answers (one seed / one device)."""
    rows = score_items(items, answers)
    report = aggregate(rows, categories)
    report["protocol"] = "v2"
    report["scoring"] = "word-boundary keyword match + polarity check"
    report["rows"] = rows

    if include_baselines:
        baselines = {"empty": aggregate(score_items(items, empty_baseline(items)), categories)}
        if train_jsonl is not None and Path(train_jsonl).exists():
            lookup_answers = keyword_lookup_baseline(items, train_jsonl)
            lookup = aggregate(score_items(items, lookup_answers), categories)
            lookup["answers"] = lookup_answers
            baselines["keyword_lookup"] = lookup
        report["baselines"] = baselines
    return report


def summarize_seeds(reports: Sequence[dict]) -> dict:
    """Mean / spread across seeds + pooled interval (multi-seed reporting)."""
    if not reports:
        return {}
    rates = [r["hit_any"] for r in reports]
    correct = [r["correct"] for r in reports]
    n = reports[0]["n"]
    pooled_hits = sum(r["hits"] for r in reports)
    pooled_n = n * len(reports)
    mean = sum(rates) / len(rates)
    var = sum((r - mean) ** 2 for r in rates) / len(rates)
    return {
        "seeds": len(reports),
        "n_per_seed": n,
        "per_seed_hit_any": rates,
        "per_seed_correct": correct,
        "mean_hit_any": mean,
        "stdev_hit_any": math.sqrt(var),
        "min_hit_any": min(rates),
        "max_hit_any": max(rates),
        "pooled_hit_any": pooled_hits / pooled_n,
        "pooled_hit_any_ci95": list(wilson_ci(pooled_hits, pooled_n)),
        "mean_correct": sum(correct) / len(correct),
    }


def format_report(report: dict, show_failures: int = 3) -> str:
    lines = []
    n = report["n"]
    lo, hi = report["hit_any_ci95"]
    lines.append(f"== protocol v2 (word-boundary match + polarity) ==")
    lines.append(f"keyword hit (any expected keyword): {report['hit_any']*100:.1f}%  "
                 f"({report['hits']}/{n})   95% CI [{lo*100:.1f}, {hi*100:.1f}]")
    lines.append(f"keyword hit (all expected)        : {report['hit_all']*100:.1f}%")
    lines.append(f"correct (hit and polarity not contradicted): {report['correct']*100:.1f}%")
    lines.append(f"polarity violations: {len(report['polarity_violations'])}   "
                 f"unverified stance: {report['polarity_unverified']}")
    lines.append(f"[sensitivity only] regular inflections counted: "
                 f"{report['hit_any_morphological_sensitivity']*100:.1f}% "
                 f"({report['hits_morphological_sensitivity']}/{n})")
    v1 = report["v1_scoring_on_same_answers"]
    lines.append(f"-- v1 scoring on the SAME answers: hit_any {v1['hit_any']*100:.1f}% --")
    for row in v1["substring_only_hits"]:
        surfaces = ", ".join(f"{k}->{v}" for k, v in (row.get("surfaces") or {}).items())
        lines.append(f"   substring-only 'hit' (not a v2 hit): {row['keywords']}"
                     + (f"  [{surfaces}]" if surfaces else "")
                     + f" <- {row['question']}")
    for name, st in report["categories"].items():
        lines.append(f"  {name:<18} n={st['n']:>2}  hit_any={st['hit_any']*100:5.1f}%  "
                     f"hit_all={st['hit_all']*100:5.1f}%  correct={st['correct']*100:5.1f}%")
    for name, bl in report.get("baselines", {}).items():
        lines.append(f"  baseline[{name}]: hit_any={bl['hit_any']*100:.1f}%  correct={bl['correct']*100:.1f}%")
    for row in report["polarity_violations"]:
        lines.append(f"  POLARITY VIOLATION (expected {row['expected']}, detected {row['detected']}): "
                     f"{row['question']} -> {row['answer'][:90]!r}")
    if show_failures > 0:
        fails = [r for r in report["rows"] if not r["hit"]]
        lines.append(f"-- {len(fails)} complete failures (no expected keyword) --")
        for r in fails[:show_failures]:
            lines.append(f"  Q: {r['question']}")
            lines.append(f"  A: {r['answer'][:140]}")
            lines.append(f"  expected: {r['keywords']}")
    return "\n".join(lines)
