"""Evaluation-protocol v2 tests (audit findings D2 substring false hits,
D3 polarity never used, D5 no baseline).

These tests are written against the *desired* protocol and fail on the v1
evaluator, which scored with ``keyword in answer.lower()`` (raw substring) and
ignored the item's ``polarity`` field entirely.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nanollm.eval_protocol import (  # noqa: E402
    build_report,
    detect_polarity,
    keyword_hit,
    keyword_lookup_baseline,
    load_items,
    score_item,
)

QA_DIR = ROOT / "data" / "qa"


# --------------------------------------------------------------------------
# D2 - word-boundary matching: substring hits must not count...
# --------------------------------------------------------------------------


def test_substring_is_not_a_keyword_hit():
    """gold ``cook`` must NOT be satisfied by ``cooking`` (audit: 3/17 of the
    published hits were pure substrings)."""
    item = {"question": "Can morels be eaten raw?", "keywords": ["cook"], "polarity": "no"}
    result = score_item(item, "Raw morels need cooking before they are safe.")
    assert result["hit"] is False, result
    assert result["hits"] == []


def test_unrelated_substring_is_not_a_keyword_hit():
    item = {"question": "Does cooking destroy amatoxins?", "keywords": ["no"], "polarity": None}
    # "poisonous" and "North America" contain "no" as a substring but not as a word
    assert score_item(item, "The mushroom is poisonous.")["hit"] is False
    assert score_item(item, "It is common in North America.")["hit"] is False


def test_true_word_hit_still_counts():
    """Reverse check (guards against over-tightening): a real standalone hit
    must still be counted."""
    item = {"question": "Can morels be eaten raw?", "keywords": ["cook"], "polarity": "no"}
    assert score_item(item, "You must cook morels thoroughly.")["hit"] is True
    assert score_item(item, "No, cook them first.")["hit"] is True

    item_no = {"question": "Does cooking destroy amatoxins?", "keywords": ["no"], "polarity": None}
    hit, hits = keyword_hit("No. Amatoxins survive boiling.", item_no["keywords"])
    assert hit is True and hits == ["no"]


def test_multiword_keyword_matches_on_word_boundaries():
    item = {"keywords": ["poison control", "green-spored"], "polarity": None}
    assert keyword_hit("Call a poison control center immediately.", item["keywords"])[0] is True
    assert keyword_hit("The green-spored parasol grows on lawns.", item["keywords"])[0] is True
    # "poison" alone is not "poison control"
    assert keyword_hit("It is poisonous.", item["keywords"])[0] is False


def test_explicit_variants_can_be_declared_in_gold():
    """Gold may declare acceptable surface forms explicitly (``a|b|c``) - the
    matcher itself stays strict, nothing is stemmed silently."""
    item = {"keywords": ["cook|cooked|cooking"], "polarity": None}
    assert keyword_hit("Morels must be thoroughly cooked.", item["keywords"])[0] is True
    assert keyword_hit("You should cook them.", item["keywords"])[0] is True
    assert keyword_hit("They are poisonous.", item["keywords"])[0] is False


def test_morphological_sensitivity_is_labeled_and_separate_from_the_headline():
    """Secondary, clearly-labelled view for the obvious reviewer question
    ("gold says *vomit*, the answer says *vomiting* - is that not a hit?"):
    a documented regular-inflection rule is reported *alongside* the strict
    number, never instead of it.  It must still reject unrelated substrings."""
    from nanollm.eval_protocol import keyword_hit_inflected

    assert keyword_hit("delayed vomiting", ["vomit"])[0] is False          # strict
    assert keyword_hit_inflected("delayed vomiting", ["vomit"])[0] is True  # sensitivity
    assert keyword_hit_inflected("raw morels need cooking", ["cook"])[0] is True
    assert keyword_hit_inflected("Amatoxins survive boiling", ["amatoxin"])[0] is True
    # unrelated substring matches must stay rejected even in the loose view
    assert keyword_hit_inflected("The mushroom is poisonous", ["no"])[0] is False
    assert keyword_hit_inflected("It grows in North America", ["no"])[0] is False
    assert keyword_hit_inflected("the flesh is inedible", ["edible"])[0] is False
    assert keyword_hit_inflected("during the rain", ["ring"])[0] is False

    item = {"question": "What is the first phase?", "keywords": ["vomit"], "polarity": None}
    result = score_item(item, "Violent vomiting and diarrhea.")
    assert result["hit"] is False                    # the reported headline
    assert result["hit_inflection"] is True          # the labelled sensitivity
    assert result["correct"] is False

    report = build_report(load_items(), ["" for _ in load_items()])
    assert "hit_any_morphological_sensitivity" in report
    assert report["hit_any_morphological_sensitivity"] == 0.0


# --------------------------------------------------------------------------
# D3 - polarity: an answer that flips the stance must not count as correct
# --------------------------------------------------------------------------


def test_reversed_polarity_is_not_correct():
    item = {
        "question": "Would it be safe to eat the death cap?",
        "keywords": ["poisonous"],
        "polarity": "no",
    }
    result = score_item(item, "It is poisonous, but yes, it is safe and edible to eat.")
    assert result["hit"] is True           # the keyword is genuinely present
    assert result["polarity_expected"] == "no"
    assert result["polarity_detected"] == "yes"
    assert result["polarity_ok"] is False
    assert result["correct"] is False      # ... but the answer contradicts the question


def test_negated_affirmation_counts_as_negative_stance():
    """'No, chanterelles are not edible' must not read as 'edible'."""
    assert detect_polarity("No, chanterelles are not edible.") == "no"
    assert detect_polarity("They are not safe to eat.") == "no"


def test_aligned_polarity_still_scores_correct():
    """Reverse check: an answer whose stance matches the question stays correct."""
    item = {
        "question": "Would it be safe to eat the death cap?",
        "keywords": ["poisonous"],
        "polarity": "no",
    }
    result = score_item(item, "No, the death cap is poisonous and should never be eaten.")
    assert result["hit"] is True
    assert result["polarity_ok"] is True
    assert result["correct"] is True

    yes_item = {
        "question": "Are golden chanterelles good to eat?",
        "keywords": ["edible"],
        "polarity": "yes",
    }
    assert score_item(yes_item, "Yes, the golden chanterelle is a choice edible mushroom.")["correct"] is True
    assert score_item(yes_item, "No, chanterelles are not edible.")["correct"] is False


def test_unspecified_polarity_is_never_penalised():
    item = {"question": "What color are the gills?", "keywords": ["white"], "polarity": None}
    result = score_item(item, "The gills are white.")
    assert result["hit"] is True
    assert result["polarity_ok"] is None
    assert result["correct"] is True


# --------------------------------------------------------------------------
# D5 - the report must carry non-trivial baselines and a small-n interval
# --------------------------------------------------------------------------


def test_report_includes_baselines_and_ci():
    items = load_items()
    assert len(items) == 44
    report = build_report(items, ["" for _ in items], train_jsonl=QA_DIR / "qa_train.jsonl")
    assert report["n"] == 44
    assert report["baselines"]["empty"]["hit_any"] == 0.0
    assert 0.0 <= report["baselines"]["keyword_lookup"]["hit_any"] <= 1.0
    lo, hi = report["hit_any_ci95"]
    assert 0.0 <= lo <= report["hit_any"] <= hi <= 1.0
    # n=44 is a tiny sample: even a coin-flip rate must carry a wide interval
    from nanollm.eval_protocol import wilson_ci
    flo, fhi = wilson_ci(22, 44)
    assert fhi - flo > 0.25, "the n=44 interval must be reported, not a bare point estimate"


def test_keyword_lookup_baseline_is_deterministic_and_uses_train_answers():
    items = load_items()
    train_path = QA_DIR / "qa_train.jsonl"
    answers = keyword_lookup_baseline(items, train_path)
    assert len(answers) == len(items)
    assert all(isinstance(a, str) and a.strip() for a in answers)
    assert answers == keyword_lookup_baseline(items, train_path)

    train_answers = {
        json.loads(line)["answer"]
        for line in train_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    # every baseline answer must be a stored training answer (a pure lookup table)
    assert set(answers) <= train_answers


def test_report_keeps_legacy_v1_scoring_for_comparison():
    """The v1 (raw substring) metric is still computed on the same answers, so
    the published 38.6% can be compared against v2 apples-to-apples."""
    items = load_items()
    answers = [""] * len(items)
    # item 6 is "Can morels be eaten raw?" with gold keyword "cook": v1 counts
    # the substring inside "cooking", v2 does not.
    answers[6] = "Raw morels need cooking before they are safe."
    report = build_report(items, answers)
    assert report["protocol"] == "v2"
    assert report["v1_scoring_on_same_answers"]["hit_any"] > 0.0
    assert report["v1_scoring_on_same_answers"]["substring_only_hits"] == [
        {"question": items[6]["question"], "keywords": ["cook"], "surfaces": {"cook": ["cooking"]}}
    ]
    # the same item is still a v2 hit - but only through a genuine word ("raw"),
    # while v1 additionally credited the substring inside "cooking"
    row = report["rows"][6]
    assert row["hit"] is True and row["hits"] == ["raw"]
    assert row["v1_hits"] == ["cook", "raw"]
    assert report["hit_any"] == 1 / len(items)
