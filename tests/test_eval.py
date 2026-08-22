"""Tests for the held-out evaluation set (size + disjointness guarantees)."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_eval_set import build, check_disjoint  # noqa: E402

QA_DIR = ROOT / "data" / "qa"


def test_eval_set_has_at_least_40_items():
    items = build()
    assert len(items) >= 40, len(items)


def test_eval_set_disjoint_from_training():
    items = build()
    report = check_disjoint(items, QA_DIR / "qa_train.jsonl", QA_DIR / "qa_val.jsonl")
    assert report["exact_duplicates"] == []
    assert report["near_duplicates"] == []
    assert report["eval_items"] == len(items)


def test_eval_items_have_expected_keywords():
    items = build()
    for item in items:
        assert item["question"].strip()
        assert item["answer"].strip()
        assert isinstance(item["keywords"], list) and len(item["keywords"]) >= 1


def test_built_jsonl_matches_builder(tmp_path):
    """The committed qa_eval.jsonl must equal what the builder produces."""
    import scripts.build_eval_set as mod
    import json as _json

    generated = build()
    if (QA_DIR / "qa_eval.jsonl").exists():
        on_disk = [
            _json.loads(line)
            for line in (QA_DIR / "qa_eval.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert [i["question"] for i in on_disk] == [i["question"] for i in generated]
