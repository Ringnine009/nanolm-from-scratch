"""Build a small held-out evaluation set for the LoRA-finetuned model.

44 mushroom-safety QA items (edibility / identification / habitat / symptoms /
first-aid), written by hand with reference answers and expected keywords.
Guaranteed **disjoint from the fine-tuning set**:

- every eval question is phrased differently from every training/validation
  question (no exact string match), and
- the normalized word-set Jaccard similarity to the nearest train/val
  question is below a threshold (asserted here).

Output: ``data/qa/qa_eval.jsonl`` (also usable from tests).

Run:  python scripts/build_eval_set.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QA_DIR = ROOT / "data" / "qa"

WORD_RE = re.compile(r"[a-z0-9']+")


def _words(text: str) -> set[str]:
    return set(WORD_RE.findall(text.lower()))


# question, reference answer, expected keywords (any-hit / all-hit), optional polarity
EVAL_ITEMS = [
    # ---- edibility yes/no ----
    ("Would it be safe to eat the death cap?", "No. The death cap is deadly poisonous and should never be eaten.",
     ["poisonous", "deadly", "amatoxin"], "no"),
    ("Could a person safely eat the destroying angel?", "No, the destroying angel is deadly poisonous.",
     ["poisonous", "deadly", "amatoxin"], "no"),
    ("Is the fly agaric something you can put in a meal?", "No, the fly agaric is poisonous and causes confusion and hallucinations.",
     ["poisonous", "muscimol", "ibotenic"], "no"),
    ("Can the common inkcap be eaten together with beer or wine?", "No. It contains coprine and reacts with alcohol, causing flushing and nausea.",
     ["coprine", "alcohol"], "no"),
    ("Are golden chanterelles good to eat?", "Yes, the golden chanterelle is a choice edible mushroom.",
     ["edible", "choice"], "yes"),
    ("Is the porcini a prized edible mushroom?", "Yes, the porcini is one of the most sought-after edible mushrooms.",
     ["edible", "prized", "choice"], "yes"),
    ("Can morels be eaten raw?", "No. Morels must be thoroughly cooked; raw or undercooked ones cause nausea and cramps.",
     ["cook", "raw", "nausea"], "no"),
    ("Is the green-spored parasol safe to eat?", "No, it is poisonous and causes violent vomiting and diarrhea.",
     ["poisonous", "vomiting"], "no"),
    ("Are young giant puffballs edible when the flesh is white?", "Yes, young puffballs with pure white flesh are edible.",
     ["white", "edible"], "yes"),
    ("Has the yellow knight been found dangerous to eat?", "No. It is now linked to rhabdomyolysis and is considered poisonous.",
     ["rhabdomyolysis", "poisonous"], "no"),
    # ---- identification ----
    ("What color are the gills of the death cap?", "The gills are white.",
     ["white"], None),
    ("What grows at the base of the destroying angel's stem?", "A cup-shaped volva.",
     ["volva"], None),
    ("How does the cap of a false morel look?", "It is wrinkled and brain-like, not honeycombed.",
     ["wrinkled", "brain"], None),
    ("What distinguishes real chanterelle ridges from true gills?", "The ridges are blunt and forked, not sharp blades.",
     ["forked", "blunt", "ridge"], None),
    ("Which spore print color does Chlorophyllum molybdites leave?", "A green spore print.",
     ["green"], None),
    ("What is the ring on an Amanita stem called?", "It is called the annulus.",
     ["annulus", "ring"], None),
    ("What color is the pore surface of the bitter bolete?", "The pores are pinkish.",
     ["pink"], None),
    ("Which feature separates puffballs from Amanita eggs?", "Cut them open: pure white flesh means puffball; a developing cap and gills means Amanita.",
     ["white", "flesh", "cut"], None),
    # ---- habitat / season ----
    ("Under which trees does the porcini typically grow?", "It grows with oak, beech, birch, spruce and pine.",
     ["oak", "beech", "birch", "spruce", "pine"], None),
    ("On what substrate do oyster mushrooms grow?", "They grow on dead broadleaf wood.",
     ["wood", "dead"], None),
    ("Where would you find a jack-o'-lantern mushroom?", "At the base of oaks, on stumps and buried wood.",
     ["wood", "stump"], None),
    ("In which season do morels appear?", "Morels fruit in spring.",
     ["spring"], None),
    ("Where does the golden chanterelle grow?", "On the ground in mossy woodland, in association with trees.",
     ["ground", "soil"], None),
    ("What tree is chaga found on?", "Chaga grows on birch trees.",
     ["birch"], None),
    ("Where do truffles grow?", "Underground, in a mycorrhizal partnership with tree roots.",
     ["underground", "below"], None),
    ("What is the typical habitat of the autumn skullcap?", "It grows on rotting conifer and hardwood logs and stumps.",
     ["wood", "log", "stump"], None),
    # ---- symptoms / poisoning ----
    ("How long does it take for amatoxin symptoms to appear?", "Symptoms appear 6 to 24 hours after eating.",
     ["6", "24", "hours", "delayed"], None),
    ("What does orellanine damage?", "It attacks the kidneys, often causing failure.",
     ["kidney"], None),
    ("Which syndrome follows eating the fool's funnel?", "Muscarine syndrome, with sweating, salivation and blurred vision.",
     ["muscarine", "sweating"], None),
    ("What happens if alcohol is drunk after eating the common inkcap?", "Flushing, sweating, nausea and palpitations due to coprine.",
     ["flush", "sweat", "nausea", "coprine"], None),
    ("Which organ does alpha-amanitin destroy?", "It destroys the liver.",
     ["liver"], None),
    ("What is the first phase of amatoxin poisoning?", "Violent cramps, vomiting and watery diarrhea.",
     ["vomit", "diarrhea", "cramp"], None),
    ("Which mushrooms cause the delayed kidney-failure syndrome?", "The deadly webcaps, which contain orellanine.",
     ["cortinarius", "webcap", "orellanine"], None),
    ("What should be done when mushroom poisoning is suspected?", "Contact a poison control center immediately and seek medical care.",
     ["poison control", "medical", "immediately"], None),
    # ---- first aid / general knowledge ----
    ("Your child ate an unknown mushroom in the garden. What should you do first?",
     "Remove any remaining mushroom, save a sample, and call a poison control center immediately — do not wait for symptoms.",
     ["poison control", "sample", "immediately"], None),
    ("Does cooking destroy amatoxins?", "No. Amatoxins survive boiling, frying and drying.",
     ["no", "survive", "boiling"], None),
    ("Is a silver spoon a reliable test for poisonous mushrooms?", "No. That is a myth; only careful identification is reliable.",
     ["myth", "no", "identify"], None),
    ("Why must morels always be cooked thoroughly?", "Raw morels contain heat-sensitive compounds that cause nausea and vomiting.",
     ["cook", "heat", "nausea"], None),
    ("What is the safest way to enjoy mushrooms without identification risk?",
     "Grow your own from known spawn, or buy cultivated mushrooms.",
     ["cultivate", "grow", "supermarket"], None),
    ("Which mushroom causes the most poisonings in North America?",
     "The green-spored parasol (Chlorophyllum molybdites), which grows on lawns.",
     ["green-spored", "chlorophyllum", "parasol"], None),
    ("What is the number one rule of foraging?",
     "Never eat a mushroom unless you are 100% certain of its identity.",
     ["certain", "identify", "never"], None),
    ("Can animals eating a mushroom prove it is safe for humans?",
     "No. Animal metabolism differs from ours; only identification is reliable.",
     ["no", "metabolism", "different"], None),
    ("Which part of the mushroom is the volva and why does it matter?",
     "The volva is the cup at the base of the stem; it marks the deadly Amanita genus.",
     ["base", "amanita", "deadly"], None),
    ("What does a spore print reveal and how is it taken?",
     "It reveals the spore color; lay the cap gill-side down on paper and wait a few hours.",
     ["spore", "paper", "color"], None),
]


def build() -> list[dict]:
    return [
        {"question": q, "answer": a, "keywords": kw, "polarity": pol}
        for q, a, kw, pol in EVAL_ITEMS
    ]


def check_disjoint(eval_items: list[dict], train_path: Path, val_path: Path) -> dict:
    """Verify the eval set does not overlap with the fine-tuning set."""
    train_qs: list[str] = []
    for p in (train_path, val_path):
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    train_qs.append(json.loads(line)["question"])
    train_word_sets = [_words(q) for q in train_qs]

    exact_dups, near_dups = [], []
    for item in eval_items:
        q = item["question"]
        wq = _words(q)
        if q in train_qs:
            exact_dups.append(q)
        if not wq:
            continue
        best = max(
            (len(wq & tw) / len(wq | tw) for tw in train_word_sets),
            default=0.0,
        )
        if best >= 0.75:
            near_dups.append((q, round(best, 2)))

    return {
        "eval_items": len(eval_items),
        "train_questions": len(train_qs),
        "exact_duplicates": exact_dups,
        "near_duplicates": near_dups,
    }


def main(argv=None):
    items = build()
    report = check_disjoint(items, QA_DIR / "qa_train.jsonl", QA_DIR / "qa_val.jsonl")
    assert not report["exact_duplicates"], f"exact duplicates: {report['exact_duplicates']}"
    assert not report["near_duplicates"], f"near duplicates: {report['near_duplicates']}"
    QA_DIR.mkdir(parents=True, exist_ok=True)
    with open(QA_DIR / "qa_eval.jsonl", "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[eval-set] {len(items)} items written to data/qa/qa_eval.jsonl")
    print(f"[eval-set] disjoint check OK vs {report['train_questions']} train/val questions")


if __name__ == "__main__":
    sys.exit(main())
