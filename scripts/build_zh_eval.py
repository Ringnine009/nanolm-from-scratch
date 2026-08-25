# -*- coding: utf-8 -*-
"""Build a held-out Chinese evaluation set (30 items, disjoint from the
150-item Chinese fine-tuning QA set).  Each item has expected keywords for
keyword-based checks; relevance/factuality are judged by an LLM in
``scripts/eval_zh.py``.

Run:  python scripts/build_zh_eval.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.zh_content_qa import QA_ZH  # noqa: E402

OUT = ROOT / "data" / "qa_zh" / "qa_eval_zh.jsonl"

# question, expected keywords (any of them), category
EVAL_ZH = [
    ("误食毒鹅膏后，中毒潜伏期一般有多长？", ["6", "24", "小时", "延迟"], "毒性"),
    ("毒鹅膏的毒性成分主要是什么？", ["鹅膏毒肽", "鹅膏蕈碱", "amanitin", "毒肽"], "毒性"),
    ("鹅膏毒肽主要损伤人体的哪个器官？", ["肝"], "毒性"),
    ("误食毒蝇伞后通常会出现哪些表现？", ["幻觉", "嗜睡", "意识", "眩晕", "恶心"], "毒性"),
    ("大青褶伞为什么会让不少人中毒？", ["呕吐", "腹泻", "胃肠", "草坪"], "毒性"),
    ("墨汁鬼伞和酒精一起食用会发生什么？", ["鬼伞素", "酒精", "脸红", "恶心", "心跳"], "毒性"),
    ("鹿花菌含有哪种毒素、主要危害什么？", ["鹿花菌素", "甲基肼", "肝"], "毒性"),
    ("丝膜菌属中毒有什么特别之处？", ["肾", "延迟", "2天", "3周", "丝膜"], "毒性"),
    ("橙黄蜡伞（jack-o'-lantern）误食后主要有什么症状？", ["呕吐", "腹泻", "胃肠", "痉挛"], "毒性"),
    ("哪一种蘑菇中毒潜伏期最长、最容易被忽视？", ["丝膜", "肾", "延迟"], "毒性"),
    ("怀疑家人误食了野生蘑菇，第一步应该做什么？", ["中毒控制", "疾控", "就医", "保留", "急救"], "急救"),
    ("误食毒蘑菇后为什么要保留没吃完的样本？", ["鉴定", "样本", "医生", "就医"], "急救"),
    ("误食毒蘑菇后暂时没有症状，还需要去医院吗？", ["需要", "立即", "等待", "就医"], "急救"),
    ("用银针、大蒜试蘑菇毒的说法可靠吗？", ["不可靠", "银针", "鉴定", "民间"], "急救"),
    ("儿童误食了花园里的野生蘑菇应该怎么办？", ["就医", "保留", "样本", "医院", "立即"], "急救"),
    ("为什么鹅膏毒肽中毒不能等出现症状再去医院？", ["延迟", "肝", "早", "就医"], "急救"),
    ("鸡油菌和橙黄蜡伞在外观上怎么区分？", ["菌褶", "假菌褶", "木头", "地面", "叉状"], "区分"),
    ("羊肚菌和鹿花菌的菌盖有什么区别？", ["蜂窝", "菌盖", "脑", "空心", "褶皱"], "区分"),
    ("毒鹅膏和可食用的草菇怎么区分？", ["菌托", "草菇", "孢子印", "粉红"], "区分"),
    ("孢子印应该怎么做、有什么用？", ["孢子印", "纸", "菌盖", "颜色"], "区分"),
    ("菌柄基部有明显的菌托，通常说明什么？", ["鹅膏", "菌托", "危险", "鹅膏属"], "区分"),
    ("毒蝇伞红色菌盖上的白色斑点是什么？", ["菌幕", "残余", "外菌幕", "鳞片"], "区分"),
    ("羊肚菌可以生吃吗？", ["不能", "做熟", "生吃", "恶心"], "食用"),
    ("超市里买的栽培蘑菇安全吗？", ["安全", "栽培", "超市", "种植"], "食用"),
    ("第一次吃一种没吃过的野生蘑菇，应该注意什么？", ["少量", "先", "做熟", "少量试"], "食用"),
    ("美味牛肝菌和苦粉孢牛肝菌怎么区分？", ["苦", "菌管", "粉红", "孔", "味道"], "食用"),
    ("平菇一般生长在什么地方？", ["木头", "朽木", "阔叶", "树"], "环境"),
    ("毒鹅膏通常和什么树形成共生关系？", ["栎", "橡树", "共生", "阔叶"], "环境"),
    ("羊肚菌一般出现在什么季节？", ["春"], "环境"),
    ("橙黄蜡伞一般长在哪里？", ["木头", "树桩", "根部", "木"], "环境"),
]

WORD_RE = re.compile(r"[\u4e00-\u9fff0-9a-zA-Z]+")


def _words(s: str) -> set[str]:
    return set(WORD_RE.findall(s.lower()))


def build() -> list[dict]:
    return [{"question": q, "keywords": kw, "category": c} for q, kw, c in EVAL_ZH]


def check_disjoint(eval_items: list[dict]) -> dict:
    train_qs = [q["question"] for q in QA_ZH]
    train_sets = [_words(q) for q in train_qs]
    exact, near = [], []
    for item in eval_items:
        q = item["question"]
        wq = _words(q)
        if q in train_qs:
            exact.append(q)
        if wq:
            best = max((len(wq & tw) / len(wq | tw) for tw in train_sets), default=0.0)
            if best >= 0.75:
                near.append((q, round(best, 2)))
    return {"eval": len(eval_items), "train": len(train_qs), "exact": exact, "near": near}


def main(argv=None):
    items = build()
    report = check_disjoint(items)
    assert not report["exact"], f"exact dup: {report['exact']}"
    assert not report["near"], f"near dup: {report['near']}"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[zh-eval] {len(items)} items written to {OUT}")
    print(f"[zh-eval] disjoint OK vs {report['train']} training questions")


if __name__ == "__main__":
    sys.exit(main())
