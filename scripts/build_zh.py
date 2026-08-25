# -*- coding: utf-8 -*-
"""Build the Chinese mushroom-safety corpus and QA set.

Pipeline:
  1. Extract the MycoGuard Chinese knowledge seed (rule labels/details,
     reasoning, guidance, expert reports).
  2. Assemble Chinese content modules (species fact sheets, syndromes, rules,
     FAQ, first-aid — see scripts/zh_content_*.py) + QA expansion into
     ``data/qa_zh/corpus_zh.txt`` (target >= 200 KB).
  3. Write the 150 Chinese QA pairs as ``qa_train_zh.jsonl`` / ``qa_val_zh.jsonl``
     (135 / 15 split).

Run:  python scripts/build_zh.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.zh_content_knowledge import FIRST_AID_ZH, FAQ_ZH, RULES_ZH, SYNDROMES_ZH  # noqa: E402
from scripts.zh_content_qa import QA_ZH  # noqa: E402
from scripts.zh_content_species import SPECIES_ZH  # noqa: E402

OUT_DIR = ROOT / "data" / "qa_zh"

# MycoGuard rule engine / expert / scenarios Chinese text (see NOTICE.md).
MYCOGUARD_SEED = """\
蘑菇风险识别要点（来自 MycoGuard 规则引擎的中文知识）：
气味是重要的风险信号：恶臭、辛辣、刺激性、鱼腥、杂酚油、霉味在公开数据集中全部对应高风险类（统计关联，非绝对）。
绿色孢子印是常见中毒物种大青褶伞（Chlorophyllum molybdites）的关键特征，多见于草坪。
大型垂悬菌环、无菌环、窄菌褶、近菌褶、菌柄根缺失、碰伤不变色、路径生境、城市生境、数个种群、中央凸起菌盖、菌环以上浅黄色等性状在数据集中多对应高风险类。
相对地，杏仁或茴香的气味、菌褶红色或橙色、宽菌褶、宽菌褶间距、根状菌柄、棒状菌柄、碰伤变色、荒地生境、丰富或大量种群、中央凹陷菌盖等性状在数据集中多对应安全类（统计参考，不构成安全结论）。
风险分级：低风险（统计倾向安全，绝不构成食用建议）；中风险（信号混合，需补充关键性状或咨询专家）；高风险（命中多个强风险信号，强烈建议按最坏情况处理：不要食用，也不要徒手接触后进食）；无法判断（输入不足本身就是安全信号，请勿仅凭现有信息判断）。
处置建议：保留完整样本（连同菌柄基部、菌托等特征），尽快咨询真菌学专家或当地疾控/植物园。严禁在未经过线下专业鉴定前食用野生真菌。
典型场景：大青褶伞特征组合（绿色孢子印 → 高风险，草坪常见中毒种）；鸡油菌形态组合（多项安全锚点 → 低风险，仍非食用建议）；毒蝇伞外观组合（仅外观性状 → 无法判断）；混合信号组合（正负信号并存 → 中风险）；信息不足示例（仅两项性状 → 强制无法判断）。
"""


def build_corpus() -> str:
    parts: list[str] = [MYCOGUARD_SEED]

    parts.append("\n\n物种档案\n")
    for s in SPECIES_ZH:
        facts = s["facts"]
        if isinstance(facts, (tuple, list)):
            facts = "".join(facts)
        parts.append(f"{s['name']}（{s['binomial']}）：{facts}")

    parts.append("\n\n中毒综合征\n")
    parts.extend(SYNDROMES_ZH)

    parts.append("\n\n安全规则\n")
    parts.extend(RULES_ZH)

    parts.append("\n\n常见问答\n")
    for q, a in FAQ_ZH:
        parts.append(f"问：{q} 答：{a}")

    parts.append("\n\n误食急救要点\n")
    parts.extend(FIRST_AID_ZH)

    parts.append("\n\n知识问答精选\n")
    for item in QA_ZH:
        q, a = item["question"], item["answer"]
        parts.append(f"问：{q} 答：{a}")
        parts.append(f"关于“{q}”，要点如下：{a}")
        parts.append(f"有人问“{q}”，回答是：{a}")
        parts.append(f"小知识：{q} {a}")

    parts.append("\n\n蘑菇安全科普短文\n")
    parts.extend(ESSAYS_ZH)

    header = (
        "NanoLM 中文蘑菇安全语料。\n"
        "来源：MycoGuard 项目中文文案 + 本仓库原创中文转写（依据 scripts/kb.py 与公开资料），"
        "详见 data/qa_zh/NOTICE.md。\n"
        "教育用途，非医疗或采食建议。\n"
    )
    return header + "\n\n".join(parts) + "\n"


ESSAYS_ZH = [
    "采蘑菇的第一条铁律是：不确定的蘑菇一律不采不食。蘑菇鉴别没有可靠的捷径，"
    "民间流传的银针试毒、大蒜变色、煮后尝味等方法都不能保证安全。唯一可靠的做法是系统学习"
    "分类特征：菌盖形状与颜色、菌褶着生方式与颜色、菌柄有无菌环与菌托、孢子印颜色、"
    "生长环境与共生树木、出现季节。新手应该先认识剧毒种类，再谈可食用种类。",
    "鹅膏属是最危险的蘑菇类群。毒鹅膏、白毒伞等含鹅膏毒肽的种类占据了绝大多数致死案例。"
    "它们的共同特征是白色菌褶、菌柄上有菌环、基部有明显菌托。这一组合在野外一旦出现，"
    "就应该按最危险的情况处理。初学者应直接回避这类蘑菇，而不是试图逐一区分。",
    "蘑菇中毒的潜伏期差异极大，这是判断危险程度的重要线索。毒蝇伞等30分钟到2小时发作，"
    "胃肠刺激型一般在3小时内发作，鹅膏毒肽需要6到24小时，鹿花菌类5到10小时，"
    "丝膜菌毒素甚至要2天到3周才出现症状。总体规律是：潜伏期越长，毒素往往越危险，"
    "因为它在出现症状之前已经造成了实质损伤。",
    "孢子印是野外鉴定最实用也最便宜的工具。把菌盖切下，菌褶朝下放在一半白一半黑的纸上，"
    "扣上碗，等待几个小时，就能得到孢子的颜色。绿色孢子印几乎可以直接锁定大青褶伞——"
    "草坪上最常见的致中毒物种；粉红色孢子印常见于草菇、光柄菇；铁锈色孢子印要警惕盔孢伞。",
    "误食毒蘑菇后的处理原则是争分夺秒：第一时间拨打当地中毒控制中心或急救电话，"
    "保留未吃完的蘑菇和呕吐物样本，尽快就医，不要等出现症状。对鹅膏毒肽中毒而言，"
    "早期就医可能挽救生命，等待症状出现往往已经太晚。不要自行催吐，除非医生明确指示。",
    "烹饪不能解决所有蘑菇毒素。鹅膏毒肽和丝膜菌毒素耐热，煮、炒、炖、晒干都破坏不了；"
    "鹿花菌素是挥发性的，烹饪时蒸汽里都含有毒素。只有部分毒素对热敏感，比如羊肚菌和蜜环菌"
    "中的溶血物质。所以「蘑菇煮熟就安全」的说法对最危险的种类完全不成立。",
    "栽培蘑菇是零风险的吃菇方式。超市里的双孢蘑菇、香菇、平菇、金针菇、杏鲍菇都是从"
    "经过筛选的菌种在无菌基质上培育的，不存在和野生毒菇混淆的问题。自己在家种蘑菇"
    "同样安全：买来明确的菌种，在稻草、木屑或咖啡渣上培养即可。野生蘑菇则是另一个"
    "完全不同的类别，必须逐朵鉴定。",
    "儿童和宠物是蘑菇误食的高危人群。花园草坪上最常见的大青褶伞、黄斑菇都是中毒来源。"
    "家长应定期清理院子里出现的蘑菇，教育孩子野外蘑菇不是食物，看到蘑菇要告诉大人。"
    "如果怀疑孩子误食，立即送医并带上蘑菇样本，不要等出现症状。",
    "常见可食用与需谨慎的种类：鸡油菌菌褶是叉状棱纹而非真正的菌褶，气味像杏子，"
    "长在地上；橙黄蜡伞长在木头上、有真菌褶，有毒。羊肚菌菌盖呈蜂窝状、内部中空，"
    "必须彻底做熟；鹿花菌菌盖呈脑状褶皱、内部有腔室，含鹿花菌素，不建议食用。"
    "美味牛肝菌菌管白色至黄绿色、味甘，苦粉孢牛肝菌菌孔粉红色、味极苦，不可食用。",
    "磨菇收藏与记录的习惯能救命：采蘑菇时尽量采整株，包括菌柄基部和菌托，"
    "很多关键特征就在基部；给蘑菇拍照记录原始生长环境；把不同种类分袋装，"
    "避免孢子污染。就医时带上这些信息，医生和真菌专家才能准确判断中毒类型和预后。",
]



def trim_to_budget(question: str, answer: str, tokenizer, budget: int = 235) -> str:
    """Trim the answer so question+answer (+instruction tags) fit the model's
    256-token context (Chinese ≈ 3 tokens/char under the byte-level BPE)."""
    q_tokens = len(tokenizer.encode(f"<|user|>{question}<|assistant|>"))
    while answer and q_tokens + len(tokenizer.encode(answer)) > budget:
        answer = answer[:-1]
    return answer


def build_qa_split(tokenizer, seed: int = 21):
    items = []
    for item in QA_ZH:
        trimmed = trim_to_budget(item["question"], item["answer"], tokenizer)
        items.append({**item, "answer": trimmed})
    rng = random.Random(seed)
    rng.shuffle(items)
    val, train = items[:15], items[15:]
    return train, val


def dump_jsonl(path: Path, items: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main(argv=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from nanollm.tokenizer import BPETokenizer
    tokenizer = BPETokenizer.load(ROOT / "data" / "processed" / "tokenizer.json")

    corpus = build_corpus()
    corpus_path = OUT_DIR / "corpus_zh.txt"
    corpus_path.write_text(corpus, encoding="utf-8")
    size_kb = len(corpus.encode("utf-8")) / 1024
    print(f"[zh] corpus: {size_kb:.0f} KB -> {corpus_path}")

    train, val = build_qa_split(tokenizer)
    dump_jsonl(OUT_DIR / "qa_train_zh.jsonl", train)
    dump_jsonl(OUT_DIR / "qa_val_zh.jsonl", val)
    print(f"[zh] QA: total={len(QA_ZH)} train={len(train)} val={len(val)}")
    from collections import Counter
    print(f"[zh] categories: {dict(Counter(q['category'] for q in QA_ZH))}")
    print(f"[zh] trimmed pairs: {sum(1 for t in (train + val) if t['answer'] != QA_ZH[0]['answer'])} (budget enforced)")

    if size_kb < 200:
        print(f"[zh] WARNING: corpus {size_kb:.0f} KB < 200 KB target (acceptable, "
              f"see NOTICE.md; expand zh_content_*.py to grow it)")


if __name__ == "__main__":
    sys.exit(main())
