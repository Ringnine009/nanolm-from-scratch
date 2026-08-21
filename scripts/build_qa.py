"""Build the mushroom-safety instruction QA set for LoRA fine-tuning.

Two sources:
  1. Programmatically generated pairs from ``kb.py`` (species fact sheets),
     phrased with varied question templates and full-sentence answers.
  2. Manually written, polished pairs (the 'manual proofreading' pass).

Output: data/qa/qa_train.jsonl and data/qa/qa_val.jsonl (90/10 split).

Run:  python scripts/build_qa.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.kb import FAQ, SPECIES  # noqa: E402

QA_DIR = ROOT / "data" / "qa"

# --------------------------------------------------------------------- #
# generated pairs
# --------------------------------------------------------------------- #

STATUS_YES = {"edible", "choice edible", "edible after cooking", "edible with caution",
              "edible when young", "edible, widely cultivated", "edible and widely cultivated",
              "edible and highly regarded", "edible only when thoroughly cooked",
              "edible after thorough cooking", "edible only after thorough cooking",
              "edible after removing the slimy cap skin", "edible, highly regarded",
              "edible when young and pure white inside", "edible when young and white",
              "edible but mediocre", "edible but insubstantial"}
STATUS_NO = {"poisonous", "deadly poisonous", "poisonous (mild)", "poisonous when raw; traditionally pickled in some regions",
             "poisonous when combined with alcohol", "poisonous, especially raw",
             "poisonous only when combined with alcohol", "poisonous to people with kidney disease",
             "inedible", "inedible (very bitter)", "not edible, used medicinally",
             "not edible; used in traditional medicine", "not eaten as food; used in traditional medicine",
             "not edible; brewed as tea or extract", "inedible; tough and bitter",
             "deadly in some regions, conditionally eaten in others", "conditionally edible, not recommended",
             "mildly poisonous / not recommended", "poisonous (formerly considered edible)",
             "some edible, some poisonous; not recommended for beginners"}


def _status_answer(status: str) -> str:
    if status in STATUS_YES:
        return "Yes"
    if status in STATUS_NO:
        return "No"
    return "It depends"


def _answer_edibility(s: dict) -> str:
    status = s["status"]
    if status in STATUS_YES:
        if "cook" in status or "thorough" in status:
            return (f"Yes, the {s['name']} is edible, but only after thorough cooking. "
                    f"{s['toxin'].capitalize()}, so eating it raw or undercooked can cause {s['symptom']}.")
        if "young" in status:
            return (f"Yes, when young. The {s['name']} is edible only while the flesh is young and pure white; "
                    f"as it matures it becomes unpalatable. {s['symptom'].capitalize()}.")
        return (f"Yes, the {s['name']} is {status} in most cases. {s['symptom'].capitalize()}.")
    if status in STATUS_NO:
        if "alcohol" in status:
            return (f"No, not if you drink alcohol. The {s['name']} contains coprine, and {s['symptom']}.")
        if "kidney" in status:
            return (f"No. The {s['name']} is poisonous to people with kidney disease: {s['symptom']}.")
        if "former" in status:
            return (f"No. The {s['name']} was once considered edible, but is now known to be poisonous: {s['symptom']}.")
        if "medicinal" in status:
            return (f"The {s['name']} is not eaten as food; it is used as a medicinal mushroom instead. {s['symptom'].capitalize()}.")
        if "inedible" in status:
            return (f"No, the {s['name']} is inedible. {s['symptom'].capitalize()}.")
        return (f"No, the {s['name']} is {status} and should never be eaten. {s['toxin'].capitalize()}. {s['symptom'].capitalize()}.")
    return (f"Not recommended. The {s['name']} is {status}; {s['symptom']}.")


def _answer_identify(s: dict) -> str:
    return (f"Look for {s['cap']}. Check the underside: {s['gills']}, and {s['stipe']}. "
            f"The species is {s['habitat']}, fruiting {s['season']} in {s['region']}. "
            f"{s['note'].capitalize()}.")


def _answer_habitat(s: dict) -> str:
    return (f"The {s['name']} is {s['habitat']}. It fruits {s['season']}, and its range covers {s['region']}. "
            f"{s['note'].capitalize()}.")


def _answer_symptoms(s: dict) -> str:
    if s["status"] in STATUS_YES | {"edible with caution"}:
        return (f"The {s['name']} is edible, so there are normally no poisoning symptoms. {s['symptom'].capitalize()}.")
    return (f"If you eat the {s['name']}, {s['symptom']}. The toxins involved are {s['toxin']}. "
            f"Seek medical help promptly if symptoms appear.")


def _answer_lookalike(s: dict) -> str:
    return (f"The {s['name']} can be confused with other mushrooms: {s['lookalike']}. "
            f"Always verify the identity before eating. {s['note'].capitalize()}.")


def _answer_poison(s: dict) -> str:
    return (f"The {s['name']} is {s['status']}. {s['toxin'].capitalize()}. {s['symptom'].capitalize()} "
            f"Contact a poison control center immediately if it is eaten.")


QUESTION_TEMPLATES = [
    ("edible", "Is the {name} edible?", "Can I eat the {name}?", "Is the {name} safe to eat?"),
    ("identify", "How can I identify the {name}?", "What does the {name} look like?", "How do I recognize the {name}?"),
    ("habitat", "Where does the {name} grow?", "When can I find the {name}?", "Where should I look for the {name}?"),
    ("symptoms", "What happens if I eat the {name}?", "What are the symptoms of eating the {name}?"),
    ("lookalike", "What mushrooms look like the {name}?", "How do I avoid confusing the {name} with other mushrooms?"),
]


def generate_pairs() -> list[dict]:
    pairs: list[dict] = []
    rng = random.Random(11)
    for s in SPECIES:
        name = s["name"]
        for kind, *qs in QUESTION_TEMPLATES:
            q = rng.choice(qs).format(name=name)
            if kind == "edible":
                a = _answer_edibility(s)
            elif kind == "identify":
                a = _answer_identify(s)
            elif kind == "habitat":
                a = _answer_habitat(s)
            elif kind == "symptoms":
                a = _answer_symptoms(s)
            else:
                a = _answer_lookalike(s)
            pairs.append({"question": q, "answer": a, "source": "generated"})
        # one poison-focused pair for every toxic/deadly species
        if s["status"] in STATUS_NO:
            pairs.append({
                "question": rng.choice([
                    "What should I know about the toxicity of the {name}?",
                    "How dangerous is the {name}?",
                ]).format(name=name),
                "answer": _answer_poison(s),
                "source": "generated",
            })
    return pairs


# --------------------------------------------------------------------- #
# manually written pairs (proofread pass)
# --------------------------------------------------------------------- #
MANUAL_PAIRS = [
    {"question": "What is the most important rule of mushroom foraging?",
     "answer": "Never eat a mushroom unless you are 100% certain of its identity. There is no reliable home test, folk trick, or cooking method that makes a poisonous mushroom safe, so careful identification is the only protection."},
    {"question": "How do I take a spore print?",
     "answer": "Cut off the cap and lay it gill-side down on a sheet of paper, half white and half dark, cover it with a bowl to keep air out, and wait a few hours. The color of the spores released tells you the mushroom's spore print color, which is one of the most useful identification features."},
    {"question": "Why is the death cap so dangerous?",
     "answer": "The death cap contains amatoxins that destroy liver cells, and symptoms only appear 6 to 24 hours after eating, when the damage is already underway. It is responsible for the majority of fatal mushroom poisonings worldwide and resembles some edible mushrooms."},
    {"question": "What are the first signs of amatoxin poisoning?",
     "answer": "Amatoxin poisoning begins 6 to 24 hours after the meal with violent cramps, vomiting and watery diarrhea. This is followed by a deceptive period of apparent recovery, and then liver and kidney failure. Anyone with these symptoms should seek emergency medical care immediately."},
    {"question": "Can cooking make the death cap safe to eat?",
     "answer": "No. Amatoxins survive boiling, frying and drying. The death cap remains deadly no matter how it is prepared, which is why identification — not cooking — is the only safety measure."},
    {"question": "How can I tell a chanterelle from a jack-o'-lantern mushroom?",
     "answer": "True chanterelles have blunt, forked ridges instead of sharp gills, grow on the ground in association with trees, and smell faintly of apricots. The jack-o'-lantern has true sharp gills, grows in clusters on wood, and is poisonous."},
    {"question": "What should I do if someone eats a poisonous mushroom?",
     "answer": "Call a poison control center immediately, save every piece of the uneaten mushroom for identification, and seek medical care even if the person feels fine, because the most dangerous toxins act slowly. Do not wait for symptoms and do not induce vomiting unless told to."},
    {"question": "Are morels safe to eat?",
     "answer": "Yes, but only after thorough cooking. Raw or undercooked morels contain heat-sensitive compounds that cause nausea, vomiting and cramps. Also make sure you have a true morel: false morels have wrinkled, brain-like caps and can be deadly."},
    {"question": "What is a volva and why should beginners fear it?",
     "answer": "The volva is a cup-shaped sac at the base of the stem, a remnant of the membrane that enclosed the young mushroom. It is the signature of the Amanita genus, which contains the deadliest mushrooms on earth, so a mushroom with a volva should be treated with the greatest caution."},
    {"question": "Why do some mushroom poisonings take days to show symptoms?",
     "answer": "The most lethal toxins attack the liver and kidneys rather than the stomach, and take hours or days to be absorbed and processed. Amatoxin symptoms appear after 6 to 24 hours, while orellanine symptoms can take two days to three weeks, by which time serious organ damage has occurred."},
    {"question": "Is the green-spored parasol poisonous?",
     "answer": "Yes. Chlorophyllum molybdites is the most common cause of mushroom poisoning in North America. It grows on lawns, looks like the edible shaggy parasol, and causes violent vomiting and diarrhea. The decisive test is a spore print: green spores mean poison."},
    {"question": "How do I tell a puffball from an Amanita egg?",
     "answer": "Cut the specimen in half vertically. An edible young puffball is pure white and homogeneous inside with no structure. An Amanita egg contains a developing cap, gills and stem inside, and belongs to the deadliest family of mushrooms. When in doubt, throw it out."},
    {"question": "Why should I avoid little brown mushrooms?",
     "answer": "Small brown gilled mushrooms, often called LBMs, are extremely hard to identify and include the deadly autumn skullcap (Galerina marginata). Beginners should skip the entire category until they can identify the genus with confidence."},
    {"question": "What does the fly agaric do if eaten?",
     "answer": "The fly agaric contains ibotenic acid and muscimol. Within 30 minutes to 2 hours it causes nausea, drowsiness, confusion, hallucinations and a drunken-like ataxia. Poisoning is unpleasant but death is rare."},
    {"question": "Do mushrooms that grow on wood have different risks?",
     "answer": "Yes. Wood-rotting mushrooms include the delicious oyster and honey fungus, but also the deadly autumn skullcap and the toxic sulfur tuft. The same rigorous identification applies: check gill color, ring, spore print, and the host tree."},
    {"question": "Can I eat mushrooms from my own lawn?",
     "answer": "Only if you can identify them with certainty. Lawns commonly host the poisonous green-spored parasol and the toxic yellow stainer, so garden mushrooms deserve exactly the same caution as forest mushrooms."},
    {"question": "What is the difference between a morel and a false morel?",
     "answer": "A true morel has a honeycomb cap of pits and ridges fused to a hollow stem. A false morel (Gyromitra) has a wrinkled, brain-like cap and a chambered interior, and contains gyromitrin, which can be deadly. The two are separated by the cap structure and the interior of the stem."},
    {"question": "Are boletes safe to eat?",
     "answer": "Most boletes are edible, but not all. Avoid any bolete with red or orange pores, any that bruise deep blue rapidly, and any that tastes bitter. The porcini's dangerous lookalike, the bitter bolete, has pink pores and an intensely bitter taste."},
    {"question": "What should I bring on my first mushroom hunt?",
     "answer": "Bring a basket, a knife, a brush, a regional field guide, paper bags, a notebook and a camera. Most importantly, go with an experienced forager the first several times, and always follow the rule: when in doubt, throw it out."},
    {"question": "How can I grow mushrooms safely at home?",
     "answer": "Growing mushrooms is the zero-risk way to enjoy them: buy spawn of a known species such as oyster, shiitake or wine cap, and grow it on straw, sawdust or wood chips. Because you control the species from spawn to harvest, there is no identification risk."},
    {"question": "Why do chanterelles not have real gills?",
     "answer": "Chanterelles carry blunt, forked ridges called false gills rather than sharp blade-like gills. This is a reliable field mark: the poisonous jack-o'-lantern lookalike has true sharp gills. Chanterelles also grow on the ground, never on wood."},
    {"question": "What is the most common mushroom poisoning?",
     "answer": "The most common category is gastrointestinal irritant poisoning — nausea, vomiting, cramps and diarrhea within a few hours — caused by mushrooms such as the green-spored parasol, the jack-o'-lantern and the sulfur tuft. It is unpleasant but usually not life-threatening."},
    {"question": "How does the common inkcap interact with alcohol?",
     "answer": "The common inkcap contains coprine, which blocks the enzyme that breaks down alcohol. Drinking alcohol within days of eating it causes flushing, sweating, nausea, palpitations and headache. The reaction is like the drug disulfiram and can be frightening, though it is rarely dangerous."},
    {"question": "Are supermarket mushrooms safe to eat raw?",
     "answer": "Yes. Cultivated mushrooms such as button, portobello, enoki and shiitake are grown from controlled spawn on sterile substrates, so they contain no wild lookalikes. Button mushrooms are safe raw in salads, though shiitake should be cooked."},
    {"question": "What makes the poison fire coral so dangerous?",
     "answer": "The poison fire coral (Podostroma cornu-damae) contains trichothecene mycotoxins that block protein synthesis. Symptoms include nausea, vomiting, low blood counts, peeling skin and hair loss, and deaths are recorded in Japan and Korea. It can be confused with edible coral fungi."},
    {"question": "Is the yellow knight mushroom edible?",
     "answer": "No. The yellow knight (Tricholoma equestre) was considered a delicacy for decades, but is now linked to rhabdomyolysis — muscle breakdown that can lead to kidney failure — with several deaths in France. It is a reminder that 'traditionally eaten' is not proof of safety."},
    {"question": "How do I identify the destroying angel?",
     "answer": "The destroying angel is a pure white mushroom with white gills, a ring on the stem and a deep cup-shaped volva at the base. It grows in mossy woodland and is deadly poisonous. Young specimens can resemble button mushrooms, which is why white Amanitas must never be eaten by beginners."},
    {"question": "What is a fairy ring?",
     "answer": "A fairy ring is a circle of mushrooms marking the expanding edge of an underground mycelium, common in lawns and meadows. The fairy ring champignon (Marasmius oreades) is edible, but the small white Clitocybe species that grow alongside it can be deadly, so identify carefully."},
    {"question": "Can dried mushrooms lose their toxicity?",
     "answer": "No. Drying, freezing and boiling do not destroy amatoxins or orellanine. Only a few toxins, such as the hemolysins in morels, are broken down by heat. Preservation changes shelf life, never toxicity."},
    {"question": "Why are truffles harvested with dogs?",
     "answer": "Truffles grow underground in a mycorrhizal partnership with tree roots and release a powerful aroma that dogs (and historically pigs) can detect through the soil. Since they are invisible from the surface and cannot be cultivated like ordinary mushrooms, trained animals are the most efficient harvesters."},
    {"question": "Is it safe to eat a mushroom that animals eat?",
     "answer": "No. Many animals can eat mushrooms that are deadly to humans because their metabolism differs. The rabbit that nibbles a mushroom tells you nothing about its safety for people. Identification is the only reliable test."},
    {"question": "What should I teach children about mushrooms?",
     "answer": "Teach children that wild mushrooms are never food, that they should never touch or taste mushrooms they find, and that they should tell an adult immediately. Remove unknown mushrooms from lawns and gardens where children and pets play."},
    {"question": "Are brightly colored mushrooms poisonous?",
     "answer": "Not necessarily. The golden chanterelle and the orange Caesar's mushroom are brightly colored and delicious, while some of the deadliest mushrooms, like the death cap, are dull olive. Color alone tells you almost nothing; identification features matter."},
    {"question": "What is the safest edible mushroom for a beginner to learn?",
     "answer": "There is no truly 'safe' shortcut — every mushroom must be identified. However, the giant puffball (cut in half to confirm pure white flesh), the hedgehog mushroom with its spiny underside, and the oyster mushroom on broadleaf wood are among the easier ones to learn with an experienced guide."},
]


def main(argv=None):
    QA_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(3)

    pairs = generate_pairs() + MANUAL_PAIRS
    rng.shuffle(pairs)
    n_val = max(1, len(pairs) // 10)
    val, train = pairs[:n_val], pairs[n_val:]

    def dump(path, items):
        with open(path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    dump(QA_DIR / "qa_train.jsonl", train)
    dump(QA_DIR / "qa_val.jsonl", val)
    print(f"[qa] total={len(pairs)} train={len(train)} val={len(val)}")
    print(f"[qa] wrote {QA_DIR/'qa_train.jsonl'} and {QA_DIR/'qa_val.jsonl'}")


if __name__ == "__main__":
    sys.exit(main())
