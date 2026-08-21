"""Build the mushroom-safety pretraining corpus.

Pipeline:
  1. Synthetic corpus: assembled deterministically from ``kb.py`` (species
     fact sheets, poisoning syndromes, safety rules, FAQ, glossary, etc.).
  2. Best-effort enrichment (skipped silently if the network is unavailable):
     - Wikipedia plain-text extracts for a curated list of mushroom articles
       (CC BY-SA 4.0; attribution in data/corpus/NOTICE.md)
     - A public-domain mushroom book from Project Gutenberg, if reachable.
  3. Merge everything into ``data/corpus/corpus.txt`` with a source header.

Run:  python scripts/build_corpus.py
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.kb import (  # noqa: E402
    CULTIVATION, COOKING, FAQ, FUNDAMENTALS, GLOSSARY, GROUP_NOTES,
    REGIONAL, RULES, SPECIES, SYNDROMES,
)

CORPUS_DIR = ROOT / "data" / "corpus"
RAW_DIR = CORPUS_DIR / "raw"

WIKIPEDIA_TITLES = [
    "Edible mushroom", "Mushroom poisoning", "Amanita phalloides", "Amanita muscaria",
    "Amanita virosa", "Amanita bisporigera", "Amanita caesarea", "Amanita rubescens",
    "Amanita pantherina", "Boletus edulis", "Tylopilus felleus", "Imleria badia",
    "Boletus satanas", "Suillus luteus", "Cantharellus cibarius", "Cantharellus cinnabarinus",
    "Omphalotus olearius", "Hygrophoropsis aurantiaca", "Morchella esculenta", "Morchella angusticeps",
    "Gyromitra esculenta", "Verpa bohemica", "Pleurotus ostreatus", "Pleurocybella porrigens",
    "Lentinula edodes", "Agaricus bisporus", "Agaricus xanthodermus", "Chlorophyllum molybdites",
    "Macrolepiota procera", "Lepiota brunneoincarnata", "Coprinus comatus", "Coprinopsis atramentaria",
    "Entoloma sinuatum", "Clitocybe dealbata", "Cortinarius rubellus", "Cortinarius orellanus",
    "Galerina marginata", "Armillaria mellea", "Hypholoma fasciculare", "Lactarius deliciosus",
    "Lactarius torminosus", "Russula emetica", "Russula cyanoxantha", "Russula virescens",
    "Scleroderma citrinum", "Calvatia gigantea", "Lycoperdon perlatum", "Hydnum repandum",
    "Hericium erinaceus", "Grifola frondosa", "Laetiporus sulphureus", "Trametes versicolor",
    "Ganoderma lucidum", "Inonotus obliquus", "Tuber melanosporum", "Tuber magnatum",
    "Ophiocordyceps sinensis", "Phallus impudicus", "Sparassis crispa", "Tricholoma equestre",
    "Tricholoma matsutake", "Clitocybe nuda", "Craterellus cornucopioides", "Craterellus tubaeformis",
    "Pluteus cervinus", "Volvariella volvacea", "Stropharia rugosoannulata", "Cyclocybe aegerita",
    "Hypomyces lactifluorum", "Tremella fuciformis", "Auricularia auricula-judae",
    "Podostroma cornu-damae", "Ramaria", "Leucocoprinus birnbaumii", "Coprinopsis picacea",
    "Sarcoscypha coccinea", "Xylaria polymorpha", "Calocera viscosa", "Mushroom hunting",
    "Mushroom", "Mycology", "Spore print", "Pileus (mycology)", "Stipe (mycology)",
    "Lamella (mycology)", "Annulus (mycology)", "Volva (mycology)", "Basidiocarp",
    "Mycelium", "Ectomycorrhiza", "Saprotrophic nutrition", "Amatoxin", "Alpha-Amanitin",
    "Muscimol", "Ibotenic acid", "Muscarine", "Coprine", "Gyromitrin", "Orellanine",
    "Psilocybin mushroom", "Little brown mushroom", "Mushroom cultivation", "Truffle",
    "Medicinal fungi", "Ergothioneine", "Beta-glucan", "Mushroom hunting in Europe",
    "Fairy ring", "Stinkhorn", "Puffball", "Polypore", "Fungi", "Agaricales",
    "Boletales", "Russulales", "Amanitaceae", "Boletaceae", "Cantharellaceae",
    "Morchellaceae", "Tuberaceae", "Mushroom poisoning in Australia", "Mushroom hunting",
    "Mycophagy", "Poison control center", "Mushroom Observer", "iNaturalist",
]

# --------------------------------------------------------------------- #
# synthetic corpus assembly
# --------------------------------------------------------------------- #

P1_LEADS = [
    "The {name} ({binomial}) is a species in the {family}.",
    "{name}, scientifically known as {binomial}, belongs to the {family}.",
    "A member of the {family}, the {name} ({binomial}) is found across the world.",
    "In the {family}, the {name} ({binomial}) stands out as a notable species.",
]
P1_MID = [
    " Its cap is {cap}. The underside carries {gills}, and {stipe}.",
    " It is recognized by {cap}. Beneath the cap sit {gills}, and {stipe}.",
    " The fruit body shows {cap}; {gills} and {stipe} complete the picture.",
    " Typically, {cap}, with {gills} and {stipe}.",
]
P2_LEADS = [
    " The species is {habitat}.",
    " It grows {habitat}.",
    " In the wild, it is {habitat}.",
    " Ecologically, it is {habitat}.",
]
P2_MID = [
    " Fruiting occurs {season}, mainly in {region}.",
    " The mushrooms appear {season}; its range covers {region}.",
    " It fruits {season}, and is distributed across {region}.",
    " Look for it {season}, throughout {region}.",
]
P3_LEADS = [
    " In terms of edibility, the {name} is {status}.",
    " The {name} is {status}.",
    " Gastronomically, this mushroom is {status}.",
    " For humans, the {name} is {status}.",
]
P3_TOXIN = [
    " {toxin_cap}",
    " The reason is its toxins: {toxin}.",
    " The active principle is that {toxin}.",
    " Toxicology notes that {toxin}.",
]
P3_SYMPTOM = [
    " In practice, {symptom}.",
    " Clinically, {symptom}.",
    " If eaten, {symptom}.",
    " The result is that {symptom}.",
]
P4_LOOKALIKE = [
    " Identification is essential because {lookalike}.",
    " Care is needed: {lookalike}.",
    " Foragers should note that {lookalike}.",
    " The main risk is that {lookalike}.",
]
P4_NOTE = [
    " {note}.",
    " In short, {note}.",
    " Notably, {note}.",
]

BODY_FILLERS = [
    " Always confirm a mushroom's identity before eating it.",
    " When in doubt, throw it out.",
    " Consult a local field guide for the species in your area.",
    " Never rely on a single feature for identification.",
    " A spore print is a quick and reliable first check.",
    " Symptoms of poisoning vary greatly between species.",
    " Call a poison control center immediately if poisoning is suspected.",
    " This information is for education; verify with expert sources.",
]


def _species_paragraphs(species: dict, rng: random.Random) -> list[str]:
    s = species
    toxin_cap = s["toxin"].capitalize()
    if toxin_cap and not toxin_cap.endswith("."):
        toxin_cap += "."
    paras = []
    p1 = rng.choice(P1_LEADS).format(**s)
    p1 += rng.choice(P1_MID).format(**s)
    paras.append(p1)
    p2 = rng.choice(P2_LEADS).format(**s) + rng.choice(P2_MID).format(**s)
    paras.append(p2)
    p3 = rng.choice(P3_LEADS).format(name=s["name"], status=s["status"])
    if s["status"] != "none":
        p3 += " " + rng.choice(P3_TOXIN).format(toxin_cap=toxin_cap, toxin=s["toxin"])
    p3 += " " + rng.choice(P3_SYMPTOM).format(symptom=s["symptom"])
    paras.append(p3)
    p4 = rng.choice(P4_LOOKALIKE).format(lookalike=s["lookalike"])
    p4 += " " + rng.choice(P4_NOTE).format(note=s["note"])
    paras.append(p4)
    return paras


def build_synthetic(rng: random.Random) -> str:
    parts: list[str] = []
    parts.append("MUSHROOM SAFETY FIELD REFERENCE\n")
    parts.append("A reference text on the identification, ecology, toxicity and use of mushrooms.\n")

    for i, species in enumerate(SPECIES):
        paras = _species_paragraphs(species, rng)
        # each species gets 2 variants (different template draws)
        if rng.random() < 0.5:
            paras += _species_paragraphs(species, rng)
        parts.append("\n\n".join(paras))
        parts.append("")
        parts.append(rng.choice(BODY_FILLERS))
        parts.append("")

    parts.append("\n\nPOISONING SYNDROMES\n")
    for syn in SYNDROMES:
        parts.append(
            f"The {syn['name']} has a latency of {syn['latency']}. "
            f"The toxins involved are {syn['toxins']}. "
            f"The principal agents are {syn['agents']}. "
            f"Clinically, {syn['progress']}. "
            f"{syn['note']}."
        )

    parts.append("\n\nSAFETY RULES AND ESSAYS\n")
    parts.extend(RULES)

    parts.append("\n\nFREQUENTLY ASKED QUESTIONS\n")
    for q, a in FAQ:
        parts.append(f"Question: {q}\nAnswer: {a}")

    parts.append("\n\nGLOSSARY OF MYCOLOGICAL TERMS\n")
    for term, definition in GLOSSARY:
        parts.append(f"{term}: {definition}.")

    parts.append("\n\nCULTIVATION\n")
    parts.extend(CULTIVATION)

    parts.append("\n\nCOOKING AND NUTRITION\n")
    parts.extend(COOKING)

    parts.append("\n\nREGIONAL FORAGING NOTES\n")
    parts.extend(REGIONAL)

    parts.append("\n\nMYCOLOGY FUNDAMENTALS\n")
    parts.extend(FUNDAMENTALS)

    parts.append("\n\nFIELD GUIDE CHAPTERS\n")
    for group, text in GROUP_NOTES:
        parts.append(f"{group}: {text}")

    return "\n\n".join(parts) + "\n"


# --------------------------------------------------------------------- #
# best-effort downloads
# --------------------------------------------------------------------- #

def download_wikipedia(out_path: Path, titles: list[str], timeout: float = 12.0) -> int:
    """Fetch plain-text extracts for ``titles`` via the MediaWiki API."""
    import requests

    got = 0
    texts = []
    failed = []
    api = "https://en.wikipedia.org/w/api.php"
    for i in range(0, len(titles), 10):
        batch = titles[i : i + 10]
        try:
            r = requests.get(
                api,
                params={
                    "action": "query", "prop": "extracts", "explaintext": 1,
                    "exintro": 0, "titles": "|".join(batch), "format": "json",
                    "redirects": 1,
                },
                timeout=timeout,
            )
            r.raise_for_status()
            pages = r.json().get("query", {}).get("pages", {})
            for pid, page in pages.items():
                if "extract" in page and page["extract"].strip():
                    texts.append(f"\n===== {page['title']} =====\n{page['extract'].strip()}\n")
                    got += 1
                else:
                    failed.append(batch)
        except Exception as e:  # network failure: give up quietly
            print(f"[wiki] batch failed ({type(e).__name__}): {e}")
            failed.extend(batch)
        time.sleep(0.2)
    if got:
        out_path.write_text("\n".join(texts), encoding="utf-8")
    if failed:
        (out_path.parent / "wiki_failed_titles.txt").write_text("\n".join(map(str, failed)), encoding="utf-8")
    return got


def download_gutenberg(out_path: Path, timeout: float = 15.0) -> int:
    """Best-effort: download a public-domain mushroom book via gutendex + gutenberg.org."""
    import requests

    try:
        r = requests.get("https://gutendex.com/books", params={"search": "mushroom"}, timeout=timeout)
        r.raise_for_status()
        results = r.json().get("results", [])
        candidates = [b for b in results if "mushroom" in (b.get("title") or "").lower()]
        if not candidates:
            return 0
        book = candidates[0]
        book_id = book["id"]
        # prefer plain text format
        text_url = None
        for fmt, url in book.get("formats", {}).items():
            if fmt.startswith("text/plain"):
                text_url = url
                break
        if text_url is None:
            return 0
        r2 = requests.get(text_url, timeout=timeout * 4)
        r2.raise_for_status()
        content = r2.text
        if len(content) < 5000:
            return 0
        out_path.write_text(
            f"===== Project Gutenberg: {book['title']} =====\n{content}", encoding="utf-8"
        )
        return len(content)
    except Exception as e:
        print(f"[gutenberg] unavailable ({type(e).__name__}): {e}")
        return 0


# --------------------------------------------------------------------- #

def main(argv=None):
    p = argparse.ArgumentParser(description="Build the NanoLM mushroom-safety corpus")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--no-download", action="store_true", help="skip Wikipedia/Gutenberg attempts")
    p.add_argument("--wiki-titles", type=int, default=len(WIKIPEDIA_TITLES), help="max Wikipedia titles to fetch")
    args = p.parse_args(argv)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    synthetic = build_synthetic(rng)
    syn_path = RAW_DIR / "synthetic.txt"
    syn_path.write_text(synthetic, encoding="utf-8")
    print(f"[synthetic] {len(synthetic)/1024:.0f} KB -> {syn_path}")

    wiki_path = RAW_DIR / "wikipedia_extracts.txt"
    gut_path = RAW_DIR / "gutenberg_book.txt"
    if not args.no_download:
        n = download_wikipedia(wiki_path, WIKIPEDIA_TITLES[: args.wiki_titles])
        print(f"[wikipedia] {n}/{min(args.wiki_titles, len(WIKIPEDIA_TITLES))} articles fetched")
        g = download_gutenberg(gut_path)
        print(f"[gutenberg] {g/1024:.0f} KB downloaded" if g else "[gutenberg] skipped")
    else:
        print("[downloads] skipped (--no-download)")

    # merge
    parts = []
    header = (
        "NanoLM mushroom-safety training corpus.\n"
        "Sources: (1) synthetic fact sheets compiled from public reference material; "
        "(2) Wikipedia article extracts (CC BY-SA 4.0) when available; "
        "(3) public-domain Project Gutenberg text when available. "
        "See NOTICE.md for full attribution and licensing.\n"
        "Educational material only; not medical or foraging advice.\n"
    )
    parts.append(header)
    for f in sorted(RAW_DIR.glob("*.txt")):
        parts.append(f"\n\n<<< FILE: {f.name} >>>\n" + f.read_text(encoding="utf-8"))

    corpus = "\n".join(parts)
    corpus_path = CORPUS_DIR / "corpus.txt"
    corpus_path.write_text(corpus, encoding="utf-8")
    print(f"[merged] {len(corpus)/1024:.0f} KB -> {corpus_path}")
    print(f"[tokens-ish] ~{len(corpus)//4} tokens (approx 4 chars/token)")


if __name__ == "__main__":
    sys.exit(main())
