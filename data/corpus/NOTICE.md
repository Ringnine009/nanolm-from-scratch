# Corpus sources & licensing

The training corpus (`data/corpus/corpus.txt`, built by `scripts/build_corpus.py`)
is a mushroom-safety reference text assembled from the following sources:

1. **Synthetic fact sheets (this repository)**
   `scripts/kb.py` contains species descriptions, poisoning syndromes, safety
   rules, FAQ, glossary, cultivation/cooking/regional notes and mycology
   fundamentals. The facts are compiled from widely published public reference
   material (e.g. public poison-control guidance, mycological field guides,
   Wikipedia). The *wording* is original to this repository (MIT licensed);
   the underlying *facts* are public knowledge and are provided for education
   only — always verify with a qualified source before acting on them.

2. **Wikipedia article extracts (CC BY-SA 4.0)** — attempted at build time
   (best-effort, network dependent). Titles are listed in
   `scripts/build_corpus.py` (`WIKIPEDIA_TITLES`). Extracts downloaded by the
   script are saved to `data/corpus/raw/wikipedia_extracts.txt` and are
   licensed under the Creative Commons Attribution-ShareAlike 4.0 license
   (https://creativecommons.org/licenses/by-sa/4.0/). Attribution: "from
   Wikipedia, the free encyclopedia".

3. **Project Gutenberg public-domain books** — attempted at build time
   (best-effort, network dependent). Downloaded text is saved to
   `data/corpus/raw/gutenberg_book.txt` and is in the public domain.

## Disclaimer

This corpus is educational material about mushroom identification, ecology,
toxicity and use. It is **not** medical advice and **not** foraging advice.
Mushroom poisoning can be fatal; always confirm identification with a
qualified expert and consult a poison control center in an emergency.
The model trained on this corpus (NanoLM) is a demonstration of deep-learning
techniques and must never be used for real mushroom identification.
