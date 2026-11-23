"""Retrieval evaluation: RAG with ontology vs. plain RAG (PDF only).

For every test question we know which PDF contains the answer. Metrics @k:
  hit@k        - is at least one chunk from the expected PDF among the top-k?
  precision@k  - share of the top-k PDF chunks that come from the expected PDF
  MRR          - 1 / rank of the first chunk from the expected PDF

    python evaluate.py
"""
import sys

import config
from rag.retriever import Retriever

sys.stdout.reconfigure(encoding="utf-8")

TASTING = {"04_tasting_menu_tudor_farmhouse.pdf", "05_tasting_menu_wine_pairing.pdf"}
SEAFOOD = {"01_seafood_restaurant_alc_menu.pdf", "02_seafood_restaurant_set_menu.pdf"}
ITALIAN = {"06_italian_vegetarian_menu.pdf", "07_italian_dietary_allergen_guide.pdf"}
HALAL = {"07_italian_dietary_allergen_guide.pdf", "09_halal_caterers_guide.pdf"}

# (question, set of PDFs that contain the answer)
TESTS = [
    ("How much does the tasting menu cost and is there a wine pairing?", TASTING),
    ("Which wine is paired with the amuse bouche?", {"05_tasting_menu_wine_pairing.pdf"}),
    ("Which seafood starters are on the menu?", SEAFOOD),
    ("How much is the fruits de mer shellfish platter?", {"01_seafood_restaurant_alc_menu.pdf"}),
    ("What desserts can children choose?", {"03_kids_menu.pdf"}),
    ("Which pasta dishes are vegan?", ITALIAN),
    ("Which dishes are lactose free?", {"07_italian_dietary_allergen_guide.pdf"}),
    ("Is the meat halal certified?", HALAL),
    ("How should caterers prepare and supervise halal food?", {"09_halal_caterers_guide.pdf"}),
    ("How can a restaurant attract vegan customers?", {"08_vegan_catering_made_easy.pdf"}),
    ("Which allergens must a food business declare?", {"10_allergen_technical_guidance.pdf", "07_italian_dietary_allergen_guide.pdf"}),
    ("What foods are typical for the Mediterranean diet?", {"11_mediterranean_diet.pdf"}),
    ("How many guests can the private dining room seat?", {"12_private_dining_brochure.pdf"}),
    ("What is the minimum spend for a private celebration?", {"12_private_dining_brochure.pdf"}),
    # Slovak questions
    ("Koľko stojí degustačné menu?", TASTING),
    ("Aké vegánske cestoviny ponúkate?", ITALIAN),
    ("Koľko hostí sa zmestí do súkromnej miestnosti?", {"12_private_dining_brochure.pdf"}),
    ("Aké dezerty si môžu vybrať deti?", {"03_kids_menu.pdf"}),
    ("Je mäso halal?", HALAL),
    ("Čo patrí do stredomorskej stravy?", {"11_mediterranean_diet.pdf"}),
]


def score(ret, use_ontology, rerank, k):
    hits, precs, rrs = [], [], []
    for q, expected in TESTS:
        r = ret.search(q, k=k, use_ontology=use_ontology, rerank=rerank)
        good = [c["source"] in expected for c in r.docs]
        hits.append(any(good))
        precs.append(sum(good) / len(r.docs) if r.docs else 0.0)
        rank = next((i for i, ok in enumerate(good, 1) if ok), None)
        rrs.append(1 / rank if rank else 0.0)
    n = len(TESTS)
    return sum(hits) / n, sum(precs) / n, sum(rrs) / n


def main():
    ret = Retriever()
    k = config.TOP_K
    print(f"{len(TESTS)} test questions, k={k}\n")
    print(f"{'setting':32s} {'hit@k':>7s} {'precision@k':>12s} {'MRR':>6s}")
    settings = (
        ("plain RAG (PDF only)", False, False),
        ("plain RAG + rerank", False, True),
        ("RAG + ontology", True, False),
        ("RAG + ontology + rerank", True, True),
    )
    for name, onto, rerank in settings:
        h, p, m = score(ret, onto, rerank, k)
        print(f"{name:32s} {h:7.2f} {p:12.2f} {m:6.2f}")


if __name__ == "__main__":
    main()
