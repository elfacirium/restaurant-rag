"""Evaluation of the knowledge-graph part (GraphRAG).

For each test question the expected answer is computed directly from data/kg/dishes.json with plain
Python filters (independent of the SPARQL code). Then the question goes through the real pipeline
(question -> ontology concepts -> SPARQL -> rdflib) and the returned dishes are compared:

    precision = correct returned / returned       recall = correct returned / expected

    python evaluate_kg.py                                          # graph from verified data
    python evaluate_kg.py --ttl data/kg/restaurant_data_llm.ttl    # graph from automatic LLM extraction

Dish names are matched fuzzily (the LLM may write a name slightly differently).
"""
import argparse
import json
import sys

import config
from extract_kg import similarity
from rag.kg import KG_TTL, KnowledgeGraph
from rag.ontology import Ontology

sys.stdout.reconfigure(encoding="utf-8")
DATA = json.loads((config.ROOT / "data" / "kg" / "dishes.json").read_text(encoding="utf-8"))
CURRENCY = {m["id"]: m["currency"] for m in DATA["menus"]}

# flatten: one record per dish with everything the gold filters need
ITEMS = []
for m in DATA["menus"]:
    for it in m["items"]:
        f = set(it.get("f", []))
        if "ve" in f:
            f.add("v")
        if "a" in it and m["source"] == "12_private_dining_brochure.pdf":
            if "G" not in it["a"]:
                f.add("gf")
            if "D" not in it["a"]:
                f.add("lf")
        ITEMS.append({"name": it["n"], "t": it["t"], "c": set(it.get("c", [])), "p": it.get("p"),
                      "cur": m["currency"], "f": f, "menu": m["type"], "cu": it.get("cu", m["cuisine"])})

MAIN = {"MainCourse", "PastaDish", "SeafoodDish", "MeatDish"}
TESTS = [
    ("Which pasta dishes are vegan?", lambda i: (i["t"] == "PastaDish" or "pasta" in i["c"]) and "ve" in i["f"]),
    ("Which vegan dishes cost less than £15?", lambda i: "ve" in i["f"] and i["p"] is not None and i["p"] <= 15 and i["cur"] == "GBP"),
    ("Which seafood starters are cheaper than £15?", lambda i: i["t"] == "Starter" and "seafood" in i["c"] and i["p"] is not None and i["p"] <= 15 and i["cur"] == "GBP"),
    ("What desserts can children choose?", lambda i: i["t"] == "Dessert" and i["menu"] == "KidsMenu"),
    ("Which main courses are vegetarian?", lambda i: i["t"] in MAIN and "v" in i["f"]),
    ("Is the meat halal certified?", lambda i: (i["t"] == "MeatDish" or "meat" in i["c"]) and "h" in i["f"]),
    ("Which gluten free Italian dishes are there?", lambda i: "gf" in i["f"] and i["cu"] == "Italian"),
    ("Which wines are under £40?", lambda i: "wine" in i["c"] and i["p"] is not None and i["p"] <= 40 and i["cur"] == "GBP"),
    ("Which lactose free desserts are there?", lambda i: i["t"] == "Dessert" and "lf" in i["f"]),
    ("Which Asian seafood dishes are on the menus?", lambda i: (i["t"] == "SeafoodDish" or "seafood" in i["c"]) and i["cu"] == "Asian"),
    ("Aké vegánske cestoviny ponúkate?", lambda i: (i["t"] == "PastaDish" or "pasta" in i["c"]) and "ve" in i["f"]),
    ("Ktoré dezerty sú lacnejšie ako 12 libier?", lambda i: i["t"] == "Dessert" and i["p"] is not None and i["p"] <= 12 and i["cur"] == "GBP"),
    ("Aké vegetariánske hlavné jedlá máte?", lambda i: i["t"] in MAIN and "v" in i["f"]),
    ("Ktoré bezlepkové dezerty máte?", lambda i: i["t"] == "Dessert" and "gf" in i["f"]),
]


def fuzzy_hits(gold: set, got: set, threshold: float = 0.75) -> tuple[set, set]:
    """Greedy best-first matching of returned names to expected names."""
    pairs = sorted(((similarity(g, r), g, r) for g in gold for r in got), reverse=True)
    hit_g, hit_r = set(), set()
    for s, g, r in pairs:
        if s < threshold:
            break
        if g not in hit_g and r not in hit_r:
            hit_g.add(g)
            hit_r.add(r)
    return hit_g, hit_r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ttl", default=str(KG_TTL), help="knowledge graph to evaluate")
    a = ap.parse_args()
    kg = KnowledgeGraph(Ontology(config.ONTOLOGY_PATH), ttl=a.ttl)
    print(f"graph: {a.ttl}\n")
    tp_all = ret_all = exp_all = 0
    print(f"{'question':48s} {'expected':>8s} {'returned':>8s} {'precision':>9s} {'recall':>7s}")
    for q, gold_fn in TESTS:
        gold = {i["name"] for i in ITEMS if gold_fn(i)}
        got = {d["name"] for d in kg.answer(q).dishes}
        hit_g, hit_r = fuzzy_hits(gold, got)
        tp = len(hit_g)
        p = tp / len(got) if got else (1.0 if not gold else 0.0)
        r = tp / len(gold) if gold else 1.0
        tp_all, ret_all, exp_all = tp_all + tp, ret_all + len(got), exp_all + len(gold)
        print(f"{q[:48]:48s} {len(gold):8d} {len(got):8d} {p:9.2f} {r:7.2f}")
        for name in sorted(gold - hit_g)[:6]:
            print(f"      missing: {name}")
        for name in sorted(got - hit_r)[:6]:
            print(f"      extra:   {name}")
    print(f"\nmicro precision {tp_all / max(ret_all, 1):.2f}   micro recall {tp_all / max(exp_all, 1):.2f}   "
          f"({len(TESTS)} questions, {exp_all} expected dishes)")


if __name__ == "__main__":
    main()
