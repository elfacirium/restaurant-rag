"""Build the knowledge graph (ABox) from data/kg/dishes.json -> data/kg/restaurant_data.ttl

The ontology (ontology/restaurant.owl) is NOT modified. The new file only contains individuals
(restaurants, menus, dishes, places) typed with the ontology's classes and linked with the
ontology's properties. Things the ontology has no vocabulary for (calories, allergens, currency,
source document, ...) live in a separate namespace kg:.

Rules applied while building:
  * vegan => vegetarian (the kids menu defines "VE = suitable for vegans and vegetarians")
  * for menus with a complete allergen legend (Chapter One): no "G" declared => GlutenFree,
    no "D" (dairy) declared => LactoseFree   (marked as derived in kg:dietNote)

    python build_kg.py
"""
import argparse
import json
import re
import sys
from pathlib import Path

from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef, XSD

import config
from rag.ontology import Ontology

KG = Namespace("http://example.org/restaurant-kg#")
DATA_JSON = config.ROOT / "data" / "kg" / "dishes.json"
OUT_TTL = config.ROOT / "data" / "kg" / "restaurant_data.ttl"

DIETS = {"v": "Vegetarian", "ve": "Vegan", "gf": "GlutenFree", "lf": "LactoseFree", "h": "Halal"}
ALLERGENS = {"D": "dairy", "E": "egg", "G": "gluten", "U": "sulphites", "C": "celery", "M": "mustard",
             "N": "nuts", "T": "sesame", "S": "soya", "F": "fish", "B": "crustaceans", "O": "molluscs",
             "P": "peanuts", "L": "lupin"}
COMPLETE_ALLERGEN_LEGEND = {"12_private_dining_brochure.pdf"}


def slug(text: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", text.encode("ascii", "ignore").decode())
    return "".join(w[:1].upper() + w[1:] for w in words)[:60]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DATA_JSON), help="dishes JSON (default: verified dishes.json)")
    ap.add_argument("--output", default=str(OUT_TTL), help="Turtle file to write")
    args = ap.parse_args()
    out_ttl = Path(args.output)
    onto = Ontology(config.ONTOLOGY_PATH)
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))

    def cls(label):
        u = onto.uri(label)
        if u is None or u not in onto.classes:
            raise SystemExit(f"Unknown ontology class: {label}")
        return u

    def prop(label):
        u = onto.uri(label)
        if u is None:
            raise SystemExit(f"Unknown ontology property: {label}")
        return u

    P = {name: prop(name) for name in ("offersMenu", "containsItem", "servesDish", "hasPrice", "hasCuisine",
                                       "hasDietaryOption", "hostsEvent", "takesPlaceAt")}
    g = Graph()
    g.bind("kg", KG)
    g.bind("onto", Namespace("http://webprotege.stanford.edu/"))

    # shared individuals for cuisines and dietary options (the ontology has them only as classes)
    cuisine_ind, diet_ind = {}, {}
    for c in ("Italian", "European", "Asian", "Mediterranean"):
        cuisine_ind[c] = KG[f"cuisine_{c}"]
        g.add((cuisine_ind[c], RDF.type, cls(c)))
        g.add((cuisine_ind[c], RDFS.label, Literal(f"{c} cuisine")))
    for key, c in DIETS.items():
        diet_ind[key] = KG[f"diet_{c}"]
        g.add((diet_ind[key], RDF.type, cls(c)))
        g.add((diet_ind[key], RDFS.label, Literal(re.sub(r"(?<=[a-z])(?=[A-Z])", " ", c).lower())))

    restaurants = {}
    for r in data["restaurants"]:
        ru = KG[r["id"]]
        restaurants[r["id"]] = r
        g.add((ru, RDF.type, cls("Restaurant")))
        g.add((ru, RDFS.label, Literal(r["label"])))
        g.add((ru, P["hasCuisine"], cuisine_ind[r["cuisine"]]))
        for pl in r.get("places", []):
            pu = KG[pl["id"]]
            g.add((pu, RDF.type, cls(pl["type"])))
            g.add((pu, RDFS.label, Literal(pl["label"])))
            g.add((pu, KG.capacity, Literal(pl["capacity"], datatype=XSD.integer)))
            g.add((pu, KG.minimumSpend, Literal(pl["minimumSpend"], datatype=XSD.decimal)))
            g.add((pu, KG.deposit, Literal(pl["deposit"], datatype=XSD.decimal)))
            g.add((pu, KG.currency, Literal(pl["currency"])))
            g.add((pu, KG.source, Literal("12_private_dining_brochure.pdf")))
            g.add((pu, KG.page, Literal(pl["page"], datatype=XSD.integer)))
            g.add((pu, KG.locatedIn, ru))
            for layout, n in pl["layouts"].items():
                g.add((pu, KG.layout, Literal(f"{layout}: up to {n} guests")))
            ev = KG[pl["id"] + "Events"]
            g.add((ev, RDF.type, cls("SpecialCelebration")))
            g.add((ev, RDFS.label, Literal(f"Private events at {r['label']}: " + ", ".join(pl["events"]))))
            g.add((ev, P["takesPlaceAt"], pu))
            g.add((ru, P["hostsEvent"], ev))

    n_dishes, n_derived = 0, 0
    for m in data["menus"]:
        mu = KG[m["id"]]
        ru = KG[m["restaurant"]]
        g.add((mu, RDF.type, cls(m["type"])))
        g.add((mu, RDFS.label, Literal(m["label"])))
        g.add((mu, KG.source, Literal(m["source"])))
        g.add((mu, KG.page, Literal(m["page"], datatype=XSD.integer)))
        g.add((ru, P["offersMenu"], mu))
        if "price" in m:
            g.add((mu, P["hasPrice"], Literal(m["price"], datatype=XSD.decimal)))
            g.add((mu, KG.currency, Literal(m["currency"])))
        for it in m["items"]:
            du = KG[f"{m['id']}_{slug(it['n'])}"]
            n_dishes += 1
            g.add((du, RDF.type, cls(it["t"])))
            g.add((du, RDFS.label, Literal(it["n"])))
            g.add((mu, P["containsItem"], du))
            g.add((ru, P["servesDish"], du))
            g.add((du, KG.source, Literal(m["source"])))
            g.add((du, KG.page, Literal(it.get("pg", m["page"]), datatype=XSD.integer)))
            g.add((du, P["hasCuisine"], cuisine_ind[it.get("cu", m["cuisine"])]))
            for c in it.get("c", []):
                g.add((du, KG.category, Literal(c)))
            if "p" in it:
                g.add((du, P["hasPrice"], Literal(it["p"], datatype=XSD.decimal)))
                g.add((du, KG.currency, Literal(m["currency"])))
            if "k" in it:
                g.add((du, KG.calories, Literal(it["k"], datatype=XSD.integer)))
            if "note" in it:
                g.add((du, KG.note, Literal(it["note"])))
            flags = set(it.get("f", []))
            if "ve" in flags:
                flags.add("v")                                  # vegan => vegetarian
            codes = it.get("a")
            for ch in codes or "":
                g.add((du, KG.allergen, Literal(ALLERGENS[ch])))
            if codes is not None and m["source"] in COMPLETE_ALLERGEN_LEGEND:
                g.add((du, KG.allergensDeclared, Literal(True)))
                if "G" not in codes and "gf" not in flags:
                    flags.add("gf")
                    n_derived += 1
                    g.add((du, KG.dietNote, Literal("gluten free: derived - no gluten allergen (G) declared")))
                if "D" not in codes and "lf" not in flags:
                    flags.add("lf")
                    n_derived += 1
                    g.add((du, KG.dietNote, Literal("lactose free: derived - no dairy allergen (D) declared")))
            for f in sorted(flags):
                g.add((du, P["hasDietaryOption"], diet_ind[f]))

    out_ttl.write_text(
        "# Knowledge graph (ABox) for the restaurant ontology - generated by build_kg.py from dishes.json\n"
        "# Classes and properties come from ontology/restaurant.owl (prefix onto:), extensions use prefix kg:\n"
        + g.serialize(format="turtle"), encoding="utf-8")
    print(f"restaurants: {len(restaurants)}  menus: {len(data['menus'])}  dishes: {n_dishes}  "
          f"derived diet facts: {n_derived}  triples: {len(g)}")
    print(f"saved {out_ttl}")


if __name__ == "__main__":
    main()
