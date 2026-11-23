"""Automatic extraction of the knowledge-graph data from the menu PDFs with a local LLM.

    PDF page --pypdf--> text --split--> segments --LLM (Ollama, JSON schema)--> dishes
        -> data/kg/dishes_llm.json   (same format as the manually verified data/kg/dishes.json)
        -> comparison with dishes.json -> output/extraction_report.md

What the LLM extracts (per dish): name, ontology class (only classes of restaurant.owl are allowed),
categories, price, calories, dietary labels, allergen codes.
What is configuration (MENU_PAGES below): which PDF page belongs to which menu / restaurant, the menu
type, currency and fixed menu price - the "skeleton" of the graph.

    python extract_kg.py                     # all menu pages, model from config (gemma3:4b)
    python extract_kg.py --model llama3.2    # another Ollama model
    python extract_kg.py --compare-only      # only re-run the comparison on an existing dishes_llm.json
    python build_kg.py --input data/kg/dishes_llm.json --output data/kg/restaurant_data_llm.ttl
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import time
import unicodedata

import requests
from pypdf import PdfReader

import config

GOLD_JSON = config.ROOT / "data" / "kg" / "dishes.json"
OUT_JSON = config.ROOT / "data" / "kg" / "dishes_llm.json"
REPORT = config.OUTPUT_DIR / "extraction_report.md"
SEGMENT_CHARS = 1500

# ---- configuration: the skeleton (PDF page -> menu). Menu metadata (restaurant, type, currency,
#      fixed price) is reused from dishes.json by menu id. "split" sends a page to two menus by section.
MENU_PAGES = [
    ("01_seafood_restaurant_alc_menu.pdf", 1, "SeafoodALC"),
    ("01_seafood_restaurant_alc_menu.pdf", 2, "SeafoodALC"),
    ("01_seafood_restaurant_alc_menu.pdf", 3, "SeafoodChildrensMenu"),
    ("02_seafood_restaurant_set_menu.pdf", 1, "SeafoodSetLunch"),
    ("03_kids_menu.pdf", 2, "PremierInnKidsMenu"),
    ("04_tasting_menu_tudor_farmhouse.pdf", 1, "TudorTastingMenu"),
    ("05_tasting_menu_wine_pairing.pdf", 1, "GordonRamsayTastingMenu"),
    ("06_italian_vegetarian_menu.pdf", 1, "BrioVegetarianMenu"),
    ("07_italian_dietary_allergen_guide.pdf", 2, "VapianoMenu"),
    ("07_italian_dietary_allergen_guide.pdf", 3, "VapianoMenu"),
    ("07_italian_dietary_allergen_guide.pdf", 4, "VapianoMenu"),
    ("07_italian_dietary_allergen_guide.pdf", 5, "VapianoMenu"),
    ("12_private_dining_brochure.pdf", 2, "ChapterOneDrinks"),
    ("12_private_dining_brochure.pdf", 3, "ChapterOneDrinks"),
    ("12_private_dining_brochure.pdf", 4, "split:ChapterOneCanapes|ChapterOneChildrensMenu"),
    ("12_private_dining_brochure.pdf", 5, "ChapterOneFunctionMenu"),
    ("12_private_dining_brochure.pdf", 6, "ChapterOneSpecialSetMenu"),
]

CLASSES = ["Starter", "MainCourse", "MeatDish", "SeafoodDish", "PastaDish", "Dessert", "Beverage", "Dish"]
DIET_ENUM = ["vegetarian", "vegan", "gluten_free", "lactose_free", "halal"]
DIET_CODE = {"vegetarian": "v", "vegan": "ve", "gluten_free": "gf", "lactose_free": "lf", "halal": "h"}

SCHEMA = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "section": {"type": "string"},
            "dish_class": {"type": "string", "enum": CLASSES},
            "categories": {"type": "array", "items": {"type": "string"}},
            "price": {"type": ["number", "null"]},
            "calories": {"type": ["integer", "null"]},
            "diets": {"type": "array", "items": {"type": "string", "enum": DIET_ENUM}},
            "allergen_codes": {"type": "string"},
            "cuisine": {"type": "string", "enum": ["Italian", "Mediterranean", "Asian", "European"]},
        },
        "required": ["name", "section", "dish_class", "categories", "price", "calories", "diets", "allergen_codes",
                     "cuisine"],
    }}},
    "required": ["items"],
}

SYSTEM = """You extract structured data from restaurant menu text (copied from a PDF, layout may be broken).
Return EVERY dish, drink or menu item in the text as one entry - do not skip items, do not invent items.
Fields:
- name: the dish name as written (short, without the long description).
- section: the menu heading the item is under (e.g. "STARTERS", "DESSERTS", "CHILDREN'S MENU", "CANAPES").
- dish_class: Starter (starters, appetisers, soups, salads, canapes, small plates), MainCourse (other mains,
  pizza, risotto, vegetarian mains), MeatDish (meat main course), SeafoodDish (fish/seafood main course),
  PastaDish (pasta, spaghetti, ravioli, lasagne, gnocchi as main), Dessert (desserts, ice cream, cheese
  course), Beverage (drinks, wine, cocktails), Dish (side dishes).
- categories: from seafood, meat, pasta, vegetable, cheese, risotto, pizza, salad, soup, side, dessert,
  wine, cocktail, non-alcoholic, canape, bread, snack, soft drink.
- price: the price NUMBER of the item (e.g. "£12.95" or "12.95" -> 12.95). Never use the calories
  (numbers followed by kcal/cal) as price. null if the item has no own price.
- calories: the number before "kcal"/"cal", else null.
- diets: only labels written in the text: "V" or "(v)" = vegetarian; "VE" or "(vegan)" = vegan; headings
  like "Vegetarian", "Lactose Free", "Does not contain gluten" apply to the items listed under them;
  "halal certified" applies only to the items it names. Empty list if none.
- allergen_codes: the allergen letters printed after the item (e.g. "D E G U" -> "DEGU"), else "".
  Legend: D dairy, E egg, G gluten, U sulphites, C celery, M mustard, N nuts, T sesame, S soya, F fish,
  B crustaceans, O molluscs, P peanuts, L lupin.
- cuisine: Italian (pasta, pizza, risotto, Italian names), Asian (Thai, Japanese, Indian, Chinese,
  Indonesian, Korean, miso, curry, sashimi...), Mediterranean (Spanish, Greek, Provencal...), otherwise European."""


# ---------------------------------------------------------------- extraction
def segments(text: str, size: int = SEGMENT_CHARS) -> list[str]:
    out, cur = [], ""
    for line in text.splitlines():
        if cur and len(cur) + len(line) > size:
            out.append(cur)
            cur = ""
        cur += line + "\n"
    if cur.strip():
        out.append(cur)
    return out


def llm_extract(text: str, model: str) -> list[dict]:
    r = requests.post(f"{config.OLLAMA_URL}/api/chat", json={
        "model": model, "stream": False, "format": SCHEMA, "keep_alive": "30m",
        "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 4096},
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": "Menu text:\n\n" + text}],
    }, timeout=600)
    r.raise_for_status()
    try:
        return json.loads(r.json()["message"]["content"])["items"]
    except (json.JSONDecodeError, KeyError):
        return []                                   # truncated / invalid output -> counted as missed items


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s)).strip()


def to_item(x: dict, page: int, default_page: int) -> dict:
    it = {"n": x["name"].strip(), "t": x["dish_class"], "c": sorted({c.lower() for c in x.get("categories", [])})}
    if isinstance(x.get("price"), (int, float)) and x["price"] > 0:
        it["p"] = round(float(x["price"]), 2)
    if isinstance(x.get("calories"), int) and x["calories"] > 0:
        it["k"] = x["calories"]
    diets = sorted({DIET_CODE[d] for d in x.get("diets", []) if d in DIET_CODE})
    if diets:
        it["f"] = diets
    codes = "".join(ch for ch in (x.get("allergen_codes") or "").upper() if ch in "DEGUCMNTSFBOPL")
    if codes:
        it["a"] = codes
    if x.get("cuisine") in ("Italian", "Mediterranean", "Asian", "European"):
        it["cu"] = x["cuisine"]
    if page != default_page:
        it["pg"] = page
    return it


def extract(model: str) -> dict:
    gold = json.loads(GOLD_JSON.read_text(encoding="utf-8"))
    menus = {m["id"]: {**{k: v for k, v in m.items() if k != "items"}, "items": []} for m in gold["menus"]}
    readers = {}
    t0 = time.time()
    for source, page, target in MENU_PAGES:
        reader = readers.setdefault(source, PdfReader(str(config.PDF_DIR / source)))
        text = reader.pages[page - 1].extract_text() or ""
        segs = segments(text)
        n = 0
        for seg in segs:
            for x in llm_extract(seg, model):
                if not x.get("name", "").strip():
                    continue
                if target.startswith("split:"):           # one PDF page, two menus -> decide by section
                    a, b = target[6:].split("|")
                    menu_id = b if re.search(r"child|kid", x.get("section", ""), re.I) else a
                else:
                    menu_id = target
                m = menus[menu_id]
                item = to_item(x, page, m["page"])
                dup = next((i for i in m["items"] if norm(i["n"]) == norm(item["n"])), None)
                if dup:                                     # same dish listed in several diet sections
                    dup["f"] = sorted(set(dup.get("f", [])) | set(item.get("f", [])))
                    if not dup["f"]:
                        dup.pop("f")
                    continue
                m["items"].append(item)
                n += 1
        print(f"  {source} p.{page}: {len(segs)} segment(s), {n} new items  ({time.time() - t0:.0f}s)", flush=True)
    data = {"_about": f"Automatically extracted by extract_kg.py with Ollama model {model} on "
                      f"{time.strftime('%Y-%m-%d')}. Menu skeleton from MENU_PAGES; dish facts from the LLM. "
                      "Same item keys as dishes.json.",
            "restaurants": gold["restaurants"], "menus": list(menus.values())}
    OUT_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved {OUT_JSON}")
    return data


# ---------------------------------------------------------------- comparison
def items_by_source(data: dict) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for m in data["menus"]:
        for it in m["items"]:
            out.setdefault(m["source"], []).append(it)
    return out


def similarity(a: str, b: str) -> float:
    a, b = norm(a.split("(")[0]), norm(b.split("(")[0])
    ta, tb = set(a.split()), set(b.split())
    jac = len(ta & tb) / max(len(ta | tb), 1)
    return max(jac, difflib.SequenceMatcher(None, a, b).ratio())


def match(gold: list[dict], pred: list[dict], threshold: float = 0.6) -> list[tuple[dict, dict]]:
    pairs = sorted(((similarity(g["n"], p["n"]), gi, pi) for gi, g in enumerate(gold) for pi, p in enumerate(pred)),
                   reverse=True)
    used_g, used_p, out = set(), set(), []
    for s, gi, pi in pairs:
        if s < threshold:
            break
        if gi not in used_g and pi not in used_p:
            used_g.add(gi)
            used_p.add(pi)
            out.append((gold[gi], pred[pi]))
    return out


def diets(it: dict) -> set:
    f = set(it.get("f", []))
    if "ve" in f:
        f.add("v")
    return f


def compare(pred_data: dict, model: str) -> None:
    gold_src = items_by_source(json.loads(GOLD_JSON.read_text(encoding="utf-8")))
    pred_src = items_by_source(pred_data)
    rows, tot = [], {"gold": 0, "pred": 0, "match": 0, "price_ok": 0, "price_n": 0, "diet_ok": 0,
                     "cls_ok": 0, "allg_ok": 0, "allg_n": 0, "price_wrong": []}
    for src in gold_src:
        g, p = gold_src[src], pred_src.get(src, [])
        pairs = match(g, p)
        price_n = sum(1 for a, _ in pairs if "p" in a)
        price_ok = sum(1 for a, b in pairs if "p" in a and abs(a["p"] - b.get("p", -1)) < 0.011)
        diet_ok = sum(1 for a, b in pairs if diets(a) == diets(b))
        cls_ok = sum(1 for a, b in pairs if a["t"] == b["t"])
        allg_n = sum(1 for a, _ in pairs if "a" in a)
        allg_ok = sum(1 for a, b in pairs if "a" in a and set(a["a"]) == set(b.get("a", "")))
        tot["price_wrong"] += [(a["n"], a["p"], b.get("p")) for a, b in pairs
                               if "p" in a and abs(a["p"] - b.get("p", -1)) >= 0.011][:3]
        rows.append((src, len(g), len(p), len(pairs), price_ok, price_n, diet_ok, cls_ok))
        for k, v in (("gold", len(g)), ("pred", len(p)), ("match", len(pairs)), ("price_ok", price_ok),
                     ("price_n", price_n), ("diet_ok", diet_ok), ("cls_ok", cls_ok), ("allg_ok", allg_ok),
                     ("allg_n", allg_n)):
            tot[k] += v

    pct = lambda a, b: f"{100 * a / b:.0f} %" if b else "-"  # noqa: E731
    md = [f"# Automatic extraction vs. verified data\n\nModel: `{model}` · {time.strftime('%Y-%m-%d')} · "
          f"reference: `data/kg/dishes.json` (manually verified)\n\n",
          "| PDF | dishes (reference) | extracted | matched | recall | precision | price correct | diets correct | class correct |\n",
          "|---|---|---|---|---|---|---|---|---|\n"]
    for src, ng, np_, nm, po, pn, do, co in rows:
        md.append(f"| {src} | {ng} | {np_} | {nm} | {pct(nm, ng)} | {pct(nm, np_)} | {po}/{pn} | {do}/{nm} | {co}/{nm} |\n")
    md.append(f"| **total** | **{tot['gold']}** | **{tot['pred']}** | **{tot['match']}** | "
              f"**{pct(tot['match'], tot['gold'])}** | **{pct(tot['match'], tot['pred'])}** | "
              f"**{pct(tot['price_ok'], tot['price_n'])}** | **{pct(tot['diet_ok'], tot['match'])}** | "
              f"**{pct(tot['cls_ok'], tot['match'])}** |\n")
    md.append(f"\nAllergen codes correct: {tot['allg_ok']}/{tot['allg_n']} "
              f"({pct(tot['allg_ok'], tot['allg_n'])}) of matched dishes that have codes.\n")
    if tot["price_wrong"]:
        md.append("\nExamples of wrong prices (reference -> extracted):\n\n")
        md += [f"- {n}: {a} -> {b}\n" for n, a, b in tot["price_wrong"][:10]]
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    REPORT.write_text("".join(md), encoding="utf-8")
    print("".join(md))
    print(f"saved {REPORT}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.OLLAMA_MODEL)
    ap.add_argument("--compare-only", action="store_true")
    a = ap.parse_args()
    if a.compare_only:
        data = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        model = re.search(r"model (\S+)", data["_about"]).group(1)
    else:
        print(f"Extracting with {a.model} ...")
        data, model = extract(a.model), a.model
    compare(data, model)


if __name__ == "__main__":
    main()
