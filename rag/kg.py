"""Knowledge-graph part of the GraphRAG: question -> SPARQL -> exact facts.

The question is translated into SPARQL deterministically (no LLM), using the ontology:
  * ontology concepts found in the question (rag/ontology.py match) become class constraints
      dish classes (Starter, PastaDish, ...)   ->  ?dish a/rdfs:subClassOf* onto:<class>
      dietary options (Vegan, Halal, ...)      ->  ?dish onto:hasDietaryOption [ a onto:<class> ]
      cuisines (Italian, Asian, ...)           ->  ?dish onto:hasCuisine [ a onto:<class> ]
      menu types (KidsMenu, TastingMenu, ...)  ->  ?menu onto:containsItem ?dish . ?menu a onto:<class>
      dining places (PrivateRoom, ...)         ->  separate query for places (capacity, minimum spend)
  * numbers in the question become price filters ("under £15", "lacnejšie ako 20 eur")
  * "cheapest" / "most expensive" become ORDER BY ... LIMIT
  * "without nuts" / "bez orechov" becomes FILTER NOT EXISTS { ?dish kg:allergen "nuts" }

Because subclasses are followed with rdfs:subClassOf*, the ontology hierarchy is used for reasoning:
asking for MainCourse also returns PastaDish, SeafoodDish and MeatDish individuals.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from rdflib import Graph, Literal, URIRef

import config
from rag.ontology import Ontology

KG_TTL = config.ROOT / "data" / "kg" / "restaurant_data.ttl"
ONTO_NS = "http://webprotege.stanford.edu/"
PREFIXES = (
    "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
    f"PREFIX onto: <{ONTO_NS}>\n"
    "PREFIX kg:   <http://example.org/restaurant-kg#>\n"
)
MAX_ROWS = 40

# dish categories that are not classes in the ontology (stored as kg:category)
CATEGORY_WORDS = {
    "wine": ["wine", "vín", "vino", "víno"],
    "cocktail": ["cocktail", "koktejl", "koktail"],
    "non-alcoholic": ["non-alcoholic", "alcohol free", "nealko"],
    "pizza": ["pizz"],
    "soup": ["soup", "polievk"],
    "salad": ["salad", "šalát"],
    "risotto": ["risott", "rizot"],
    "cheese": ["cheese", "syr"],
    "side": ["side dish", "sides", "príloh"],
    "canape": ["canap", "kanap", "jednohubk"],
}
ALLERGEN_WORDS = {
    "nuts": ["nut", "orech", "orieš"], "egg": ["egg", "vajec", "vajc", "vajíč"], "fish": ["fish", "rýb", "ryb"],
    "soya": ["soy", "sój", "soj"], "sesame": ["sesame", "sezam"], "celery": ["celery", "zeler"],
    "mustard": ["mustard", "horčic"], "crustaceans": ["crustacean", "kôrovc"], "molluscs": ["mollusc", "mäkkýš"],
    "peanuts": ["peanut", "arašid"], "sulphites": ["sulphite", "sulfite", "siričitan"],
}
_NUM = r"(?P<cur1>£|\$|€)?\s*(?P<num>\d+(?:[.,]\d+)?)\s*(?P<cur2>£|\$|€|eur\w*|libi?er\w*|libr\w*|pounds?|gbp|dollars?|dolár\w*|usd)?"
_UNDER = re.compile(r"\b(?:under|below|less than|cheaper than|up to|at most|max(?:imum)?|pod|menej ako|"
                    r"lacnejš\w*(?: ako| než)?|maximálne|najviac)\s*" + _NUM
                    + r"|\bdo\s+(?P<cur3>£|\$|€)?\s*(?P<num2>\d+(?:[.,]\d+)?)\s*(?P<cur4>£|\$|€|eur\w*|libi?er\w*|libr\w*|dolár\w*)",
                    re.I)   # Slovak 'do 15 eur' only together with a currency (English 'do' is a verb)
_OVER = re.compile(r"\b(?:over|above|more than|more expensive than|at least|nad|viac ako|"
                   r"drahš\w*(?: ako| než)?|aspoň|minimálne)\s*" + _NUM, re.I)
_CHEAP = re.compile(r"\b(cheapest|least expensive|lowest price|najlacn\w*)", re.I)
_EXPENSIVE = re.compile(r"\b(most expensive|priciest|highest price|najdrah\w*)", re.I)
_PROCESS = re.compile(r"\b(how (?:should|to|do|can)|why|explain|ako (?:sa|by|pripraviť|správne)|prečo)\b", re.I)
_WITHOUT = r"(?:without|no|free of|bez)\s+"


def _currency(sym: str | None) -> str | None:
    if not sym:
        return None
    s = sym.lower()
    if s in ("£",) or s.startswith(("lib", "pound", "gbp")):
        return "GBP"
    if s in ("$",) or s.startswith(("dollar", "dolár", "usd")):
        return "USD"
    if s in ("€",) or s.startswith("eur"):
        return "EUR"
    return None


@dataclass
class KGResult:
    applied: bool = False
    constraints: list = field(default_factory=list)     # human readable conditions
    queries: list = field(default_factory=list)         # SPARQL strings that were executed
    dishes: list = field(default_factory=list)
    menus: list = field(default_factory=list)
    places: list = field(default_factory=list)
    text: str = ""                                       # context block for the LLM

    @property
    def count(self) -> int:
        return len(self.dishes) + len(self.menus) + len(self.places)


class KnowledgeGraph:
    def __init__(self, onto: Ontology, ttl=KG_TTL):
        self.onto = onto
        self.g = Graph()
        self.g.parse(str(config.ONTOLOGY_PATH), format="xml")   # TBox: classes + hierarchy
        self.g.parse(str(ttl), format="turtle")                 # ABox: restaurants, menus, dishes
        self.uri = lambda label: onto.uri(label)
        dish = self.uri("Dish")
        self.dish_classes = {dish} | onto.descendants(dish)
        self.diet_classes = onto.descendants(self.uri("DietaryOption"))
        self.cuisine_classes = onto.descendants(self.uri("Cuisine"))
        self.menu_classes = onto.descendants(self.uri("Menu"))
        place = self.uri("DiningPlace")
        self.place_classes = {place} | onto.descendants(place)
        self.n_dishes = len(set(self.g.objects(None, self.uri("containsItem"))))

    # ------------------------------------------------------------ helpers
    def _o(self, label: str) -> str:
        """onto:<iri> with the readable label as comment-friendly text."""
        return "onto:" + str(self.uri(label))[len(ONTO_NS):]

    def _type(self, var: str, label: str, comment: bool = True) -> str:
        return f"{var} a/rdfs:subClassOf* {self._o(label)} ." + (f"   # {label}" if comment else "")

    @staticmethod
    def _words(question: str, table: dict) -> list:
        q = question.lower()
        return [k for k, words in table.items() if any(re.search(r"(?<!\w)" + re.escape(w), q) for w in words)]

    # ------------------------------------------------------------ parsing
    def parse(self, question: str) -> dict:
        concepts = self.onto.match(question)
        names = {u: self.onto.name(u) for u in concepts}
        spec = {
            "dish": [names[u] for u in concepts if u in self.dish_classes and names[u] not in ("Dish", "SignatureDish")],
            "generic_dish": any(names[u] == "Dish" for u in concepts),
            "diet": [names[u] for u in concepts if u in self.diet_classes],
            "cuisine": [names[u] for u in concepts if u in self.cuisine_classes],
            "menu": [names[u] for u in concepts if u in self.menu_classes],
            "place": [names[u] for u in concepts if u in self.place_classes],
            "category": self._words(question, CATEGORY_WORDS),
            "without": [a for a, ws in ALLERGEN_WORDS.items()
                        if any(re.search(_WITHOUT + re.escape(w), question, re.I) or
                               re.search(re.escape(w) + r"\w*[- ]free", question, re.I) for w in ws)],
            "price_max": None, "price_min": None, "currency": None, "order": None,
            "asks_price": any(names[u] == "hasPrice" for u in concepts),
        }
        # the dish class of a category is implied (e.g. 'wine' -> Beverage) - keep the category only
        if spec["category"] and set(spec["dish"]) <= {"Beverage"}:
            spec["dish"] = [d for d in spec["dish"] if d != "Beverage"]
        for rx, key in ((_UNDER, "price_max"), (_OVER, "price_min")):
            m = rx.search(question)
            if m:
                gd = m.groupdict()
                spec[key] = float((gd.get("num") or gd.get("num2")).replace(",", "."))
                spec["currency"] = spec["currency"] or _currency(
                    gd.get("cur1") or gd.get("cur2") or gd.get("cur3") or gd.get("cur4"))
        if _CHEAP.search(question):
            spec["order"] = "ASC"
        elif _EXPENSIVE.search(question):
            spec["order"] = "DESC"
        top = re.search(r"\b(?:top\s*)?(\d{1,2})\s+(?:\w+\s+)?(?:cheapest|most expensive|najlacn|najdrah)", question, re.I)
        spec["limit"] = int(top.group(1)) if top and spec["order"] else 5
        return spec

    def should_apply(self, question: str, spec: dict) -> bool:
        if _PROCESS.search(question) and not (spec["price_max"] or spec["price_min"] or spec["order"]):
            return False
        dishy = spec["dish"] or spec["generic_dish"] or spec["category"]
        return bool(dishy or spec["diet"] or spec["menu"] or spec["place"] or spec["without"]
                    or spec["price_max"] is not None or spec["price_min"] is not None or spec["order"])

    # ------------------------------------------------------------ queries
    def dish_query(self, spec: dict) -> str:
        w = []
        for d in spec["dish"]:
            if d in ("SeafoodDish", "MeatDish", "PastaDish"):      # also starters with seafood/meat/pasta
                cat = {"SeafoodDish": "seafood", "MeatDish": "meat", "PastaDish": "pasta"}[d]
                w.append(f"{{ {self._type('?dish', d, comment=False)} }} UNION {{ ?dish kg:category \"{cat}\" . }}"
                         f"   # {d} (or starter with category {cat})")
            else:
                w.append(self._type("?dish", d))
        for i, d in enumerate(spec["diet"]):
            w.append(f"?dish {self._o('hasDietaryOption')} ?diet{i} .   # hasDietaryOption\n  "
                     + self._type(f"?diet{i}", d))
        for i, c in enumerate(spec["cuisine"]):
            w.append(f"?dish {self._o('hasCuisine')} ?cuisine{i} .   # hasCuisine\n  "
                     + self._type(f"?cuisine{i}", c))
        for c in spec["category"]:
            w.append(f'?dish kg:category "{c}" .')
        for i, mt in enumerate(spec["menu"]):
            w.append(f"?m{i} {self._o('containsItem')} ?dish .   # containsItem\n  " + self._type(f"?m{i}", mt))
        for a in spec["without"]:
            w.append(f'?dish kg:allergensDeclared true .\n  FILTER NOT EXISTS {{ ?dish kg:allergen "{a}" }}')
        need_price = spec["price_max"] is not None or spec["price_min"] is not None or spec["order"]
        price = (f"?dish {self._o('hasPrice')} ?price ; kg:currency ?currency .   # hasPrice" if need_price else
                 f"OPTIONAL {{ ?dish {self._o('hasPrice')} ?price ; kg:currency ?currency }}   # hasPrice")
        filters = []
        if spec["price_max"] is not None:
            filters.append(f"?price <= {spec['price_max']}")
        if spec["price_min"] is not None:
            filters.append(f"?price >= {spec['price_min']}")
        if spec["currency"] and need_price:
            filters.append(f'?currency = "{spec["currency"]}"')
        order = f"ORDER BY {spec['order']}(?price) ?name" if spec["order"] else "ORDER BY ?restaurant ?name"
        limit = spec["limit"] if spec["order"] else MAX_ROWS + 1
        body = "\n  ".join(w + [
            f"?menu {self._o('containsItem')} ?dish ; rdfs:label ?menuName .",
            f"?rest {self._o('servesDish')} ?dish ; rdfs:label ?restaurant .   # servesDish",
            "?dish rdfs:label ?name ; a ?type ; kg:source ?source ; kg:page ?page .",
            "?type rdfs:label ?typeName .",
            price,
            "OPTIONAL { ?dish kg:calories ?kcal }",
        ] + ([f"FILTER({' && '.join(filters)})"] if filters else []))
        return (PREFIXES + "SELECT DISTINCT ?dish ?name ?typeName ?price ?currency ?kcal ?restaurant ?menuName ?source ?page\n"
                f"WHERE {{\n  {body}\n}}\n{order}\nLIMIT {limit}")

    def _add_diets(self, rows: list[dict]) -> list[dict]:
        has_diet = self.uri("hasDietaryOption")
        for r in rows:
            labels = sorted(str(self.g.value(o, URIRef("http://www.w3.org/2000/01/rdf-schema#label")))
                            for o in self.g.objects(URIRef(r["dish"]), has_diet))
            r["diets"] = ", ".join(labels)
        return rows

    def menu_query(self, spec: dict) -> str:
        types = "\n  ".join(self._type("?menu", m) for m in spec["menu"])
        return (PREFIXES + "SELECT ?menu ?menuName ?price ?currency ?restaurant ?source ?page (COUNT(?item) AS ?items)\n"
                f"WHERE {{\n  {types}\n"
                "  ?menu rdfs:label ?menuName ; kg:source ?source ; kg:page ?page .\n"
                f"  ?rest {self._o('offersMenu')} ?menu ; rdfs:label ?restaurant .   # offersMenu\n"
                f"  OPTIONAL {{ ?menu {self._o('hasPrice')} ?price ; kg:currency ?currency }}   # hasPrice\n"
                f"  OPTIONAL {{ ?menu {self._o('containsItem')} ?item }}\n"
                "}\nGROUP BY ?menu ?menuName ?price ?currency ?restaurant ?source ?page\nORDER BY ?price")

    def place_query(self, spec: dict) -> str:
        types = "\n  ".join(self._type("?place", p) for p in spec["place"])
        return (PREFIXES + "SELECT ?place ?name ?capacity ?minimumSpend ?deposit ?currency ?restaurant ?source ?page\n"
                "       (GROUP_CONCAT(DISTINCT ?layout; separator=\"; \") AS ?layouts)\n"
                "       (GROUP_CONCAT(DISTINCT ?eventName; separator=\"; \") AS ?events)\n"
                f"WHERE {{\n  {types}\n"
                "  ?place rdfs:label ?name ; kg:capacity ?capacity ; kg:source ?source ; kg:page ?page .\n"
                "  ?place kg:locatedIn ?rest . ?rest rdfs:label ?restaurant .\n"
                "  OPTIONAL { ?place kg:minimumSpend ?minimumSpend ; kg:deposit ?deposit ; kg:currency ?currency }\n"
                "  OPTIONAL { ?place kg:layout ?layout }\n"
                f"  OPTIONAL {{ ?ev {self._o('takesPlaceAt')} ?place ; rdfs:label ?eventName }}   # takesPlaceAt\n"
                "}\nGROUP BY ?place ?name ?capacity ?minimumSpend ?deposit ?currency ?restaurant ?source ?page")

    # ------------------------------------------------------------ answer
    @staticmethod
    def _rows(result) -> list[dict]:
        out = []
        for row in result:
            d = {}
            for k, v in row.asdict().items():
                d[k] = v.toPython() if isinstance(v, Literal) else (str(v) if isinstance(v, URIRef) else v)
            out.append(d)
        return out

    def describe(self, spec: dict) -> list[str]:
        c = []
        c += [f"class {d}" for d in spec["dish"]]
        c += [f"diet {d}" for d in spec["diet"]]
        c += [f"cuisine {d}" for d in spec["cuisine"]]
        c += [f"on a {m}" for m in spec["menu"]]
        c += [f"category {x}" for x in spec["category"]]
        c += [f"without {a}" for a in spec["without"]]
        cur = spec["currency"] or ""
        if spec["price_max"] is not None:
            c.append(f"price ≤ {spec['price_max']:g} {cur}".strip())
        if spec["price_min"] is not None:
            c.append(f"price ≥ {spec['price_min']:g} {cur}".strip())
        if spec["order"]:
            c.append("cheapest first" if spec["order"] == "ASC" else "most expensive first")
        return c

    def answer(self, question: str) -> KGResult:
        spec = self.parse(question)
        res = KGResult()
        if not self.should_apply(question, spec):
            return res
        res.applied = True
        res.constraints = self.describe(spec)
        dish_wanted = (spec["dish"] or spec["generic_dish"] or spec["diet"] or spec["category"] or spec["without"]
                       or spec["price_max"] is not None or spec["price_min"] is not None or spec["order"]
                       or (spec["menu"] and not spec["asks_price"]))
        if spec["place"]:
            q = self.place_query(spec)
            res.queries.append(q)
            res.places = self._rows(self.g.query(q))
            if not (spec["dish"] or spec["diet"] or spec["category"]):
                dish_wanted = False
        if spec["menu"]:
            q = self.menu_query(spec)
            res.queries.append(q)
            res.menus = self._rows(self.g.query(q))
        if dish_wanted:
            q = self.dish_query(spec)
            res.queries.append(q)
            res.dishes = self._add_diets(self._rows(self.g.query(q)))
        res.text = self._context(res)
        return res

    @staticmethod
    def _money(price, cur) -> str:
        if price is None:
            return ""
        sym = {"GBP": "£", "USD": "$", "EUR": "€"}.get(cur, "")
        return f"{sym}{float(price):.2f}"

    def _context(self, res: KGResult) -> str:
        lines = [f"Knowledge graph result (exact SPARQL query over the ontology and the restaurant data; "
                 f"conditions: {', '.join(res.constraints) or 'none'})."]
        for p in res.places:
            lines.append(f"- PLACE {p['name']} ({p['restaurant']}): capacity up to {p['capacity']} guests; "
                         f"minimum spend {self._money(p.get('minimumSpend'), p.get('currency'))}, deposit "
                         f"{self._money(p.get('deposit'), p.get('currency'))}; layouts: {p.get('layouts')}; "
                         f"events: {p.get('events')} [{p['source']} p.{p['page']}]")
        for m in res.menus:
            price = self._money(m.get("price"), m.get("currency")) or "no fixed price (dishes priced individually)"
            lines.append(f"- MENU {m['menuName']} ({m['restaurant']}): price {price}; {m['items']} items "
                         f"[{m['source']} p.{m['page']}]")
        if res.dishes or not (res.menus or res.places):
            shown = res.dishes[:MAX_ROWS]
            lines.append(f"{len(res.dishes) if len(res.dishes) <= MAX_ROWS else f'more than {MAX_ROWS}'} "
                         f"matching dishes in the knowledge graph ({self.n_dishes} dishes from 13 menus in 8 PDFs)"
                         + (":" if shown else ". None of the menus contains a dish with all these properties."))
            for d in shown:
                extra = [x for x in (self._money(d.get("price"), d.get("currency")),
                                     d.get("diets") or "") if x]
                lines.append(f"- {d['name']}" + (f" - {' - '.join(extra)}" if extra else "")
                             + f" ({d['restaurant']})")
        return "\n".join(lines)
