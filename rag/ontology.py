"""Loading the OWL ontology and using it inside the RAG pipeline.

The ontology is used in three ways:
  1. verbalize()      -> every class / property / individual becomes a short English text
                         ("ontology facts") that is indexed together with the PDF chunks;
  2. match()+expand() -> concepts mentioned in a question are found and expanded with
                         sub-/super-classes and related concepts (query expansion);
  3. edges()          -> graph of the ontology (or of the sub-graph used for one answer)
                         for visualisation.
"""
from __future__ import annotations

import re
from collections import defaultdict

from rdflib import BNode, Graph, OWL, RDF, RDFS, URIRef
from rdflib.collection import Collection


def split_camel(name: str) -> str:
    """'SignatureDish' -> 'Signature Dish'."""
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)


# English synonyms + Slovak stems so that Slovak questions still hit the English labels
# of the ontology. Matching is prefix-based, which handles most inflected forms
# (dezert, dezerty, dezertov ...).
SYNONYMS: dict[str, list[str]] = {
    "Restaurant": ["reštaur", "restaur"],
    "OceanPearlRestaurant": ["ocean pearl", "oceán"],
    "Dish": ["jedl", "jedál"],
    "Menu": ["menu", "jedálny lístok", "jedálnom lístku"],
    "RegularMenu": ["regular", "štandardn", "hlavné menu"],
    "KidsMenu": ["kids", "children", "detsk", "deti"],
    "SeasonalMenu": ["seasonal", "sezón", "jesen"],
    "TastingMenu": ["degustačné menu"],
    "Starter": ["appetizer", "predjed"],
    "MainCourse": ["main course", "hlavné jedl", "hlavný chod"],
    "MeatDish": ["meat", "mäs"],
    "SeafoodDish": ["seafood", "fish", "morsk", "plody mora", "ryb"],
    "PastaDish": ["pasta", "cestovin"],
    "Dessert": ["dezert", "sladk"],
    "Beverage": ["drink", "nápoj", "víno", "wine"],
    "SignatureDish": ["signature", "špecialit"],
    "Cuisine": ["cuisine", "kuchyn"],
    "Italian": ["talian"],
    "European": ["európ"],
    "Asian": ["ázij"],
    "Mediterranean": ["stredomor"],
    "DietaryOption": ["diet", "diét", "allerg", "alergé"],
    "Vegetarian": ["vegetari"],
    "Vegan": ["vegán"],
    "GlutenFree": ["gluten", "bezlepk", "lepok"],
    "LactoseFree": ["lactose", "laktóz", "bezlakt"],
    "Halal": ["halal"],
    "Event": ["event", "podujat", "akci"],
    "EntertainmentEvent": ["entertainment", "zábav"],
    "LiveMusicEvent": ["music", "hudb", "jazz"],
    "SpecialCelebration": ["celebrat", "oslav", "narodenin", "svadb"],
    "CulinaryEvent": ["culinary", "kulinár"],
    "TastingEvent": ["tasting", "degust"],
    "DinnerEvent": ["dinner", "večer"],
    "DiningPlace": ["room", "hall", "seat", "table", "miestnos", "sál", "priestor", "stôl", "stol"],
    "PrivateRoom": ["private", "súkrom", "privát", "vip"],
    "PanoramicTable": ["panoram", "view", "výhľad"],
    "UnderwaterHall": ["underwater", "aquarium", "podmor", "pod vodou", "akvár"],
    "hasPrice": ["price", "cost", "cen", "stoj"],
    "hasDuration": ["duration", "how long", "trvá", "trvan", "dĺžk"],
}

_XSD = "http://www.w3.org/2001/XMLSchema#"
_KIND_TEXT = {"some": "at least one", "only": "only", "value": "the value"}


class Ontology:
    def __init__(self, path):
        g = Graph()
        g.parse(str(path), format="xml")
        self.g = g

        self.labels: dict = {s: str(o) for s, o in g.subject_objects(RDFS.label)}
        self.classes = {s for s in g.subjects(RDF.type, OWL.Class) if isinstance(s, URIRef)}
        self.obj_props = {s for s in g.subjects(RDF.type, OWL.ObjectProperty)}
        self.data_props = {s for s in g.subjects(RDF.type, OWL.DatatypeProperty)}
        self.individuals = {s for s in g.subjects(RDF.type, OWL.NamedIndividual)}

        self.parents: dict = defaultdict(set)
        self.children: dict = defaultdict(set)
        self.restrictions: dict = defaultdict(list)   # class -> [(prop, kind, filler)]
        self.definitions: dict = {}                     # class -> (named parts, restrictions)

        for c in self.classes:
            for sup in g.objects(c, RDFS.subClassOf):
                if isinstance(sup, URIRef):
                    if sup in self.classes:
                        self.parents[c].add(sup)
                        self.children[sup].add(c)
                else:
                    r = self._restriction(sup)
                    if r:
                        self.restrictions[c].append(r)
            for eq in g.objects(c, OWL.equivalentClass):
                lst = g.value(eq, OWL.intersectionOf)
                if lst is None:
                    continue
                items = list(Collection(g, lst))
                named = [i for i in items if isinstance(i, URIRef)]
                rs = [self._restriction(i) for i in items if isinstance(i, BNode)]
                self.definitions[c] = (named, [r for r in rs if r])

        self.domain = {p: g.value(p, RDFS.domain) for p in self.obj_props | self.data_props}
        self.range = {p: g.value(p, RDFS.range) for p in self.obj_props | self.data_props}

        self.types = {i: [t for t in g.objects(i, RDF.type) if t in self.classes] for i in self.individuals}
        self.instances: dict = defaultdict(list)
        for i, ts in self.types.items():
            for t in ts:
                self.instances[t].append(i)
        self.assertions = [
            (i, p, o)
            for i in self.individuals
            for p, o in g.predicate_objects(i)
            if p in self.obj_props or p in self.data_props
        ]

        self._by_label = {self.name(u): u for u in self.classes | self.obj_props | self.data_props | self.individuals}
        self._patterns = self._build_patterns()

    # ------------------------------------------------------------------ helpers
    def _restriction(self, node):
        prop = self.g.value(node, OWL.onProperty)
        if prop is None:
            return None
        for pred, kind in ((OWL.someValuesFrom, "some"), (OWL.allValuesFrom, "only"), (OWL.hasValue, "value")):
            v = self.g.value(node, pred)
            if v is not None:
                return (prop, kind, v)
        return None

    def name(self, u) -> str:
        if u is None:
            return "Thing"
        if u in self.labels:
            return self.labels[u]
        s = str(u)
        if s.startswith(_XSD):
            return "xsd:" + s[len(_XSD):]
        return re.split(r"[#/]", s)[-1]

    def uri(self, label: str):
        return self._by_label.get(label)

    def descendants(self, c) -> set:
        out, stack = set(), [c]
        while stack:
            for ch in self.children.get(stack.pop(), ()):
                if ch not in out:
                    out.add(ch)
                    stack.append(ch)
        return out

    def ancestors(self, c) -> set:
        out, stack = set(), [c]
        while stack:
            for p in self.parents.get(stack.pop(), ()):
                if p not in out:
                    out.add(p)
                    stack.append(p)
        return out

    def stats(self) -> dict:
        return {
            "classes": len(self.classes),
            "object_properties": len(self.obj_props),
            "data_properties": len(self.data_props),
            "individuals": len(self.individuals),
            "triples": len(self.g),
        }

    # ------------------------------------------------------------ verbalization
    def _names(self, items) -> str:
        return ", ".join(sorted(self.name(i) for i in items))

    def verbalize(self) -> list[dict]:
        """Turn the ontology into small natural-language documents (one per entity)."""
        docs = []
        for c in sorted(self.classes, key=self.name):
            n = self.name(c)
            lines = [f"Ontology concept {n} ({split_camel(n).lower()}) is a class of the restaurant ontology."]
            if self.parents.get(c):
                lines.append(f"{n} is a kind of {self._names(self.parents[c])}.")
            anc = self.ancestors(c) - self.parents.get(c, set())
            if anc:
                lines.append(f"Indirectly {n} is also a {self._names(anc)}.")
            if self.children.get(c):
                lines.append(f"Kinds (subclasses) of {n}: {self._names(self.children[c])}.")
            for prop, kind, filler in self.restrictions.get(c, []):
                lines.append(f"Every {n} {self.name(prop)} {_KIND_TEXT[kind]} {self.name(filler)}.")
            if c in self.definitions:
                named, rs = self.definitions[c]
                cond = " and ".join(f"{self.name(p)} {k} {self.name(f)}" for p, k, f in rs)
                lines.append(f"{n} is defined as a {self._names(named)}" + (f" that {cond}." if cond else "."))
            for p in sorted(self.obj_props | self.data_props, key=self.name):
                if self.domain.get(p) == c:
                    lines.append(f"A {n} can have property {self.name(p)} with values of type {self.name(self.range.get(p))}.")
                if self.range.get(p) == c:
                    lines.append(f"A {self.name(self.domain.get(p))} can point to a {n} via {self.name(p)}.")
            if self.instances.get(c):
                lines.append(f"Known instances of {n}: {self._names(self.instances[c])}.")
            docs.append({"text": " ".join(lines), "source": "restaurant.owl", "page": n, "kind": "ontology"})

        for p in sorted(self.obj_props | self.data_props, key=self.name):
            kind = "object property" if p in self.obj_props else "data property"
            n = self.name(p)
            docs.append({
                "text": f"Ontology {kind} {n} ({split_camel(n).lower()}) connects {self.name(self.domain.get(p))} "
                        f"(domain) to {self.name(self.range.get(p))} (range).",
                "source": "restaurant.owl", "page": n, "kind": "ontology",
            })

        for i in sorted(self.individuals, key=self.name):
            n = self.name(i)
            lines = [f"{n} is an individual (instance) of {self._names(self.types.get(i, [])) or 'Thing'} in the ontology."]
            for s, p, o in self.assertions:
                if s == i:
                    lines.append(f"{n} {self.name(p)} {self.name(o)}.")
            docs.append({"text": " ".join(lines), "source": "restaurant.owl", "page": n, "kind": "ontology"})
        return docs

    # ----------------------------------------------------- concept matching
    def _build_patterns(self):
        pats = []
        for label, u in self._by_label.items():
            variants = {label.lower(), split_camel(label).lower()}
            variants |= {s.lower() for s in SYNONYMS.get(label, [])}
            for v in variants:
                if len(v) >= 3:
                    pats.append((re.compile(r"(?<!\w)" + re.escape(v), re.IGNORECASE), u, v))
        # longer variants first so 'kids menu' beats 'menu'
        pats.sort(key=lambda t: -len(t[2]))
        return pats

    def match(self, text: str) -> list:
        """Ontology entities mentioned in a question (by label, CamelCase split or synonym)."""
        found = []
        for rx, u, _ in self._patterns:
            if u not in found and rx.search(text):
                found.append(u)
        return found

    def expand(self, concepts) -> set:
        """Concepts + their neighbourhood in the ontology (used for query expansion)."""
        out = set()
        for c in concepts:
            out.add(c)
            if c in self.classes:
                out |= self.descendants(c)
                out |= self.parents.get(c, set())
                out |= {f for _, _, f in self.restrictions.get(c, []) if isinstance(f, URIRef)}
                out |= {p for p, _, _ in self.restrictions.get(c, [])}
                if c in self.definitions:
                    named, rs = self.definitions[c]
                    out |= set(named) | {f for _, _, f in rs}
                out |= set(self.instances.get(c, []))
            elif c in self.obj_props or c in self.data_props:
                for x in (self.domain.get(c), self.range.get(c)):
                    if x in self.classes:
                        out.add(x)
            elif c in self.individuals:
                out |= set(self.types.get(c, []))
        return {u for u in out if isinstance(u, URIRef) and not str(u).startswith(_XSD)}

    def most_specific(self, concepts) -> list:
        """Drop concepts that are ancestors of another matched concept ('dish' + 'pasta' -> 'pasta')."""
        cs = list(concepts)
        return [c for c in cs if not any(c in self.ancestors(o) for o in cs if o != c)]

    def query_focus(self, concepts, max_children: int = 4) -> set:
        """Concepts used for query expansion: the specific concepts + their subclasses,
        but only when the concept is narrow (a broad class like Dish or Restaurant would
        pull in the whole ontology and drown the question)."""
        focus = set()
        for c in self.most_specific(concepts):
            focus.add(c)
            desc = self.descendants(c)
            if 0 < len(desc) <= max_children:
                focus |= desc
        return focus

    def expansion_terms(self, concepts) -> list[str]:
        terms = set()
        for u in concepts:
            label = self.name(u)
            if u in self.obj_props or u in self.data_props:
                label = re.sub(r"^(has|is)(?=[A-Z])", "", label)      # hasPrice -> Price
            terms.add(split_camel(label))
        return sorted(terms)

    # ------------------------------------------------------------------ graph
    def edges(self, nodes=None) -> list[tuple]:
        """(source, target, label, kind) edges; restricted to `nodes` when given."""
        es = []
        for c in self.classes:
            for p in self.parents.get(c, ()):
                es.append((c, p, "subClassOf", "isa"))
            for prop, kind, f in self.restrictions.get(c, []):
                es.append((c, f, f"{self.name(prop)} ({kind})", "restriction"))
            if c in self.definitions:
                for prop, kind, f in self.definitions[c][1]:
                    es.append((c, f, f"≡ {self.name(prop)} ({kind})", "definition"))
        for p in self.obj_props:
            d, r = self.domain.get(p), self.range.get(p)
            if d is not None and r is not None:
                es.append((d, r, self.name(p), "property"))
        for i, ts in self.types.items():
            for t in ts:
                es.append((i, t, "type", "instance"))
        for s, p, o in self.assertions:
            if isinstance(o, URIRef):
                es.append((s, o, self.name(p), "assertion"))
        if nodes is not None:
            nodes = set(nodes)
            es = [e for e in es if e[0] in nodes and e[1] in nodes]
        # de-duplicate (restriction and property edge can coincide)
        seen, out = set(), []
        for e in es:
            key = (e[0], e[1], e[2])
            if key not in seen:
                seen.add(key)
                out.append(e)
        return out

    def node_kind(self, u) -> str:
        if u in self.individuals:
            return "individual"
        if u in self.classes:
            return "class"
        return "other"
