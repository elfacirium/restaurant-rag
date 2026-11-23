"""Command-line question answering.

    python ask.py "Which dishes are vegan?"
    python ask.py --no-ontology "Which dishes are vegan?"
    python ask.py               (interactive mode)
"""
import argparse
import sys

from rag.llm import generate
from rag.retriever import Retriever

sys.stdout.reconfigure(encoding="utf-8")


def answer(ret: Retriever, q: str, use_ontology: bool, backend: str | None):
    r = ret.search(q, use_ontology=use_ontology)
    print("\nOntology concepts:", ", ".join(ret.onto.name(u) for u in r.concepts) or "-")
    if r.kg.applied:
        print(f"Knowledge graph:   {r.kg.count} exact results  (conditions: {', '.join(r.kg.constraints) or '-'})")
    if r.expansion_terms:
        print("Query expansion:  ", ", ".join(r.expansion_terms))
    print("\nRetrieved context:")
    for n, c in enumerate(r.results, 1):
        where = {"ontology": f"ontology:{c['page']}", "kg": "knowledge graph (SPARQL)"}.get(
            c["kind"], f"{c['source']} p.{c['page']}")
        how = ("exact SPARQL result" if c["kind"] == "kg" else "ontology fact" if c["score"] is None
               else f"score={c['score']}  via {'+'.join(c['found_by'])}")
        print(f"  [{n}] {where:40s} {how}")
    text, used = generate(q, r.results, backend)
    print(f"\nAnswer ({used}):\n{text}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="*")
    ap.add_argument("--no-ontology", action="store_true", help="plain RAG baseline without the ontology")
    ap.add_argument("--backend", choices=["auto", "ollama", "none"], default=None)
    a = ap.parse_args()
    ret = Retriever()
    if a.question:
        answer(ret, " ".join(a.question), not a.no_ontology, a.backend)
        return
    while True:
        try:
            q = input("Question (empty = quit): ").strip()
        except EOFError:
            break
        if not q:
            break
        answer(ret, q, not a.no_ontology, a.backend)


if __name__ == "__main__":
    main()
