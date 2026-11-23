"""Side-by-side: text RAG vs. GraphRAG (text RAG + knowledge graph) on list / price questions.

For each question the LLM answers twice - once with the PDF chunks + ontology facts only, once with
the exact SPARQL result added. We count how many of the expected dishes the answer mentions and
how many wrong dish names it lists. Output: output/graphrag_comparison.md

    python compare_graphrag.py
"""
import re
import sys
import time

import config
from evaluate_kg import ITEMS, TESTS
from rag.llm import stream_answer, warm_up_ollama
from rag.retriever import Retriever

sys.stdout.reconfigure(encoding="utf-8")
QUESTIONS = [0, 1, 2, 3, 8, 10, 11]          # subset of evaluate_kg.TESTS (indices) - list-type questions


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", s.lower())


def key_words(name: str) -> list[str]:
    """Distinctive words of a dish name, used to detect it in a free-text answer."""
    stop = {"with", "and", "the", "a", "of", "in", "on", "children", "s", "set", "menu", "dessert", "canap"}
    return [w for w in norm(name.split("(")[0]).split() if len(w) > 3 and w not in stop][:3]


def mentioned(answer: str, name: str) -> bool:
    a = norm(answer)
    words = key_words(name)
    return bool(words) and all(w in a for w in words)


def main():
    ret = Retriever()
    warm_up_ollama()
    all_names = {i["name"] for i in ITEMS}
    rows, md = [], ["# Text RAG vs. GraphRAG\n", f"LLM: {config.OLLAMA_MODEL} · {time.strftime('%Y-%m-%d')}\n"]
    for qi in QUESTIONS:
        q, gold_fn = TESTS[qi]
        gold = {i["name"] for i in ITEMS if gold_fn(i)}
        md.append(f"\n## {q}\nExpected ({len(gold)}): " + "; ".join(sorted(gold)) + "\n")
        for label, use_kg in (("text RAG", False), ("GraphRAG", True)):
            r = ret.search(q, use_kg=use_kg)
            ans = "".join(stream_answer(q, r.results, "ollama"))
            hit = {n for n in gold if mentioned(ans, n)}
            # a wrong name whose key words are all contained in a found correct name is the same mention
            # ("Pasta bolognese" inside "Vegan pasta bolognese") - not counted as an error
            wrong = {n for n in all_names - gold if mentioned(ans, n)
                     and not any(set(key_words(n)) <= set(key_words(h)) for h in hit)}
            rows.append((q, label, len(gold), len(hit), len(wrong)))
            md.append(f"\n**{label}** - found {len(hit)}/{len(gold)}, wrong {len(wrong)}\n\n> "
                      + ans.replace("\n", "\n> ") + "\n")
            print(f"{label:9s} found {len(hit):2d}/{len(gold):2d}  wrong {len(wrong):2d}  | {q}")
    md.append("\n## Summary\n\n| Question | Text RAG found / wrong | GraphRAG found / wrong |\n|---|---|---|\n")
    for q in dict.fromkeys(r[0] for r in rows):
        t = next(r for r in rows if r[0] == q and r[1] == "text RAG")
        g = next(r for r in rows if r[0] == q and r[1] == "GraphRAG")
        md.append(f"| {q} | {t[3]}/{t[2]} · {t[4]} wrong | {g[3]}/{g[2]} · {g[4]} wrong |\n")
    tot = lambda lab, k: sum(r[k] for r in rows if r[1] == lab)  # noqa: E731
    exp = tot("text RAG", 2)
    md.append(f"| **total** | **{tot('text RAG', 3)}/{exp} · {tot('text RAG', 4)} wrong** | "
              f"**{tot('GraphRAG', 3)}/{exp} · {tot('GraphRAG', 4)} wrong** |\n")
    print(f"\nTOTAL  text RAG {tot('text RAG', 3)}/{exp} found, {tot('text RAG', 4)} wrong   |   "
          f"GraphRAG {tot('GraphRAG', 3)}/{exp} found, {tot('GraphRAG', 4)} wrong")
    out = config.OUTPUT_DIR / "graphrag_comparison.md"
    out.write_text("".join(md), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
