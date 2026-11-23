"""LLM-as-judge: evaluation of the generated answers (not only of retrieval).

For every test question the RAG answers it, then a judge LLM scores the answer 1-5:
  faithfulness - is everything in the answer supported by the retrieved context? (no hallucination)
  relevance    - does the answer actually answer the question?

    python judge.py                 # all test questions, with and without the ontology
    python judge.py --limit 6       # quicker run on the first 6 questions
"""
import argparse
import json
import re
import sys
import time

import config
from evaluate import TESTS
from rag.llm import build_prompt, complete, generate, resolve_backend
from rag.retriever import Retriever

sys.stdout.reconfigure(encoding="utf-8")

JUDGE_SYSTEM = """You are a strict evaluator of a retrieval-augmented QA system.
You get a QUESTION, the CONTEXT the system retrieved, and the system's ANSWER.
Score the ANSWER on two criteria, each an integer 1-5:
- faithfulness: 5 = every claim in the answer is supported by the context; 1 = mostly unsupported or invented.
  An answer that correctly says the context does not contain the information is faithful (5).
- relevance: 5 = the answer directly and completely answers the question; 1 = does not answer it.
Reply ONLY with JSON: {"faithfulness": <int>, "relevance": <int>, "reason": "<one short sentence>"}"""


def parse_scores(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    try:
        d = json.loads(m.group(0) if m else text)
        return {"faithfulness": int(d["faithfulness"]), "relevance": int(d["relevance"]),
                "reason": str(d.get("reason", ""))}
    except (ValueError, KeyError, TypeError, AttributeError):
        return {"faithfulness": None, "relevance": None, "reason": "unparseable judge output"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=len(TESTS))
    ap.add_argument("--backend", choices=["auto", "ollama"], default=None)
    a = ap.parse_args()
    backend = resolve_backend(a.backend)
    if backend == "none":
        raise SystemExit("No LLM available - start Ollama.")

    ret = Retriever()
    tests = TESTS[: a.limit]
    print(f"LLM: {backend} ({config.OLLAMA_MODEL}), "
          f"{len(tests)} questions\n")
    report = []
    summary = {}
    for name, use_onto in (("plain RAG", False), ("RAG + ontology", True)):
        f_scores, r_scores = [], []
        t0 = time.time()
        for q, _ in tests:
            r = ret.search(q, use_ontology=use_onto)
            answer, _ = generate(q, r.results, backend)
            verdict = parse_scores(complete(
                JUDGE_SYSTEM,
                f"QUESTION: {q}\n\n{build_prompt(q, r.results)}\n\nANSWER: {answer}",
                backend, json_mode=True))
            if verdict["faithfulness"] is not None:
                f_scores.append(verdict["faithfulness"])
                r_scores.append(verdict["relevance"])
            report.append({"setting": name, "question": q, "answer": answer, **verdict})
            print(f"[{name}] F={verdict['faithfulness']} R={verdict['relevance']}  {q}")
        summary[name] = (sum(f_scores) / max(len(f_scores), 1), sum(r_scores) / max(len(r_scores), 1),
                         len(f_scores), time.time() - t0)
        print()

    print(f"{'setting':18s} {'faithfulness':>13s} {'relevance':>10s} {'judged':>7s}")
    for name, (f, r, n, _) in summary.items():
        print(f"{name:18s} {f:13.2f} {r:10.2f} {n:7d}")

    config.OUTPUT_DIR.mkdir(exist_ok=True)
    out = config.OUTPUT_DIR / "judge_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nDetailed answers + judge reasons: {out}")


if __name__ == "__main__":
    main()
