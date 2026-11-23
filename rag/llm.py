"""Augment + generate: build the grounded prompt and call an LLM.

Backends
  ollama    : local model via Ollama (https://ollama.com), e.g. `ollama pull gemma3:4b`
  none      : no LLM - returns the retrieved context (retrieval can still be demonstrated)
  auto      : ollama if it is running, else none
"""
from __future__ import annotations

import json
import re

import requests

import config

SYSTEM_PROMPT = """You are an assistant answering questions about restaurants, menus, dishes, diets and dining events.
Answer ONLY from the numbered context below (restaurant documents and facts from the restaurant ontology).
If the context does not contain the answer, say that you do not know.
Be precise: when a document defines labels, symbols or rules (e.g. "V = vegetarian", "VE = vegan",
"dishes are vegetarian unless marked vegan"), apply them literally. List only the items the context
explicitly marks as matching the question - never all items of a nearby list.
A "knowledge graph result" is an exact database query: it is complete and correct for its conditions.
For lists, prices, counts and comparisons answer from it: list EVERY item it contains (never shorten
that list, never add items that are not in it); if it says no dish matches, say so. Use the documents
only for explanations and details.
Format: start with one short sentence that answers the question; if there are several items,
continue with a bulleted list ("- item - price [n]"). Put the source number in square brackets right
after each fact, e.g. [3]. Do not start the answer with citations. Keep explanations short."""

_SLOVAK = re.compile(r"[áäčďéíĺľňóôŕšťúýž]|\b(aké|aký|aká|koľko|ktor\w*|čo|je|sú|máte|ponúkate|môž\w*)\b", re.I)


def question_language(question: str) -> str:
    return "Slovak" if _SLOVAK.search(question) else "English"


def build_prompt(question: str, results: list[dict]) -> str:
    parts = []
    for n, c in enumerate(results, start=1):
        where = {"ontology": f"ontology concept {c['page']}",
                 "kg": "knowledge graph - exact SPARQL result"}.get(c["kind"], f"{c['source']}, page {c['page']}")
        parts.append(f"[{n}] ({where})\n{c['text']}")
    # the exact knowledge-graph block goes last (closest to the question) - small models follow it better
    kg = [p for p, c in zip(parts, results) if c["kind"] == "kg"]
    parts = [p for p, c in zip(parts, results) if c["kind"] != "kg"] + kg
    lang = question_language(question)
    return ("Context:\n\n" + "\n\n".join(parts) + f"\n\nQuestion: {question}\n\n"
            f"Write the answer in {lang}.")


def system_for(question: str) -> str:
    lang = question_language(question)
    extra = (f"\nIMPORTANT: the user writes in {lang}. Write your whole answer in {lang} "
             f"(the documents are in English - translate the facts; dish names may stay in the original).")
    return SYSTEM_PROMPT + (extra if lang != "English" else "")


def _ollama_alive() -> bool:
    try:
        return requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=1.5).ok
    except requests.RequestException:
        return False


def resolve_backend(backend: str | None = None) -> str:
    backend = backend or config.LLM_BACKEND
    if backend != "auto":
        return backend
    if _ollama_alive():
        return "ollama"
    return "none"


OLLAMA_OPTIONS = {"temperature": 0, "num_ctx": 4096, "num_predict": 512}   # hard cap -> no runaway answers
KEEP_ALIVE = "30m"          # keep the model loaded in memory between questions


def warm_up_ollama() -> bool:
    """Load the model into memory in advance (first answer is then much faster)."""
    try:
        requests.post(f"{config.OLLAMA_URL}/api/generate",
                      json={"model": config.OLLAMA_MODEL, "keep_alive": KEEP_ALIVE}, timeout=300)
        return True
    except requests.RequestException:
        return False


def stream_answer(question: str, results: list[dict], backend: str | None = None):
    """Yields the answer piece by piece (Ollama streams tokens; without an LLM it yields once)."""
    backend = resolve_backend(backend)
    if backend != "ollama":
        yield generate(question, results, backend)[0]
        return
    body = {
        "model": config.OLLAMA_MODEL,
        "stream": True,
        "keep_alive": KEEP_ALIVE,
        "options": OLLAMA_OPTIONS,
        "messages": [{"role": "system", "content": system_for(question)},
                     {"role": "user", "content": build_prompt(question, results)}],
    }
    with requests.post(f"{config.OLLAMA_URL}/api/chat", json=body, stream=True, timeout=(5, 120)) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            msg = json.loads(line)
            piece = msg.get("message", {}).get("content", "")
            if piece:
                yield piece
            if msg.get("done"):
                break


def _ollama(prompt: str, system: str = SYSTEM_PROMPT, json_mode: bool = False) -> str:
    body = {
        "model": config.OLLAMA_MODEL,
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "options": OLLAMA_OPTIONS,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
    }
    if json_mode:
        body["format"] = "json"
    r = requests.post(f"{config.OLLAMA_URL}/api/chat", json=body, timeout=600)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def complete(system: str, prompt: str, backend: str | None = None, json_mode: bool = False) -> str:
    """Generic single LLM call (used by the LLM-as-judge evaluation)."""
    backend = resolve_backend(backend)
    if backend == "ollama":
        return _ollama(prompt, system, json_mode)
    raise RuntimeError("No LLM available - start Ollama.")


def generate(question: str, results: list[dict], backend: str | None = None) -> tuple[str, str]:
    """Returns (answer, backend_used)."""
    backend = resolve_backend(backend)
    prompt = build_prompt(question, results)
    if backend == "ollama":
        return _ollama(prompt, system_for(question)), backend
    return ("(No LLM configured - showing the retrieved context only. "
            "Start Ollama to get a generated answer.)"), "none"
