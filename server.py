"""Custom web UI for the ontology RAG (FastAPI + plain HTML/JS).

    .venv\\Scripts\\python.exe server.py        -> http://localhost:8000

Designed to be fast:
  * models (embeddings, reranker, LLM) are loaded once at start-up and kept in memory,
  * retrieved sources are sent to the browser immediately,
  * the answer is streamed token by token while the LLM writes it.
"""
from __future__ import annotations

import json
import sys
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from rag.documents import doc_title
from rag.graphviz import graph_json
from rag.index import embed, rerank_scores
from rag.llm import resolve_backend, stream_answer, warm_up_ollama
from rag.retriever import Retriever

sys.stdout.reconfigure(encoding="utf-8")
WEB = Path(__file__).parent / "web"
app = FastAPI(title="Ontology Kitchen")
app.mount("/static", StaticFiles(directory=WEB), name="static")

STATE = {"retriever": None, "ready": False, "llm_ready": False}
READY = threading.Event()


def _load():
    t = time.time()
    ret = Retriever()
    embed(["warm up"])                       # load embedding model
    rerank_scores("warm up", ["warm up"])    # load reranker
    STATE["retriever"] = ret
    STATE["ready"] = True
    READY.set()
    print(f"Retrieval ready in {time.time() - t:.1f}s", flush=True)
    if resolve_backend() == "ollama":
        STATE["llm_ready"] = warm_up_ollama()
        print(f"LLM {config.OLLAMA_MODEL} loaded: {STATE['llm_ready']}  -> ready, ask away!", flush=True)
    else:
        print(f"LLM backend: {resolve_backend()}", flush=True)


@app.on_event("startup")
def startup():
    threading.Thread(target=_load, daemon=True).start()


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/status")
def status():
    backend = resolve_backend()
    model = config.OLLAMA_MODEL if backend == "ollama" else "-"
    out = {"ready": STATE["ready"], "llm_ready": STATE["llm_ready"], "backend": backend, "model": model}
    if STATE["ready"]:
        ret = STATE["retriever"]
        out["stats"] = ret.onto.stats()
        out["documents"] = len({c["source"] for c in ret.index.chunks if c["kind"] == "pdf"})
        out["chunks"] = len(ret.index.chunks)
        if ret.kg is not None:
            out["kg"] = {"dishes": ret.kg.n_dishes, "triples": len(ret.kg.g)}
    return out


@app.get("/api/graph")
def full_graph():
    onto = STATE["retriever"].onto
    return graph_json(onto)


class Ask(BaseModel):
    question: str
    use_ontology: bool = True
    rerank: bool = True
    use_kg: bool = True


def _source(c: dict, n: int) -> dict:
    if c["kind"] == "ontology":
        title, where = c["page"], "ontology fact"
    elif c["kind"] == "kg":
        title, where = "Knowledge graph", "SPARQL over ontology + restaurant_data.ttl"
    else:
        title, where = doc_title(Path(c["source"])), f"{c['source']} · page {c['page']}"
    text = c["text"].split("] ", 1)[1] if c["text"].startswith("[") else c["text"]
    return {"n": n, "kind": c["kind"], "title": title, "where": where, "text": text,
            "score": c["score"], "found_by": c["found_by"]}


MAX_GRAPH_INDIVIDUALS = 12


def _kg_payload(kg) -> dict | None:
    if not kg.applied:
        return None
    return {"constraints": kg.constraints, "queries": kg.queries, "count": len(kg.dishes),
            "dishes": kg.dishes[:40], "menus": kg.menus, "places": kg.places}


def _graph(onto, r) -> dict | None:
    """Concept neighbourhood + the individuals found in the knowledge graph (dish -> its class)."""
    rows = r.kg.dishes[:MAX_GRAPH_INDIVIDUALS] if r.kg.applied else []
    classes = {onto.uri(d["typeName"]) for d in rows} - {None}
    nodes = set(r.expanded) | classes
    if not nodes:
        return None
    g = graph_json(onto, nodes=nodes, highlight=r.concepts)
    for d in rows:
        g["nodes"].append({"id": d["dish"], "label": d["name"][:38], "kind": "individual", "hit": False})
        g["edges"].append({"from": d["dish"], "to": str(onto.uri(d["typeName"])), "label": "type", "kind": "instance"})
    for p in (r.kg.places if r.kg.applied else []):
        g["nodes"].append({"id": p["place"], "label": f"{p['name']} ({p['capacity']} guests)", "kind": "individual", "hit": False})
        for c in r.concepts:
            if c in onto.classes:
                g["edges"].append({"from": p["place"], "to": str(c), "label": "type", "kind": "instance"})
    return g


@app.post("/api/ask")
def ask(req: Ask):
    def events():
        if not READY.is_set():                 # wait for the models instead of failing
            yield json.dumps({"type": "status", "text": "Loading models, the question will run in a moment…"}) + "\n"
            if not READY.wait(timeout=180):
                yield json.dumps({"type": "error", "text": "Models failed to load - see the server window."}) + "\n"
                return
        ret = STATE["retriever"]
        t0 = time.time()
        r = ret.search(req.question, use_ontology=req.use_ontology, rerank=req.rerank, use_kg=req.use_kg)
        onto = ret.onto
        yield json.dumps({
            "type": "retrieval",
            "ms": int((time.time() - t0) * 1000),
            "concepts": [onto.name(u) for u in r.concepts],
            "terms": r.expansion_terms,
            "sources": [_source(c, i) for i, c in enumerate(r.results, 1)],
            "kg": _kg_payload(r.kg),
            "graph": _graph(onto, r),
        }, default=str) + "\n"
        t1 = time.time()
        print(f"[ask] {req.question!r}  retrieval {t1 - t0:.1f}s  concepts={[onto.name(u) for u in r.concepts]}"
              + (f"  graph: {r.kg.count} results" if r.kg.applied else ""), flush=True)
        try:
            for piece in stream_answer(req.question, r.results):
                yield json.dumps({"type": "token", "text": piece}) + "\n"
        except Exception as e:  # noqa: BLE001 - show any LLM error in the UI
            print(f"[ask] LLM error: {e}", flush=True)
            yield json.dumps({"type": "error", "text": f"LLM error: {e} - is Ollama running?"}) + "\n"
        print(f"[ask] answer in {time.time() - t1:.1f}s", flush=True)
        yield json.dumps({"type": "done", "ms": int((time.time() - t1) * 1000)}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


if __name__ == "__main__":
    import os

    host = os.getenv("HOST", "127.0.0.1")           # 0.0.0.0 in the cloud (Docker / Hugging Face Spaces)
    port = int(os.getenv("PORT", "8000"))
    print(f"Ontology Kitchen -> http://localhost:{port}   (keep this window open; Ctrl+C to stop)", flush=True)
    if host == "127.0.0.1":
        threading.Timer(2.0, lambda: webbrowser.open(f"http://localhost:{port}")).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
