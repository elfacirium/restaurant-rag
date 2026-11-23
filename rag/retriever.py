"""Ontology-aware hybrid retrieval.

    question
      -> ontology.match()            concepts mentioned in the question (EN / SK synonyms)
      -> concept + its subclasses    -> English expansion terms
      -> dense search  (question)                      ┐
      -> dense search  (ontology terms, English)       ├─ Reciprocal Rank Fusion -> top-k PDF chunks
      -> BM25 search   (question + ontology terms)    ┘
      + ontology facts about the matched concepts      -> extra context block for the LLM
      + knowledge graph: question -> SPARQL over ontology + restaurant_data.ttl -> exact facts (GraphRAG)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import config
from rag.index import Index, rerank_scores
from rag.kg import KG_TTL, KGResult, KnowledgeGraph
from rag.ontology import Ontology

MAX_FACTS = 3


@dataclass
class Retrieval:
    query: str
    concepts: list = field(default_factory=list)       # matched ontology URIs
    expanded: set = field(default_factory=set)         # neighbourhood (for the graph view)
    expansion_terms: list = field(default_factory=list)
    facts: list = field(default_factory=list)          # ontology fact chunks
    docs: list = field(default_factory=list)           # PDF chunks
    kg: KGResult = field(default_factory=KGResult)     # knowledge-graph (SPARQL) result

    @property
    def kg_chunk(self) -> list:
        if not self.kg.applied:
            return []
        return [{"text": self.kg.text, "source": "restaurant_data.ttl", "page": "SPARQL", "kind": "kg",
                 "score": None, "found_by": ["sparql"]}]

    @property
    def results(self) -> list:
        """Context passed to the LLM: knowledge-graph result, ontology facts, then documents."""
        return self.kg_chunk + self.facts + self.docs


def rrf(rankings: dict[str, list[int]], k: int = config.RRF_K) -> list[tuple[int, float, list[str]]]:
    scores: dict[int, float] = {}
    origin: dict[int, list[str]] = {}
    for name, ranking in rankings.items():
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
            origin.setdefault(idx, []).append(name)
    return sorted(((i, s, origin[i]) for i, s in scores.items()), key=lambda t: -t[1])


class Retriever:
    def __init__(self, index: Index | None = None, ontology: Ontology | None = None):
        self.index = index or Index()
        self.onto = ontology or Ontology(config.ONTOLOGY_PATH)
        self.pdf_ids = {i for i, c in enumerate(self.index.chunks) if c["kind"] == "pdf"}
        self.fact_ids = {c["page"]: i for i, c in enumerate(self.index.chunks) if c["kind"] == "ontology"}
        self.kg = KnowledgeGraph(self.onto) if KG_TTL.exists() else None

    def _pdf_only(self, hits, n):
        return [i for i, _ in hits if i in self.pdf_ids][:n]

    def search(self, query: str, k: int = config.TOP_K, use_ontology: bool = True,
               rerank: bool | None = None, use_kg: bool = True) -> Retrieval:
        r = Retrieval(query=query)
        if use_kg and use_ontology and self.kg is not None:
            r.kg = self.kg.answer(query)
        n = config.CANDIDATES
        wide = n * 3

        rankings = {"dense": self._pdf_only(self.index.dense(query, wide), n)}
        kw_query = query
        if use_ontology:
            r.concepts = self.onto.most_specific(self.onto.match(query))
            r.expansion_terms = self.onto.expansion_terms(self.onto.query_focus(r.concepts))
            r.expanded = self.onto.expand(r.concepts)
            if r.expansion_terms:
                terms = " ".join(r.expansion_terms)
                kw_query = f"{query} {terms}"
                rankings["dense_onto"] = self._pdf_only(self.index.dense(terms, wide), n)
            for c in r.concepts[:MAX_FACTS]:
                idx = self.fact_ids.get(self.onto.name(c))
                if idx is not None:
                    r.facts.append({**self.index.chunks[idx], "score": None, "found_by": ["ontology"]})
        rankings["bm25"] = self._pdf_only(self.index.keyword(kw_query, wide), n)

        fused = rrf(rankings)
        if not (config.RERANK if rerank is None else rerank):
            for idx, score, origin in fused[:k]:
                r.docs.append({**self.index.chunks[idx], "score": round(score, 4), "found_by": origin})
            return r

        # Reranking + similarity threshold
        cand = fused[:config.RERANK_CANDIDATES]
        rr_query = f"{query} ({', '.join(r.expansion_terms)})" if r.expansion_terms else query
        scores = rerank_scores(rr_query, [self.index.chunks[i]["text"] for i, _, _ in cand])
        ranked = sorted(zip(cand, scores), key=lambda t: -t[1])
        for (idx, _, origin), s in ranked:
            if s < config.MIN_RERANK_SCORE and r.docs:
                break
            r.docs.append({**self.index.chunks[idx], "score": round(s, 3), "found_by": origin + ["rerank"]})
            if len(r.docs) == k:
                break
        return r
