# Restaurant Ontology RAG + GraphRAG

RAG (Retrieval-Augmented Generation) nad doménou reštaurácií, prepojený s vlastnou OWL ontológiou
(`ontology/restaurant.owl`, vytvorená vo WebProtégé), 12 PDF dokumentmi z domény a znalostným grafom
(277 jedál) s dotazmi SPARQL. Podrobnosti a vyhodnotenie: **`DOKUMENTACIA.md`**.

## Rýchle spustenie

1. Spustiť **Ollama** (ikona v systémovej lište; model `gemma3:4b` – `ollama pull gemma3:4b`).
2. Dvojklik na **`run_app.bat`** → aplikácia **Ontology Kitchen** sa otvorí na http://localhost:8000.
   Modely sa načítajú pri štarte (~20 s), potom vyhľadávanie trvá < 1 s a odpoveď ~3–5 s.

Tip: `http://localhost:8000/?q=Which vegan dishes cost less than £15?` položí otázku hneď po otvorení.

### Webové rozhranie

- otázka (EN / SK) s prepínačmi **Use ontology**, **Rerank**, **Knowledge graph (SPARQL)** – na porovnanie variantov naživo,
- **Answer** – odpoveď LLM sa vypisuje priebežne, citácie `[n]` sú klikateľné,
- **Knowledge graph · SPARQL** – presný výsledok z grafu (tabuľka jedál s triedou, diétami a cenou) + vygenerovaný dotaz SPARQL,
- **Sources** – nájdené úryvky PDF, fakty ontológie a výsledok grafu so skóre,
- **Ontology** – nájdené koncepty a graf ich okolia (vrátane nájdených jedál ako individuí), tlačidlo na celú ontológiu,
- svetlá / tmavá téma (zelená paleta, písma Playfair Display + DM Sans).

## Inštalácia (nový počítač)

```powershell
cd restaurant-rag
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

PDF dokumenty nie sú súčasťou repozitára (patria ich vydavateľom) – stiahnu sa z pôvodných odkazov
v `data/pdfs/SOURCES.md`: `.\.venv\Scripts\python.exe download_pdfs.py`. Index (`index/`) a znalostný graf
(`data/kg/`) sú v repozitári, takže aplikácia funguje aj bez nich.

(`.\.venv\Scripts\activate` vyžaduje `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`;
všetky príkazy nižšie fungujú aj bez aktivácie.)

## Príkazy v poradí spracovania

```powershell
# 1. príprava dát
.\.venv\Scripts\python.exe extract_pdf_text.py   # PDF -> data/pdf_text/*.txt (pypdf)
.\.venv\Scripts\python.exe ingest.py             # PDF + ontológia -> chunky, embeddingy, BM25 index (index/)
.\.venv\Scripts\python.exe build_kg.py           # data/kg/dishes.json -> data/kg/restaurant_data.ttl (znalostný graf)
.\.venv\Scripts\python.exe extract_kg.py         # automatická extrakcia jedál z PDF lokálnym LLM -> data/kg/dishes_llm.json
                                                 #   + porovnanie s overenými dátami -> output/extraction_report.md

# 2. používanie
.\.venv\Scripts\python.exe server.py             # webové rozhranie (alebo run_app.bat)
.\.venv\Scripts\python.exe ask.py "Which pasta dishes are vegan?"
.\.venv\Scripts\python.exe ask.py --no-ontology "Which pasta dishes are vegan?"   # bez ontológie a grafu
.\.venv\Scripts\python.exe visualize.py          # celý graf ontológie -> output/ontology_graph.html

# 3. vyhodnotenie
.\.venv\Scripts\python.exe evaluate.py           # textové vyhľadávanie: s/bez ontológie, s/bez rerankingu
.\.venv\Scripts\python.exe evaluate_kg.py        # presnosť odpovedí znalostného grafu (precision / recall)
.\.venv\Scripts\python.exe evaluate_kg.py --ttl data/kg/restaurant_data_llm.ttl   # graf z automatickej extrakcie
.\.venv\Scripts\python.exe compare_graphrag.py   # textový RAG vs. GraphRAG s LLM -> output/graphrag_comparison.md
.\.venv\Scripts\python.exe judge.py --limit 6    # LLM-as-judge: vernosť a relevantnosť odpovedí
```

Graf z automatickej extrakcie (po `extract_kg.py`): `build_kg.py --input data/kg/dishes_llm.json --output data/kg/restaurant_data_llm.ttl`.

## LLM (generovanie odpovede)

| Backend | Nastavenie |
|---|---|
| Ollama (lokálne, zadarmo) – **predvolené** | model `gemma3:4b`; iný model: `$env:OLLAMA_MODEL="llama3.2"` |
| bez LLM | zobrazí sa nájdený kontext a výsledok SPARQL |

`auto`: Ollama, ak beží, inak bez LLM (aplikácia ukáže nájdené zdroje a výsledok SPARQL).

## Štruktúra

```
ontology/restaurant.owl      ontológia (WebProtégé export, RDF/XML) – nemení sa
data/pdfs/                   SOURCES.md – odkazy na 12 PDF (stiahnuť: download_pdfs.py)
data/pdf_text/               text extrahovaný z PDF (extract_pdf_text.py, nie je v repozitári)
data/kg/dishes.json          overené fakty z menu (277 jedál) – zdroj znalostného grafu
data/kg/restaurant_data.ttl  znalostný graf (RDF/Turtle) – individuá typované triedami ontológie
index/                       embeddingy + chunky pre textové vyhľadávanie

rag/ontology.py              načítanie OWL, verbalizácia, rozpoznanie konceptov, rozšírenie dopytu, graf
rag/documents.py             načítanie PDF, čistenie textu, chunking
rag/index.py                 embeddingy (sentence-transformers), BM25, reranker (cross-encoder)
rag/retriever.py             hybridné vyhľadávanie + RRF + reranking + znalostný graf
rag/kg.py                    GraphRAG: otázka -> SPARQL nad ontológiou + znalostným grafom
rag/llm.py                   prompt + volanie lokálneho LLM (Ollama)
rag/graphviz.py              dáta grafu pre webové rozhranie / pyvis
server.py + web/             webové rozhranie Ontology Kitchen (FastAPI, streamovanie, vis-network)

download_pdfs.py, extract_pdf_text.py, ingest.py, build_kg.py, extract_kg.py     príprava dát
ask.py, server.py, run_app.bat, visualize.py                    používanie
evaluate.py, evaluate_kg.py, compare_graphrag.py, judge.py      vyhodnotenie
output/                      reporty vyhodnotenia a graf ontológie (vytvoria ich skripty, nie sú v repozitári)
DOKUMENTACIA.md              dokumentácia k zadaniu
```

Vlastné PDF: nakopírovať do `data/pdfs/` a znova spustiť `ingest.py`.

