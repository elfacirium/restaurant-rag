"""Central configuration for the Restaurant Ontology RAG project."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

ONTOLOGY_PATH = ROOT / "ontology" / "restaurant.owl"
PDF_DIR = ROOT / "data" / "pdfs"
INDEX_DIR = ROOT / "index"
OUTPUT_DIR = ROOT / "output"

# Multilingual embedding model -> questions can be asked in English or Slovak.
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Chunking (characters). Sentence-aware splitting with overlap.
CHUNK_SIZE = 700
CHUNK_OVERLAP = 150

# Retrieval
TOP_K = int(os.getenv("TOP_K", "5"))   # chunks passed to the LLM (smaller online = faster CPU answers)
CANDIDATES = 30      # candidates taken from each retriever before fusion
RRF_K = 60           # reciprocal rank fusion constant

# Reranking: the top RERANK_CANDIDATES fused chunks are re-scored by a cross-encoder,
# chunks below MIN_RERANK_SCORE (similarity threshold) are dropped.
RERANK = os.getenv("RERANK", "1") == "1"
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
RERANK_CANDIDATES = 20
MIN_RERANK_SCORE = float(os.getenv("MIN_RERANK_SCORE", "-5.0"))

# LLM backends: "auto" | "ollama" | "none"
LLM_BACKEND = os.getenv("LLM_BACKEND", "auto")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")   # best Slovak + instruction following of the tested small models
