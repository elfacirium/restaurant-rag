"""Step 1-2 of the RAG workflow: ingest PDFs + ontology, chunk, embed and index.

    python ingest.py
"""
import config
from rag.documents import load_and_chunk
from rag.index import save_index
from rag.ontology import Ontology


def main():
    pdfs = sorted(config.PDF_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs in {config.PDF_DIR} - copy your PDFs there.")
    pdf_chunks = load_and_chunk(config.PDF_DIR, config.CHUNK_SIZE, config.CHUNK_OVERLAP)

    onto = Ontology(config.ONTOLOGY_PATH)
    onto_chunks = onto.verbalize()

    chunks = pdf_chunks + onto_chunks
    print(f"PDF files:        {len(pdfs)}")
    print(f"PDF chunks:       {len(pdf_chunks)}")
    print(f"Ontology:         {onto.stats()}")
    print(f"Ontology facts:   {len(onto_chunks)}")
    print(f"Embedding model:  {config.EMBED_MODEL}")
    save_index(chunks)
    print(f"Index saved to {config.INDEX_DIR} ({len(chunks)} chunks).")


if __name__ == "__main__":
    main()
