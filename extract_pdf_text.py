"""Step 0 of the data preparation: raw text from every PDF -> data/pdf_text/<name>.txt

The text (with === PAGE n === markers) is what was read when the verified knowledge-graph data
(data/kg/dishes.json) was written, and it is the same text that extract_kg.py sends to the LLM.

    python extract_pdf_text.py
"""
import sys

from pypdf import PdfReader

import config

OUT_DIR = config.ROOT / "data" / "pdf_text"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for pdf in sorted(config.PDF_DIR.glob("*.pdf")):
        pages = PdfReader(str(pdf)).pages
        text = "\n".join(f"=== PAGE {i} ===\n" + (p.extract_text() or "") for i, p in enumerate(pages, 1))
        out = OUT_DIR / (pdf.stem + ".txt")
        out.write_text(text, encoding="utf-8")
        print(f"{pdf.name:45s} {len(pages):3d} pages  {len(text.split()):6d} words  -> {out.name}")


if __name__ == "__main__":
    main()
