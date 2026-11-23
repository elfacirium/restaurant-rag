"""Download the 12 domain PDFs from their original publishers (links in data/pdfs/SOURCES.md).

The PDFs are not stored in the repository because they belong to their publishers.

    python download_pdfs.py
"""
import re
import sys

import requests

import config

SOURCES = config.PDF_DIR / "SOURCES.md"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    config.PDF_DIR.mkdir(parents=True, exist_ok=True)
    rows = re.findall(r"^\| (\S+\.pdf) \| .*? \| (https?://\S+) \|$", SOURCES.read_text(encoding="utf-8"), re.M)
    for name, url in rows:
        target = config.PDF_DIR / name
        if target.exists():
            print(f"{name:45s} already there")
            continue
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=120)
            r.raise_for_status()
            if not r.content.startswith(b"%PDF"):
                raise ValueError("not a PDF (the page may have moved)")
            target.write_bytes(r.content)
            print(f"{name:45s} OK  {len(r.content) // 1024} KB")
        except Exception as e:  # noqa: BLE001
            print(f"{name:45s} FAILED: {e}")
    print("\nNext: python extract_pdf_text.py && python ingest.py")


if __name__ == "__main__":
    main()
