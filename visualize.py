"""Draw the whole ontology as an interactive graph -> output/ontology_graph.html

    python visualize.py
"""
import webbrowser

import config
from rag.graphviz import build_html
from rag.ontology import Ontology


def main():
    onto = Ontology(config.ONTOLOGY_PATH)
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    out = config.OUTPUT_DIR / "ontology_graph.html"
    out.write_text(build_html(onto, height="900px"), encoding="utf-8")
    print(f"Saved {out}  {onto.stats()}")
    webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main()
