import json
import os
from pathlib import Path
from typing import List, Dict, Optional
from flask import Flask, render_template, send_from_directory, jsonify, request, abort
import webbrowser
import threading
import time


class AnnotationData:
    """Structured annotation data for interactive exploration."""

    def __init__(self):
        self.citations: List[Dict] = []
        self.abbreviations: List[Dict] = []
        self.symbols: List[Dict] = []

    def to_dict(self) -> Dict:
        return {
            "citations": self.citations,
            "abbreviations": self.abbreviations,
            "symbols": self.symbols,
            "stats": {
                "total_citations": len(self.citations),
                "total_abbreviations": len(self.abbreviations),
                "total_symbols": len(self.symbols),
                "total_annotations": len(self.citations) + len(self.abbreviations) + len(self.symbols),
            },
        }


def _load_annotation_data(annotated_dir: Path, stem: str) -> AnnotationData:
    """Load the sidecar JSON for a given PDF stem (e.g. '01' for 01.pdf)."""
    data = AnnotationData()
    json_path = annotated_dir / f"{stem}.json"
    if not json_path.exists():
        return data
    try:
        with open(str(json_path), "r", encoding="utf-8") as f:
            raw = json.load(f)
        data.citations = raw.get("citations", [])
        data.abbreviations = raw.get("abbreviations", [])
        data.symbols = raw.get("symbols", [])
    except Exception:
        pass
    return data


class InteractiveViewer:
    """
    Flask-based interactive viewer for annotated PDFs.

    Supports multi-paper browsing: scans `annotated_dir` for all *.pdf files
    and exposes an API for dynamic paper switching without a page reload.
    """

    def __init__(self, annotated_dir: str):
        self.annotated_dir = Path(annotated_dir).resolve()
        self.app = Flask(
            __name__,
            template_folder=str(Path(__file__).parent.parent / "templates"),
            static_folder=str(Path(__file__).parent.parent / "static"),
        )
        self._setup_routes()

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _list_papers(self) -> List[Dict]:
        """Return sorted list of available annotated PDFs."""
        pdfs = sorted(self.annotated_dir.glob("*.pdf"))
        return [{"filename": p.name, "stem": p.stem} for p in pdfs]

    def _safe_stem(self, stem: str) -> str:
        """Validate stem to prevent path traversal."""
        # Allow only alphanumeric, underscore, hyphen, dot
        import re
        if not re.fullmatch(r"[\w\-\.]+", stem):
            abort(400, "Invalid paper identifier.")
        return stem

    # ── Routes ─────────────────────────────────────────────────────────────────

    def _setup_routes(self):

        @self.app.route("/")
        def index():
            papers = self._list_papers()
            first = papers[0]["filename"] if papers else ""
            return render_template("viewer.html", initial_pdf=first, paper_count=len(papers))

        @self.app.route("/pdf/<path:filename>")
        def serve_pdf(filename):
            """Serve any annotated PDF from the annotated_papers directory."""
            safe_name = Path(filename).name  # strip any directory component
            target = self.annotated_dir / safe_name
            if not target.exists() or target.suffix.lower() != ".pdf":
                abort(404)
            return send_from_directory(str(self.annotated_dir), safe_name)

        @self.app.route("/api/papers")
        def list_papers():
            """List all annotated PDFs available for viewing."""
            return jsonify(self._list_papers())

        @self.app.route("/api/annotations")
        def get_annotations():
            """
            Return annotations for a specific paper.
            Query param: ?paper=01  (stem without extension)
            Falls back to the first available paper if omitted.
            """
            stem = request.args.get("paper", "")
            if not stem:
                papers = self._list_papers()
                stem = papers[0]["stem"] if papers else ""
            if not stem:
                return jsonify({"citations": [], "abbreviations": [], "symbols": [], "stats": {}})
            self._safe_stem(stem)
            data = _load_annotation_data(self.annotated_dir, stem)
            return jsonify(data.to_dict())

        @self.app.route("/api/filter")
        def filter_annotations():
            """Filter annotations by type, confidence, and search query."""
            stem = request.args.get("paper", "")
            if not stem:
                papers = self._list_papers()
                stem = papers[0]["stem"] if papers else ""

            annotation_type = request.args.get("type", "all")
            min_confidence = request.args.get("min_confidence", "LOW")
            search_query = request.args.get("search", "").lower()

            confidence_order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
            min_conf_level = confidence_order.get(min_confidence, 1)

            if stem:
                self._safe_stem(stem)
                ann_data = _load_annotation_data(self.annotated_dir, stem)
            else:
                ann_data = AnnotationData()

            def _keep(ann: Dict, meaning_key: str = "definition") -> bool:
                if confidence_order.get(ann.get("confidence", "MEDIUM"), 2) < min_conf_level:
                    return False
                if search_query:
                    text = (ann.get("text") or "").lower()
                    defn = (ann.get(meaning_key) or ann.get("meaning") or "").lower()
                    if search_query not in text and search_query not in defn:
                        return False
                return True

            filtered = {
                "citations": [a for a in ann_data.citations if _keep(a)] if annotation_type in ("all", "citations") else [],
                "abbreviations": [a for a in ann_data.abbreviations if _keep(a)] if annotation_type in ("all", "abbreviations") else [],
                "symbols": [a for a in ann_data.symbols if _keep(a, "meaning")] if annotation_type in ("all", "symbols") else [],
            }
            return jsonify(filtered)

        @self.app.route("/api/citation_network")
        def get_citation_network():
            """Citation network for D3.js visualisation."""
            stem = request.args.get("paper", "")
            if not stem:
                papers = self._list_papers()
                stem = papers[0]["stem"] if papers else ""

            citations = []
            if stem:
                self._safe_stem(stem)
                citations = _load_annotation_data(self.annotated_dir, stem).citations

            nodes = [{"id": "current_paper", "label": f"Paper {stem}", "type": "root", "radius": 10}]
            links = []
            for i, citation in enumerate(citations):
                node_id = f"cite_{i}"
                nodes.append({
                    "id": node_id,
                    "label": citation.get("definition", "Unknown"),
                    "type": "citation",
                    "confidence": citation.get("confidence", "MEDIUM"),
                    "radius": 5,
                })
                links.append({"source": "current_paper", "target": node_id, "strength": 1})

            return jsonify({"nodes": nodes, "links": links})

        @self.app.route("/api/log_interaction", methods=["POST"])
        def log_interaction():
            interaction = request.json
            print(f"[INTERACTION LOG] {interaction}")
            return jsonify({"status": "logged"})

    # ── Server ─────────────────────────────────────────────────────────────────

    def run(self, port: int = 5000, open_browser: bool = True):
        papers = self._list_papers()
        url = f"http://localhost:{port}"

        if open_browser:
            def _open():
                time.sleep(1.5)
                webbrowser.open(url)
            threading.Thread(target=_open, daemon=True).start()

        print(f"\n{'='*60}")
        print(f"  GlossVis Interactive Viewer")
        print(f"{'='*60}")
        print(f"  URL      : {url}")
        print(f"  Papers   : {len(papers)} annotated PDFs in {self.annotated_dir}")
        print(f"{'='*60}\n")
        print("Press Ctrl+C to stop the server\n")

        self.app.run(host="0.0.0.0", port=port, debug=False)


# ── Convenience entry-point ────────────────────────────────────────────────────

def launch_viewer(annotated_dir: str, port: int = 5000, open_browser: bool = True):
    """
    Launch the multi-paper interactive viewer.

    Args:
        annotated_dir: Directory containing annotated PDFs and their sidecar JSON files.
        port: Flask server port.
        open_browser: Whether to open the browser automatically.
    """
    viewer = InteractiveViewer(annotated_dir)
    viewer.run(port=port, open_browser=open_browser)


# ── Legacy single-paper helper (backwards compatibility) ──────────────────────

def extract_annotations_from_pdf(pdf_path: str) -> AnnotationData:
    """Load annotation metadata from the sidecar JSON for a single PDF."""
    p = Path(pdf_path).resolve()
    return _load_annotation_data(p.parent, p.stem)