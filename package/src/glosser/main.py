import re
import time
import json
import pymupdf
from pathlib import Path, PurePath
from typing import Callable, Optional, Union
from .services import parser, pdf_transform, definitions
from .services.visual_design import ConfidenceVisualizer


def _source_to_confidence(source: str) -> str:
    """Map extraction source directly to confidence level without any critique call."""
    if source == "extracted":
        return "HIGH"
    if source == "inferred":
        return "MEDIUM"
    return "LOW"


async def annotate(
    path,
    out_path: Optional[Union[Path, str]] = None,
    GROQ_API_KEY: str = None,
    use_local_llm: bool = True,
    scaling: float = 1.2,
    find_references: bool = True,
    find_abbreviation: bool = True,
    find_symbols: bool = True,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
):
    """
    Annotate a PDF with citation titles and abbreviation/symbol full-forms.

    Returns [out_path, processed_count, log] where log contains detailed
    timing metrics and annotation counts per category.
    """

    def _progress(step: str, done: int, total: int):
        if progress_callback:
            progress_callback(step, done, total)

    dest = PurePath(path)
    processed = 0
    _ann_data: dict = {"citations": [], "abbreviations": [], "symbols": []}

    # ── Timing accumulators ──────────────────────────────────────────────────
    t_total_start = time.perf_counter()
    step_times: dict = {
        "scaling_seconds": 0.0,
        "references_seconds": 0.0,
        "abbreviations_seconds": 0.0,
        "symbols_seconds": 0.0,
        "save_seconds": 0.0,
    }

    # ── Annotation counters ──────────────────────────────────────────────────
    refs_log = {"found_total": 0, "annotated_count": 0}
    abbs_log = {"found_total": 0, "annotated_count": 0, "annotated_green": 0,
                "annotated_orange": 0, "annotated_red": 0}
    syms_log = {"found_total": 0, "annotated_count": 0, "annotated_green": 0,
                "annotated_orange": 0, "annotated_red": 0}

    try:
        original_doc = pymupdf.open(str(dest))

        # ── Scale ────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        scaled_doc, original_bboxes = pdf_transform.scale_content_horizontally(
            original_doc,
            scaling,
            progress_callback=lambda d, t: _progress("Scaling PDF pages", d, t),
        )
        _progress("Scaling PDF pages", 1, 1)
        step_times["scaling_seconds"] = round(time.perf_counter() - t0, 3)

        # ── Symbols ───────────────────────────────────────────────────────────
        if find_symbols:
            t0 = time.perf_counter()

            symbols = parser.find_symbols(
                original_doc,
                progress_callback=lambda d, t: _progress("Scanning for symbols", d, t),
            )
            _progress("Scanning for symbols", 1, 1)
            syms_log["found_total"] = len(symbols)

            initial_sym_counts: dict = {}
            for sym in symbols:
                sym_text = sym["text"]
                initial_sym_counts[sym_text] = initial_sym_counts.get(sym_text, 0) + 1

            # Build per-symbol context lookup (first occurrence wins)
            sym_context_map: dict = {}
            for sym in symbols:
                sym_text = sym["text"]
                if sym_text not in sym_context_map:
                    sym_context_map[sym_text] = sym.get("context", "")

            unique_syms = list(initial_sym_counts.keys())
            sym_meaning_map: dict = {}
            _progress("Extracting symbol meanings", 0, len(unique_syms))

            for i, sym_text in enumerate(unique_syms):
                context = sym_context_map.get(sym_text, "")
                # Strictly individual inference — no batch calls
                res = definitions.find_symbol_meaning(
                    sym_text,
                    context,
                    pdf_path=str(dest),
                    groq_api_key=GROQ_API_KEY,
                    use_local_llm=use_local_llm,
                )
                if res and res.get("meaning") not in ["NOT_FOUND", None, ""]:
                    source = res.get("source", "inferred")
                    # Bypass critique — map source directly to confidence
                    confidence = _source_to_confidence(source)
                    res["confidence"] = confidence
                    sym_meaning_map[sym_text] = res

                _progress("Extracting symbol meanings", i + 1, len(unique_syms))

            sym_occurrence_count: dict = {}
            sym_added_pages: dict = {}
            total_symbols = len(symbols)

            for i, sym in enumerate(symbols):
                sym_text = sym["text"]
                page = sym.get("page")
                sym_occurrence_count[sym_text] = sym_occurrence_count.get(sym_text, 0) + 1

                if sym_text not in sym_added_pages:
                    sym_added_pages[sym_text] = set()

                if sym_text not in sym_meaning_map:
                    _progress("Annotating symbols", i + 1, total_symbols or 1)
                    continue

                meaning = sym_meaning_map[sym_text].get("meaning")
                desc = sym_meaning_map[sym_text].get("description")
                confidence = sym_meaning_map[sym_text].get("confidence", "MEDIUM")

                if not meaning or meaning == "NOT_FOUND":
                    _progress("Annotating symbols", i + 1, total_symbols or 1)
                    continue

                if page in sym_added_pages[sym_text]:
                    _progress("Annotating symbols", i + 1, total_symbols or 1)
                    continue

                location_data = {
                    "page": page,
                    "column": sym.get("column"),
                    "bbox": sym.get("bbox"),
                }
                placed = pdf_transform.add_symbol_definition_to_margin(
                    doc=scaled_doc,
                    scaling_factor=scaling,
                    symbol=sym_text,
                    meaning=meaning,
                    description=desc,
                    original_location=location_data,
                    original_content_bboxes=original_bboxes,
                    confidence=confidence,
                )
                if placed:
                    processed += 1
                    sym_added_pages[sym_text].add(page)
                    syms_log["annotated_count"] += 1
                    if confidence == "HIGH":
                        syms_log["annotated_green"] += 1
                    elif confidence == "MEDIUM":
                        syms_log["annotated_orange"] += 1
                    else:
                        syms_log["annotated_red"] += 1
                    _ann_data["symbols"].append({
                        "text": sym_text,
                        "meaning": meaning,
                        "definition": f"{meaning}: {desc}" if desc and desc not in ("NOT_FOUND", "none") else meaning,
                        "page": page,
                        "confidence": confidence,
                    })

                _progress("Annotating symbols", i + 1, total_symbols or 1)

            step_times["symbols_seconds"] = round(time.perf_counter() - t0, 3)

        # ── Abbreviations ─────────────────────────────────────────────────────
        if find_abbreviation:
            t0 = time.perf_counter()

            abbs = parser.find_abbreviations(
                original_doc,
                progress_callback=lambda d, t: _progress("Scanning for abbreviations", d, t),
            )
            _progress("Scanning for abbreviations", 1, 1)
            abbs_log["found_total"] = len(abbs)

            initial_abbr_counts: dict = {}
            for abbr in abbs:
                abbr_text = abbr["text"]
                initial_abbr_counts[abbr_text] = initial_abbr_counts.get(abbr_text, 0) + 1

            unique_list = [t for t, c in initial_abbr_counts.items() if c >= 2]
            to_process_abbs = [{"id": abbr, "abbr": abbr} for abbr in unique_list]

            full_form_map: dict = {}
            _progress("Looking up full forms", 0, len(to_process_abbs))

            for i, item in enumerate(to_process_abbs):
                res = definitions.find_full_form(
                    item["abbr"],
                    pdf_path=str(dest),
                    groq_api_key=GROQ_API_KEY,
                    use_local_llm=use_local_llm,
                )
                if res and res.get("ans") not in ["NOT_FOUND", None, ""]:
                    source = "extracted" if not res.get("using_llm") else "inferred"
                    confidence = "HIGH" if source == "extracted" else "MEDIUM"
                    res["confidence"] = confidence
                    full_form_map[item["id"]] = res

                _progress("Looking up full forms", i + 1, len(to_process_abbs))

            abbr_occurrence_count: dict = {}
            abbr_last_annotated_page: dict = {}
            total_abbs = len(abbs)

            for i, abbr in enumerate(abbs):
                abbr_text = abbr["text"]
                page = abbr.get("page")
                abbr_occurrence_count[abbr_text] = abbr_occurrence_count.get(abbr_text, 0) + 1

                if abbr_text not in full_form_map:
                    _progress("Annotating abbreviations", i + 1, total_abbs or 1)
                    continue

                last_page = abbr_last_annotated_page.get(abbr_text)
                if last_page is not None and (page - last_page) <= 5:
                    _progress("Annotating abbreviations", i + 1, total_abbs or 1)
                    continue

                definition = full_form_map[abbr_text].get("ans")
                confidence = full_form_map[abbr_text].get("confidence", "MEDIUM")

                if not definition or definition == "NOT_FOUND":
                    _progress("Annotating abbreviations", i + 1, total_abbs or 1)
                    continue

                if last_page is None:
                    context = abbr.get("context", "")
                    if re.search(rf'\({re.escape(abbr_text)}\)', context):
                        abbr_last_annotated_page[abbr_text] = page
                        _progress("Annotating abbreviations", i + 1, total_abbs or 1)
                        continue

                location_data = {
                    "page": page,
                    "column": abbr.get("column"),
                    "bbox": abbr.get("bbox"),
                }
                placed = pdf_transform.add_definition_to_margin(
                    doc=scaled_doc,
                    scaling_factor=scaling,
                    main_word=abbr_text,
                    definition=definition,
                    original_location=location_data,
                    original_content_bboxes=original_bboxes,
                    confidence=confidence,
                )
                if placed:
                    processed += 1
                    abbr_last_annotated_page[abbr_text] = page
                    abbs_log["annotated_count"] += 1
                    if confidence == "HIGH":
                        abbs_log["annotated_green"] += 1
                    elif confidence == "MEDIUM":
                        abbs_log["annotated_orange"] += 1
                    else:
                        abbs_log["annotated_red"] += 1
                    _ann_data["abbreviations"].append({
                        "text": abbr_text,
                        "definition": definition,
                        "page": page,
                        "confidence": confidence,
                    })

                _progress("Annotating abbreviations", i + 1, total_abbs or 1)

            step_times["abbreviations_seconds"] = round(time.perf_counter() - t0, 3)

        # ── References ───────────────────────────────────────────────────────
        if find_references:
            t0 = time.perf_counter()

            refs_db = parser.build_references_db(
                original_doc,
                GROQ_API_KEY,
                use_local_llm=use_local_llm,
                progress_callback=lambda d, t: _progress("Building references database", d, t),
            )
            numeric_refs_db = refs_db.get("numeric", {})
            author_year_refs_db = refs_db.get("author_year", {})

            refs = parser.find_references(original_doc)
            total_refs = len(refs)
            refs_log["found_total"] = total_refs
            cited_refs = set()

            for i, ref in enumerate(refs):
                ref_format = ref.get("format_type", "NUMERIC_BRACKET")
                if ref_format == "NUMERIC_BRACKET":
                    ref_key = ref.get("number")
                    ref_info = numeric_refs_db.get(ref_key)
                else:
                    author_key = ref.get("author_key")
                    year = ref.get("year")
                    ref_key = f"{author_key}_{year}" if author_key and year else None
                    ref_info = author_year_refs_db.get(ref_key) if ref_key else None
                    if not ref_info and year:
                        ref_info = {"title": None, "year": year}

                if ref_key in cited_refs:
                    _progress("Annotating references", i + 1, total_refs or 1)
                    continue

                if ref_info:
                    title = ref_info.get("title")
                    year = ref_info.get("year")
                    definition = (
                        f"{title} ({year})".strip()
                        if title and title != "NOT_FOUND" and year and year != "NOT_FOUND"
                        else None
                    )
                else:
                    definition = None

                if definition:
                    location_data = {"page": ref["page"], "column": ref["column"], "bbox": ref["bbox"]}
                    placed = pdf_transform.add_definition_to_margin(
                        doc=scaled_doc,
                        scaling_factor=scaling,
                        main_word=f"{ref['text']}",
                        definition=definition,
                        original_location=location_data,
                        original_content_bboxes=original_bboxes,
                    )
                    if placed:
                        cited_refs.add(ref_key)
                        processed += 1
                        refs_log["annotated_count"] += 1
                        _ann_data["citations"].append({
                            "text": ref["text"],
                            "definition": definition,
                            "page": ref["page"],
                            "confidence": "HIGH",
                        })

                _progress("Annotating references", i + 1, total_refs or 1)

            step_times["references_seconds"] = round(time.perf_counter() - t0, 3)

        # ── Save ──────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        _progress("Saving annotated PDF", 0, 1)

        if not out_path:
            timestamp = int(time.time())
            out_path = Path(dest.parent) / f"{dest.stem}_glossed_{timestamp}.pdf"
        else:
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)

        pdf_transform.add_confidence_legend(scaled_doc)
        scaled_doc.save(str(out_path))

        json_path = out_path.with_suffix(".json")
        with open(str(json_path), "w", encoding="utf-8") as f:
            json.dump(_ann_data, f, ensure_ascii=False, indent=2)

        _progress("Saving annotated PDF", 1, 1)
        step_times["save_seconds"] = round(time.perf_counter() - t0, 3)

        total_elapsed = round(time.perf_counter() - t_total_start, 3)
        step_times["total_seconds"] = total_elapsed

        log = {
            "references": refs_log,
            "abbreviations": abbs_log,
            "symbols": syms_log,
            "timing": step_times,
        }

        return [out_path, processed, log]

    except Exception as e:
        raise RuntimeError(f"Annotation failed: {e}") from e