import time
import pymupdf
from pathlib import Path, PurePath
from typing import Callable, Optional
from .services import parser, pdf_transform, definitions


async def annotate(
    path,
    out_path: Optional[Path | str] = None,
    GROQ_API_KEY: str = None,
    use_local_llm: bool = True,
    scaling: float = 1.2,
    find_references: bool = True,
    find_abbreviation: bool = True,
    find_symbols: bool = True,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
):
    """
    Annotate a PDF with citation titles and abbreviation full-forms.

    progress_callback(step_name, done, total) is called at each major step
    so the CLI can display a live progress bar.
    """

    def _progress(step: str, done: int, total: int):
        if progress_callback:
            progress_callback(step, done, total)

    dest = PurePath(path)
    processed = 0

    try:
        original_doc = pymupdf.open(str(dest))

        scaled_doc, original_bboxes = pdf_transform.scale_content_horizontally(
            original_doc,
            scaling,
            progress_callback=lambda d, t: _progress("Scaling PDF pages", d, t)
        )
        _progress("Scaling PDF pages", 1, 1)

        if find_references:
            refs_db = parser.build_references_db(
                original_doc,
                GROQ_API_KEY,
                use_local_llm=use_local_llm,
                progress_callback=lambda d, t: _progress("Building references database", d, t)
            )
            numeric_refs_db = refs_db.get("numeric", {})
            author_year_refs_db = refs_db.get("author_year", {})

            refs = parser.find_references(original_doc)
            total_refs = len(refs)

            for i, ref in enumerate(refs):
                ref_format = ref.get("format_type", "NUMERIC_BRACKET")
                if ref_format == "NUMERIC_BRACKET":
                    ref_info = numeric_refs_db.get(ref.get("number"))
                else:
                    author_key = ref.get("author_key")
                    year = ref.get("year")
                    lookup_key = f"{author_key}_{year}" if author_key and year else None
                    ref_info = author_year_refs_db.get(lookup_key) if lookup_key else None
                    if not ref_info and year:
                        ref_info = {"title": None, "year": year}

                if ref_info:
                    title = ref_info.get("title")
                    year = ref_info.get("year")
                    if not title or title == "NOT_FOUND" or not year or year == "NOT_FOUND":
                        definition = None
                    else:
                        definition = f"{title} ({year})".strip()
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
                        processed += 1

                _progress("Annotating references", i + 1, total_refs if total_refs else 1)

        if find_abbreviation:
            abbs = parser.find_abbreviations(
                original_doc,
                progress_callback=lambda d, t: _progress("Scanning for abbreviations", d, t)
            )
            _progress("Scanning for abbreviations", 1, 1)

            initial_abbr_counts: dict = {}
            for abbr in abbs:
                abbr_text = abbr["text"]
                initial_abbr_counts[abbr_text] = initial_abbr_counts.get(abbr_text, 0) + 1

            unique_list = [t for t, c in initial_abbr_counts.items() if c >= 2]
            to_process_abbs = [{"id": abbr, "abbr": abbr} for abbr in unique_list]

            full_form_map: dict = {}
            for i, item in enumerate(to_process_abbs):
                res = definitions.find_full_form(item["abbr"], pdf_path=str(dest), groq_api_key=GROQ_API_KEY, use_local_llm=use_local_llm)
                if res:
                    full_form_map[item["id"]] = res
                _progress("Looking up full forms", i + 1, len(to_process_abbs))

            abbr_occurrence_count: dict = {}
            abbr_added_pages: dict = {}
            total_abbs = len(abbs)

            for i, abbr in enumerate(abbs):
                abbr_text = abbr["text"]
                page = abbr.get("page")
                abbr_occurrence_count[abbr_text] = abbr_occurrence_count.get(abbr_text, 0) + 1

                if abbr_text not in abbr_added_pages:
                    abbr_added_pages[abbr_text] = set()

                if abbr_text not in full_form_map:
                    _progress("Annotating abbreviations", i + 1, total_abbs if total_abbs else 1)
                    continue

                if abbr_occurrence_count[abbr_text] < 2:
                    _progress("Annotating abbreviations", i + 1, total_abbs if total_abbs else 1)
                    continue

                if page in abbr_added_pages[abbr_text]:
                    _progress("Annotating abbreviations", i + 1, total_abbs if total_abbs else 1)
                    continue

                definition = full_form_map[abbr_text].get("ans")
                using_llm = bool(full_form_map[abbr_text].get("using_llm"))

                if not definition or definition == "NOT_FOUND":
                    _progress("Annotating abbreviations", i + 1, total_abbs if total_abbs else 1)
                    continue

                location_data = {
                    "page": page,
                    "column": abbr.get("column"),
                    "bbox": abbr.get("bbox")
                }
                placed = pdf_transform.add_definition_to_margin(
                    doc=scaled_doc,
                    scaling_factor=scaling,
                    main_word=abbr_text,
                    definition=definition,
                    original_location=location_data,
                    original_content_bboxes=original_bboxes,
                    using_llm=using_llm,
                )
                if placed:
                    processed += 1
                    abbr_added_pages[abbr_text].add(page)

                _progress("Annotating abbreviations", i + 1, total_abbs if total_abbs else 1)

        if find_symbols:
            symbols = parser.find_symbols(
                original_doc,
                progress_callback=lambda d, t: _progress("Scanning for symbols", d, t)
            )
            _progress("Scanning for symbols", 1, 1)

            initial_sym_counts: dict = {}
            for sym in symbols:
                sym_text = sym["text"]
                initial_sym_counts[sym_text] = initial_sym_counts.get(sym_text, 0) + 1

            unique_syms_for_lookup = list(initial_sym_counts.keys())

            sym_meaning_map: dict = {}
            to_process_syms = []
            for sym_text in unique_syms_for_lookup:
                context = next((s.get("context", "") for s in symbols if s["text"] == sym_text), "")
                to_process_syms.append({"id": sym_text, "symbol": sym_text, "context": context})

            for i, item in enumerate(to_process_syms):
                res = definitions.find_symbol_meaning(item["symbol"], item["context"], pdf_path=str(dest), groq_api_key=GROQ_API_KEY, use_local_llm=use_local_llm)
                if res:
                    sym_meaning_map[item["id"]] = res
                _progress("Extracting symbol meanings", i + 1, len(to_process_syms))

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
                    _progress("Annotating symbols", i + 1, total_symbols if total_symbols else 1)
                    continue

                meaning = sym_meaning_map[sym_text].get("meaning")
                desc = sym_meaning_map[sym_text].get("description")
                is_inferred = sym_meaning_map[sym_text].get("source") == "inferred"

                if not meaning or meaning == "NOT_FOUND":
                    _progress("Annotating symbols", i + 1, total_symbols if total_symbols else 1)
                    continue

                if page in sym_added_pages[sym_text]:
                    _progress("Annotating symbols", i + 1, total_symbols if total_symbols else 1)
                    continue

                location_data = {
                    "page": page,
                    "column": sym.get("column"),
                    "bbox": sym.get("bbox")
                }
                placed = pdf_transform.add_symbol_definition_to_margin(
                    doc=scaled_doc,
                    scaling_factor=scaling,
                    symbol=sym_text,
                    meaning=meaning,
                    description=desc,
                    original_location=location_data,
                    original_content_bboxes=original_bboxes,
                    is_inferred=is_inferred,
                )
                if placed:
                    processed += 1
                    sym_added_pages[sym_text].add(page)

                _progress("Annotating symbols", i + 1, total_symbols if total_symbols else 1)

        _progress("Saving annotated PDF", 0, 1)
        if not out_path:
            timestamp = int(time.time())
            out_path = Path(dest.parent) / f"{dest.stem}_glossed_{timestamp}.pdf"
        else:
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)

        scaled_doc.save(str(out_path))
        _progress("Saving annotated PDF", 1, 1)

        return [out_path, processed]

    except Exception as e:
        raise RuntimeError(f"Annotation failed: {e}") from e