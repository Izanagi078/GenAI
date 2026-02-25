import time
import pymupdf
from pathlib import Path, PurePath
from typing import Callable, Optional
from .services import parser, pdf_transform, definitions


async def annotate(
    path,
    GROQ_API_KEY: str,
    scaling: float = 1.2,
    find_references: bool = True,
    find_abbreviation: bool = True,
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
                progress_callback=lambda d, t: _progress("Building references database", d, t)
            )

            refs = parser.find_references(original_doc)
            total_refs = len(refs)

            for i, ref in enumerate(refs):
                ref_info = refs_db.get(ref['number'])
                if ref_info and (ref_info.get('title') or ref_info.get('year')):
                    title = ref_info.get('title') or ""
                    year = ref_info.get('year') or ""
                    definition = f"{title} ({year})".strip() if title or year else "Reference not found"
                else:
                    definition = "Reference not found"

                if definition != "Reference not found":
                    location_data = {"page": ref["page"], "column": ref["column"], "bbox": ref["bbox"]}
                    pdf_transform.add_definition_to_margin(
                        doc=scaled_doc,
                        scaling_factor=scaling,
                        main_word=f"{ref['text']}",
                        definition=definition,
                        original_location=location_data,
                        original_content_bboxes=original_bboxes,
                    )
                    processed += 1

                _progress("Annotating references", i + 1, total_refs if total_refs else 1)

        if find_abbreviation:
            abbs = parser.find_abbreviations(
                original_doc,
                progress_callback=lambda d, t: _progress("Scanning for abbreviations", d, t)
            )
            _progress("Scanning for abbreviations", 1, 1)

            # Count occurrences
            initial_abbr_counts: dict = {}
            for abbr in abbs:
                abbr_text = abbr["text"]
                initial_abbr_counts[abbr_text] = initial_abbr_counts.get(abbr_text, 0) + 1

            # Only look up abbreviations that appear at least twice
            unique_abbreviations_for_lookup = {
                abbr_text for abbr_text, count in initial_abbr_counts.items() if count >= 2
            }

            # Resolve full forms
            full_form_map: dict = {}
            unique_list = list(unique_abbreviations_for_lookup)
            total_lookups = len(unique_list)
            for idx, abbr_text in enumerate(unique_list):
                full_form = definitions.find_full_form(
                    pdf_path=str(dest),
                    abbr=abbr_text,
                    groq_api_key=GROQ_API_KEY
                )
                if full_form:
                    full_form_map[abbr_text] = full_form
                _progress("Looking up full forms", idx + 1, total_lookups if total_lookups else 1)

            # Annotate
            abbr_occurrence_count: dict = {}
            abbr_added_pages: dict = {}
            total_abbs = len(abbs)

            for i, abbr in enumerate(abbs):
                abbr_text = abbr["text"]
                page = abbr.get("page")
                abbr_occurrence_count[abbr_text] = abbr_occurrence_count.get(abbr_text, 0) + 1

                if abbr_text not in abbr_added_pages:
                    abbr_added_pages[abbr_text] = set()

                # Skip the first occurrence entirely. Start adding on the second and subsequent occurrences.
                # Also ensuring we only add once per page for a given abbreviation.
                if (
                    abbr_text in full_form_map and
                    abbr_occurrence_count[abbr_text] >= 2 and
                    page not in abbr_added_pages[abbr_text]
                ):
                    definition = full_form_map[abbr_text].get("ans")

                    location_data = {
                        "page": page,
                        "column": abbr.get("column"),
                        "bbox": abbr.get("bbox")
                    }
                    using_llm = bool(full_form_map[abbr_text].get("using_llm"))
                    pdf_transform.add_definition_to_margin(
                        doc=scaled_doc,
                        scaling_factor=scaling,
                        main_word=abbr_text,
                        definition=definition,
                        original_location=location_data,
                        original_content_bboxes=original_bboxes,
                        using_llm=using_llm,
                    )
                    processed += 1
                    abbr_added_pages[abbr_text].add(page)

                _progress("Annotating abbreviations", i + 1, total_abbs if total_abbs else 1)

        _progress("Saving annotated PDF", 0, 1)
        timestamp = int(time.time())
        out_path = dest.parent / f"{dest.stem}_glossed_{timestamp}.pdf"
        scaled_doc.save(str(out_path))
        _progress("Saving annotated PDF", 1, 1)

        return [out_path, processed]

    except Exception as e:
        raise RuntimeError(f"Annotation failed: {e}") from e