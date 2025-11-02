import fitz  # PyMuPDF
import re
from typing import List, Optional


def find_references(doc) -> List[dict]:
    """
    Find occurrences of references like "[n]" where n is any integer in a two-column research paper.
    Returns a list of reference dicts (same shape as original analyzer.py).
    """
    pattern = r'\[(\d+)\]'
    refs = []

    for page_idx, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        page_width = page.rect.width

        for block_idx, block in enumerate(blocks):
            if "lines" not in block:
                continue

            block_center = (block["bbox"][0] + block["bbox"][2]) / 2
            column = 1 if block_center < page_width / 2 else 2

            for line_idx, line in enumerate(block["lines"]):
                for span in line["spans"]:
                    text = span["text"]
                    matches = list(re.finditer(pattern, text))

                    for match in matches:
                        refs.append({
                            "text": match.group(0),
                            "number": int(match.group(1)),
                            "page": page_idx,
                            "column": column,
                            "block": block_idx,
                            "line": line_idx,
                            "bbox": span["bbox"],
                        })

    sorted_refs = sorted(refs, key=lambda x: (x["page"], x["column"], x["block"], x["line"]))

    if not sorted_refs:
        return []

    # Remove trailing References section sequence if present
    last_idx = len(sorted_refs) - 1
    seq_start = last_idx

    while seq_start > 0:
        if (
            sorted_refs[seq_start]["number"]
            != sorted_refs[seq_start - 1]["number"] + 1
            or sorted_refs[seq_start]["page"] != sorted_refs[seq_start - 1]["page"]
            or abs(sorted_refs[seq_start]["block"] - sorted_refs[seq_start - 1]["block"]) > 1
        ):
            break
        seq_start -= 1

    if seq_start < last_idx and sorted_refs[seq_start]["number"] == 1:
        sorted_refs = sorted_refs[:seq_start]

    return sorted_refs


def find_paper_info(n: int, doc_path: str = "uploads/twocolpaper.pdf") -> Optional[str]:
    """
    Find the title of a paper referenced by number n in the References section of the PDF at doc_path.
    """
    doc = fitz.open(doc_path)

    references_page = None
    references_content = ""

    for page_num in range(doc.page_count - 1, -1, -1):
        page = doc[page_num]
        text = page.get_text()

        if "References" in text or "Bibliography" in text:
            references_page = page_num
            break

    if references_page is None:
        return None

    for page_num in range(references_page, doc.page_count):
        page = doc[page_num]
        references_content += page.get_text()

    patterns = [
        fr'\[{n}\]\s*(.*?)(?=\[\d+\]|\n\s*\[\d+\]|$)',
        fr'{n}\.\s*(.*?)(?=\d+\.|$)',
        fr'{n}\)\s*(.*?)(?=\d+\)|$)'
    ]

    for pattern in patterns:
        matches = re.finditer(pattern, references_content, re.DOTALL)
        for match in matches:
            reference_text = match.group(1).strip()
            if reference_text:
                return reference_text

    return None
