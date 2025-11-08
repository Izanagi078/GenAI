import fitz  # PyMuPDF
import re
from typing import List, Optional, Dict
from . import definitions


def find_references(doc: fitz.Document) -> List[dict]:
    """
    Find occurrences of references like "[n]" where n is any integer in a two-column research paper.
    Returns a list of reference dicts (same shape as original analyzer.py).
    """
    pattern = r'\[\s*(\d+)\s*\]'
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
                # Build full line text by concatenating spans (keeps same reading order)
                spans = line.get("spans", [])
                if not spans:
                    continue
                span_texts = [s.get("text", "") for s in spans]
                full_line_text = "".join(span_texts)

                # build cumulative char ranges for each span to map match positions -> span
                cum = []
                pos = 0
                for s in spans:
                    t = s.get("text", "")
                    start = pos
                    end = pos + len(t)
                    cum.append((start, end, s))
                    pos = end

                # find matches on the concatenated line text
                for match in re.finditer(pattern, full_line_text):
                    mstart = match.start()
                    # find the span that contains the start of the match (fallback to first span)
                    use_span = None
                    for (s_start, s_end, s_obj) in cum:
                        if s_start <= mstart < s_end:
                            use_span = s_obj
                            break
                    if use_span is None and cum:
                        use_span = cum[0][2]

                    refs.append({
                        "text": match.group(0),
                        "number": int(match.group(1)),
                        "page": page_idx,
                        "column": column,
                        "block": block_idx,
                        "line": line_idx,
                        "bbox": use_span.get("bbox"),
                    })

    sorted_refs = sorted(refs, key=lambda x: (x["page"], x["column"], x["block"], x["line"]))

    if not sorted_refs:
        return []

    # Remove references that appear in the References section (from the page containing "References" onwards)
    references_page = None
    for page_idx in range(len(doc) - 1, -1, -1):
        page_text = doc[page_idx].get_text()
        if "References" in page_text or "Bibliography" in page_text:
            references_page = page_idx
            break

    if references_page is not None:
        sorted_refs = [ref for ref in sorted_refs if ref["page"] < references_page]

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

def find_abbreviations(doc: fitz.Document) -> List[dict]:
    """
    Finds all-uppercase abbreviations of at least 3 characters in the document.

    Args:
        doc: The PyMuPDF document object.

    Returns:
        A list of dictionaries, where each dictionary represents a found
        abbreviation and contains its text and location details.
    """
    # Regex to find whole words consisting of 3 to 5 uppercase letters.
    # \b is a word boundary to ensure we don't match parts of other words.
    pattern = r'\b[A-Z]{3,5}\b'
    abbs = []

    for page_idx, page in enumerate(doc):
        # Extract text blocks with detailed structural information
        blocks = page.get_text("dict")["blocks"]
        page_width = page.rect.width

        for block_idx, block in enumerate(blocks):
            if "lines" not in block:
                continue

            # Determine if the block is in the left (1) or right (2) column
            block_center = (block["bbox"][0] + block["bbox"][2]) / 2
            column = 1 if block_center < page_width / 2 else 2

            for line_idx, line in enumerate(block["lines"]):
                for span in line["spans"]:
                    text = span["text"]
                    # Find all occurrences of the abbreviation pattern in the span's text
                    matches = list(re.finditer(pattern, text))

                    # For each match, create a dictionary with its details
                    for match in matches:
                        abbs.append({
                            "text": match.group(0),
                            "page": page_idx,
                            "column": column,
                            "block": block_idx,
                            "line": line_idx,
                            "bbox": span["bbox"],
                        })

    return abbs


def build_references_db(doc: fitz.Document, google_api_key: Optional[str] = None) -> Dict[int, Dict[str, Optional[str]]]:
    """
    Build a database of references with their titles and years extracted from the references section.
    Scans the last two pages to find the references section, extracts individual reference texts,
    and uses LLM to parse title and year for each.
    """
    db = {}

    # Get the last two pages
    num_pages = len(doc)
    pages_to_check = doc[-2:] if num_pages >= 2 else doc

    references_content = ""
    found_references = False

    for page in pages_to_check:
        text = page.get_text()
        if not found_references:
            if "References" in text or "Bibliography" in text:
                found_references = True
                # Find the position of "References" and take text from there
                ref_pos = text.find("References") if "References" in text else text.find("Bibliography")
                if ref_pos != -1:
                    references_content += text[ref_pos:]
                else:
                    references_content += text
            # If not found, continue to next page
        else:
            references_content += text

    if not references_content:
        print("No references section found in the last two pages.")
        return db

    # Regex to find individual references: [1] text until next [2] or end
    pattern = r'\[(\d+)\]\s*(.*?)(?=\[\d+\]|$)'
    matches = re.findall(pattern, references_content, re.DOTALL)

    for number_str, ref_text in matches:
        number = int(number_str)
        ref_text = ref_text.strip()
        if ref_text:
            print(f"Processing reference [{number}]: {ref_text[:100]}...")
            extracted = definitions.extract_title_year_from_reference(ref_text, google_api_key)
            if extracted:
                db[number] = extracted
                print(f"Extracted for [{number}]: Title='{extracted.get('title')}', Year='{extracted.get('year')}'")
            else:
                db[number] = {"title": None, "year": None}
                print(f"Failed to extract for [{number}]")

    return db
