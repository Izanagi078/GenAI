import pymupdf
import re
from typing import List, Optional, Dict, Tuple
from . import definitions

from PIL import Image
import io

try:
    from pix2tex.cli import LatexOCR
    latex_ocr_model = LatexOCR()
except Exception:
    latex_ocr_model = None


_NUMERIC_CITATION_PATTERN = re.compile(r'\[\s*(\d{1,3})\s*\]')
_NUMERIC_CITATION_CLUSTER_PATTERN = re.compile(r'\[\s*(\d{1,3}(?:\s*,\s*\d{1,3})+)\s*\]')
_AUTHOR_YEAR_ET_AL_PATTERN = re.compile(r'\b([A-Z][A-Za-z\'\-]+)\s+et al\.\s*\(\s*((?:19|20)\d{2}[a-z]?)\s*\)')
_AUTHOR_YEAR_ET_AL_BRACKET_PATTERN = re.compile(r'\b([A-Z][A-Za-z\'\-]+)\s+et al\.\s*\[\s*((?:19|20)\d{2}[a-z]?)\s*\]')
_AUTHOR_YEAR_BRACKET_GROUP_PATTERN = re.compile(r'\[([^\[\]]*(?:19|20)\d{2}[a-z]?[^\[\]]*)\]')
_AUTHOR_YEAR_PARENTHESES_GROUP_PATTERN = re.compile(r'\(([^)]*(?:19|20)\d{2}[a-z]?[^)]*)\)')
_AUTHOR_YEAR_PART_PATTERN = re.compile(
    r'([A-Z][A-Za-z\'\-]+(?:\s+et al\.)?(?:\s+(?:and|&)\s+[A-Z][A-Za-z\'\-]+)*)\s*,?\s*((?:19|20)\d{2}[a-z]?)'
)
_YEAR_PATTERN = re.compile(r'\b((?:19|20)\d{2})[a-z]?\b')
_REFERENCE_ENTRY_START_PATTERN = re.compile(r"[A-Z][A-Za-z'\-]+,\s+[A-Z]")
_AUTHOR_PARTICLES = {"de", "del", "der", "van", "von", "da", "di", "la", "le"}

def _canonical_year(year_text: str) -> Optional[str]:
    match = _YEAR_PATTERN.search(year_text.lower())
    return match.group(1) if match else None

def _extract_author_key_from_segment(segment: str) -> Optional[str]:
    text = segment.strip()
    if not text: return None
    text = re.sub(r'^\W+', '', text)
    text = re.sub(r'^(?:see|e\.g\.|cf\.|for example|for instance)\s+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bet\s+al\.?$', '', text, flags=re.IGNORECASE).strip()
    text = re.split(r'\s+(?:and|&)\s+', text, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    tokens = re.findall(r"[A-Za-z][A-Za-z'\-]+", text)
    if not tokens: return None
    for token in reversed(tokens):
        lowered = token.lower()
        if lowered not in _AUTHOR_PARTICLES and len(lowered) > 1:
            return re.sub(r"[^a-z]", "", lowered)
    return re.sub(r"[^a-z]", "", tokens[-1].lower())

def _extract_author_year_pairs_from_parenthetical(inner_text: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for part in re.split(r';', inner_text):
        part = part.strip()
        if not part: continue
        for match in _AUTHOR_YEAR_PART_PATTERN.finditer(part):
            author_key = _extract_author_key_from_segment(match.group(1))
            year = _canonical_year(match.group(2))
            if author_key and year:
                pairs.append((author_key, year))
    return pairs

def _find_references_start_page(doc: pymupdf.Document) -> Optional[int]:
    for page_idx in range(len(doc) - 1, -1, -1):
        page_text = doc[page_idx].get_text()
        match = re.search(r'\b(references|bibliography)\b', page_text, re.IGNORECASE)
        if match:
            return page_idx
    return None

def _extract_author_year_from_entry(entry_text: str) -> Optional[Tuple[str, str]]:
    text = re.sub(r"\s+", " ", entry_text).strip()
    if not text: return None
    text = re.sub(r'^\s*(?:\[\d+\]|\d+[\.]|\d+[\)])\s*', '', text)
    year_match = _YEAR_PATTERN.search(text)
    if not year_match: return None
    year = _canonical_year(year_match.group(0))
    prefix = text[:year_match.start()].strip()
    author_match = re.match(r"([A-Z][A-Za-z'\-]+)\s*,", prefix)
    key = re.sub(r"[^a-z]", "", author_match.group(1).lower()) if author_match else _extract_author_key_from_segment(prefix)
    return (key, year) if key and year else None

def _extract_author_year_entries(doc: pymupdf.Document, references_page: int) -> List[str]:
    entries = []
    
    for page_num in range(references_page, doc.page_count):
        blocks = doc[page_num].get_text("blocks")
        for b in blocks:
            if b[6] != 0: continue
            text = b[4].strip()
            if not text or text.lower() in ["references", "bibliography"]: continue
            
            if re.search(r"(?:^|\n)\[\d+\]", text):
                parts = re.split(r"(?:^|\n)(?=\[\d+\])", text)
                for p in parts:
                    p = p.strip().replace('\n', ' ')
                    if p: entries.append(p)
            elif re.search(r"(?:^|\n)\[[A-Z]", text):
                parts = re.split(r"(?:^|\n)(?=\[[A-Z])", text)
                for p in parts:
                    p = p.strip().replace('\n', ' ')
                    if p: entries.append(p)
            else:
                entries.append(text.replace('\n', ' '))
                
    return entries

def build_references_db(doc: pymupdf.Document, groq_api_key: Optional[str] = None, use_local_llm: bool = False, progress_callback: Optional[callable] = None) -> Dict[str, Dict]:
    db = {"numeric": {}, "author_year": {}}
    ref_start_page = _find_references_start_page(doc)
    if ref_start_page is None: return db

    full_ref_text = ""
    ref_blocks = []
    for p in range(ref_start_page, doc.page_count):
        full_ref_text += doc[p].get_text() + "\n"
        for b in doc[p].get_text("blocks"):
            if b[6] == 0:
                text = b[4].strip()
                if text and text.lower() not in ["references", "bibliography"]:
                    ref_blocks.append(text.replace("\n", " "))

    all_entries = _extract_author_year_entries(doc, ref_start_page)
    to_process_refs = []

    for i, entry in enumerate(all_entries):
        num_match = re.match(r"^\[(\d+)\]", entry)
        if num_match:
            idx = int(num_match.group(1))
            to_process_refs.append({"id": f"num_{idx}", "text": entry})
        else:
            ay = _extract_author_year_from_entry(entry)
            if ay:
                to_process_refs.append({"id": f"ay_{ay[0]}_{ay[1]}", "text": entry, "target_author": ay[0], "target_year": ay[1]})

    citation_refs = find_references(doc)
    existing_ay_ids = {r["id"] for r in to_process_refs if r["id"].startswith("ay_")}

    for ref in citation_refs:
        if ref.get("format_type") == "NUMERIC_BRACKET": continue

        auth = ref.get("author_key")
        yr = ref.get("year")
        if not auth or not yr: continue

        key = f"ay_{auth}_{yr}"
        if key not in existing_ay_ids:
            snippet = None
            search_regex = rf"{auth}.*?{yr}"

            for block_text in ref_blocks:
                if re.search(search_regex, block_text, re.IGNORECASE | re.DOTALL):
                    snippet = block_text
                    break

            if not snippet:
                match = re.search(search_regex, full_ref_text, re.IGNORECASE | re.DOTALL)
                if match:
                    start_idx = max(0, match.start() - 300)
                    end_idx = min(len(full_ref_text), match.end() + 300)
                    snippet = full_ref_text[start_idx:end_idx].replace('\n', ' ')

            if snippet:
                to_process_refs.append({"id": key, "text": snippet, "target_author": auth, "target_year": yr})
                existing_ay_ids.add(key)
            else:
                db["author_year"][f"{auth}_{yr}"] = {"title": None, "year": yr}

    if to_process_refs:
        results = {}
        for i, ref in enumerate(to_process_refs):
            res = definitions.extract_title_year_from_reference(ref["text"], groq_api_key, target_author=ref.get("target_author"), target_year=ref.get("target_year"), use_local_llm=use_local_llm)
            if res:
                results[ref["id"]] = res
            if progress_callback:
                progress_callback(i + 1, len(to_process_refs))

        for ref in to_process_refs:
            ref_id = ref["id"]
            if ref_id in results:
                if ref_id.startswith("num_"):
                    idx = int(ref_id[4:])
                    db["numeric"][idx] = results[ref_id]
                elif ref_id.startswith("ay_"):
                    key = ref_id[3:]
                    db["author_year"][key] = results[ref_id]

    return db


def find_references(doc: pymupdf.Document, progress_callback: Optional[callable] = None) -> List[dict]:
    """
        Find in-text citations in a two-column research paper.
        Supports:
            1) Numeric bracket style: [n]
            2) Author-year parenthetical: (Author, 2020)
            3) Author-year narrative: Author et al. (2020)
    """
    refs = []
    num_pages = len(doc)

    for page_idx, page in enumerate(doc):
        if progress_callback:
            progress_callback(page_idx, num_pages)
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

                # find numeric bracket citations: [n]
                for cluster_match in _NUMERIC_CITATION_CLUSTER_PATTERN.finditer(full_line_text):
                    mstart = cluster_match.start()
                    use_span = None
                    for (s_start, s_end, s_obj) in cum:
                        if s_start <= mstart < s_end:
                            use_span = s_obj
                            break
                    if use_span is None and cum:
                        use_span = cum[0][2]
                    for num_str in re.split(r'\s*,\s*', cluster_match.group(1)):
                        refs.append({
                            "text": f"[{num_str.strip()}]",
                            "number": int(num_str.strip()),
                            "format_type": "NUMERIC_BRACKET",
                            "page": page_idx,
                            "column": column,
                            "block": block_idx,
                            "line": line_idx,
                            "bbox": use_span.get("bbox"),
                        })

                for match in _NUMERIC_CITATION_PATTERN.finditer(full_line_text):
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
                        "format_type": "NUMERIC_BRACKET",
                        "page": page_idx,
                        "column": column,
                        "block": block_idx,
                        "line": line_idx,
                        "bbox": use_span.get("bbox"),
                    })

                for match in _AUTHOR_YEAR_PARENTHESES_GROUP_PATTERN.finditer(full_line_text):
                    mstart = match.start()
                    use_span = None
                    for (s_start, s_end, s_obj) in cum:
                        if s_start <= mstart < s_end:
                            use_span = s_obj
                            break
                    if use_span is None and cum:
                        use_span = cum[0][2]

                    pairs = _extract_author_year_pairs_from_parenthetical(match.group(1))
                    for author_key, year in pairs:
                        refs.append({
                            "text": match.group(0),
                            "author_key": author_key,
                            "year": year,
                            "format_type": "AUTHOR_YEAR_PARENTHESES",
                            "page": page_idx,
                            "column": column,
                            "block": block_idx,
                            "line": line_idx,
                            "bbox": use_span.get("bbox"),
                        })

                # find author-year narrative citations: Author et al. (2020)
                for match in _AUTHOR_YEAR_ET_AL_PATTERN.finditer(full_line_text):
                    mstart = match.start()
                    use_span = None
                    for (s_start, s_end, s_obj) in cum:
                        if s_start <= mstart < s_end:
                            use_span = s_obj
                            break
                    if use_span is None and cum:
                        use_span = cum[0][2]

                    author_key = _extract_author_key_from_segment(match.group(1))
                    year = _canonical_year(match.group(2))
                    if author_key and year:
                        refs.append({
                            "text": match.group(0),
                            "author_key": author_key,
                            "year": year,
                            "format_type": "AUTHOR_YEAR_ET_AL",
                            "page": page_idx,
                            "column": column,
                            "block": block_idx,
                            "line": line_idx,
                            "bbox": use_span.get("bbox"),
                        })

                # find author-year square-bracket group citations: [Author et al., 2020] or [A, 2019; B, 2020]
                for match in _AUTHOR_YEAR_BRACKET_GROUP_PATTERN.finditer(full_line_text):
                    mstart = match.start()
                    use_span = None
                    for (s_start, s_end, s_obj) in cum:
                        if s_start <= mstart < s_end:
                            use_span = s_obj
                            break
                    if use_span is None and cum:
                        use_span = cum[0][2]

                    pairs = _extract_author_year_pairs_from_parenthetical(match.group(1))
                    for author_key, year in pairs:
                        refs.append({
                            "text": match.group(0),
                            "author_key": author_key,
                            "year": year,
                            "format_type": "AUTHOR_YEAR_BRACKET",
                            "page": page_idx,
                            "column": column,
                            "block": block_idx,
                            "line": line_idx,
                            "bbox": use_span.get("bbox"),
                        })

                # find author-year narrative citations with square brackets: Author et al. [2020]
                for match in _AUTHOR_YEAR_ET_AL_BRACKET_PATTERN.finditer(full_line_text):
                    mstart = match.start()
                    use_span = None
                    for (s_start, s_end, s_obj) in cum:
                        if s_start <= mstart < s_end:
                            use_span = s_obj
                            break
                    if use_span is None and cum:
                        use_span = cum[0][2]

                    author_key = _extract_author_key_from_segment(match.group(1))
                    year = _canonical_year(match.group(2))
                    if author_key and year:
                        refs.append({
                            "text": match.group(0),
                            "author_key": author_key,
                            "year": year,
                            "format_type": "AUTHOR_YEAR_ET_AL_BRACKET",
                            "page": page_idx,
                            "column": column,
                            "block": block_idx,
                            "line": line_idx,
                            "bbox": use_span.get("bbox"),
                        })

    sorted_refs = sorted(refs, key=lambda x: (x["page"], x["column"], x["block"], x["line"]))

    if not sorted_refs:
        return []

    # Remove citations that appear in the References section and after it.
    references_page = _find_references_start_page(doc)

    if references_page is not None:
        sorted_refs = [ref for ref in sorted_refs if ref["page"] < references_page]

    return sorted_refs


def find_abbreviations(doc: pymupdf.Document, progress_callback: Optional[callable] = None) -> List[dict]:
    """
    Finds all-uppercase abbreviations of at least 3 characters in the document.

    Args:
        doc: The PyMuPDF document object.
        progress_callback: Optional callback for progress reporting.

    Returns:
        A list of dictionaries, where each dictionary represents a found
        abbreviation and contains its text and location details.
    """
    # Regex to find whole words consisting of 3 to 5 uppercase letters.
    # \b is a word boundary to ensure we don't match parts of other words.
    pattern = r'\b[A-Z]{3,5}\b'
    abbs = []
    num_pages = len(doc)
    ref_start_page = _find_references_start_page(doc)

    for page_idx, page in enumerate(doc):
        if progress_callback:
            progress_callback(page_idx, num_pages)

        if ref_start_page is not None and page_idx >= ref_start_page:
            continue

        blocks = page.get_text("dict")["blocks"]
        page_width = page.rect.width
        page_height = page.rect.height

        for block_idx, block in enumerate(blocks):
            if "lines" not in block:
                continue

            block_y0 = block["bbox"][1]
            block_y1 = block["bbox"][3]
            if block_y0 < 30 or block_y1 > page_height - 30:
                continue

            block_center = (block["bbox"][0] + block["bbox"][2]) / 2
            column = 1 if block_center < page_width / 2 else 2
            block_text = "".join(span["text"] for line in block["lines"] for span in line["spans"])

            for line_idx, line in enumerate(block["lines"]):
                for span in line["spans"]:
                    text = span["text"]
                    matches = list(re.finditer(pattern, text))

                    for match in matches:
                            abbs.append({
                                "text": match.group(0),
                                "page": page_idx,
                                "column": column,
                                "block": block_idx,
                                "line": line_idx,
                                "bbox": span["bbox"],
                                "context": block_text,
                            })

    return abbs

def find_symbols(doc: pymupdf.Document, progress_callback: Optional[callable] = None) -> List[dict]:
    """
    Finds mathematical symbols in the document using Unicode ranges and LatexOCR.
    """
    symbols = []
    num_pages = len(doc)

    unicode_pattern = re.compile(r'[\u0370-\u03FF\u2200-\u22FF\u2A00-\u2AFF\u2070-\u209F]+')
    latex_symbol_pattern = re.compile(r'\\[a-zA-Z]+|[a-zA-Z](?:_[a-zA-Z0-9]+|\^[a-zA-Z0-9]+)')

    _LATEX_NON_SYMBOLS = {
        '\\text', '\\begin', '\\end', '\\frac', '\\left', '\\right',
        '\\mathbf', '\\mathrm', '\\mathcal', '\\mathit', '\\mathtt',
        '\\quad', '\\qquad',
        '\\tiny', '\\scriptsize', '\\footnotesize', '\\small', '\\normalsize',
        '\\large', '\\Large', '\\LARGE', '\\huge', '\\Huge',
        '\\displaystyle', '\\textstyle', '\\scriptstyle', '\\scriptscriptstyle',
        '\\bf', '\\rm', '\\it', '\\sf', '\\tt', '\\boldmath', '\\cal',
        '\\mathbb', '\\mathsf', '\\mathfrak',
        '\\bigg', '\\Bigg', '\\Big', '\\big',
        '\\bigl', '\\bigr', '\\Bigl', '\\Bigr', '\\biggl', '\\biggr',
        '\\operatorname', '\\mbox', '\\hbox',
        '\\underbrace', '\\overbrace', '\\stackrel', '\\underset', '\\overset',
        '\\bar', '\\hat', '\\tilde', '\\vec', '\\dot', '\\ddot',
        '\\overline', '\\underline', '\\widehat', '\\widetilde',
        '\\strut', '\\phantom',
    }

    _VALID_UPPERCASE_LATEX = {
        '\\Gamma', '\\Delta', '\\Theta', '\\Lambda', '\\Xi', '\\Pi',
        '\\Sigma', '\\Upsilon', '\\Phi', '\\Psi', '\\Omega',
        '\\Alpha', '\\Beta', '\\Epsilon', '\\Zeta', '\\Eta',
        '\\Iota', '\\Kappa', '\\Mu', '\\Nu', '\\Rho', '\\Tau', '\\Chi',
    }

    _COMMON_SYMBOLS = {
        '=', '+', '-', '*', '/', '%', '^', '&', '|', '~', '!', '>', '<', '≥', '≤', '≈', '≠', '±', '×', '÷',
        '(', ')', '[', ']', '{', '}', ',', '.', ';', ':', '?', '!', '°',
        '∞', '∝', '∂', '∑', '√', '∝', '∞',
        '∘', '∙', '∧', '∨', '∩', '∪', '∫', '∴', '∵', '∼', '≡', '≪', '≫', '⊖', '⊗', '⊘', '⊙', '⊥', '⊢', '⊣', '⊤',
        '¬', '∏', '∑', '−', '∕', '∗', '∙', '√', '∝', '∞',
        '∠', '∨', '∪', '∫', '∬', '∭', '∮', '∯', '∰', '∱', '∲', '∳',
        # LaTeX versions
        '\\ge', '\\le', '\\neq', '\\approx', '\\pm', '\\mp', '\\times', '\\div',
        '\\infty', '\\propto', '\\partial', '\\nabla', '\\in', '\\notin', '\\ni',
        '\\prod', '\\sum', '\\sqrt', '\\int', '\\oint', '\\forall', '\\exists',
        '\\emptyset', '\\Delta', '\\nabla', '\\to', '\\leftarrow', '\\rightarrow',
        '\\leftrightarrow', '\\uparrow', '\\downarrow', '\\langle', '\\rangle',
        '\\cdot', '\\cdots', '\\vdots', '\\ddots', '\\quad', '\\qquad', '\\text',
    }
    import logging
    logging.getLogger('pix2tex').setLevel(logging.ERROR)
    logging.getLogger('PIL').setLevel(logging.ERROR)

    for page_idx, page in enumerate(doc):
        if progress_callback:
            progress_callback(page_idx, num_pages)
        
        blocks = page.get_text("dict")["blocks"]
        page_width = page.rect.width
        
        for block_idx, block in enumerate(blocks):
            if "lines" not in block:
                continue
            
            block_center = (block["bbox"][0] + block["bbox"][2]) / 2
            column = 1 if block_center < page_width / 2 else 2
            
            block_text = "".join(span["text"] for line in block["lines"] for span in line["spans"])
            
            def _get_symbol_context(text: str, index: int, word_margin: int = 100) -> str:
                if index < 0:
                    return text[:1200]
                
                before = text[:index]
                after = text[index:]
                
                words_before = before.split()
                words_after = after.split()
                
                selected_before = " ".join(words_before[-word_margin:])
                selected_after = " ".join(words_after[:word_margin])
                
                return (selected_before + " " + selected_after).strip()

            words = [w for w in block_text.split() if w.isalpha()]
            is_equation = False
            
            # Avoid likely bibliography, author blocks, or pure citation/list blocks
            has_year = bool(re.search(r'\b(?:19|20)\d{2}\b', block_text))
            is_author_list = (
                re.search(r'\bet\s+al\.?', block_text, re.I)
                or block_text.count(",") > 4
                or has_year
            )
            
            # Pure citation/list blocks like "[1] Text..." or "1. Text..."
            is_list_item = bool(re.match(r'^\s*(?:\[\d+\]|\d+[\.\)])\s+', block_text))

            if not is_author_list and not is_list_item and len(words) < 12:
                # Require unambiguous math signals: = or sub/superscripts
                has_math_op = any(c in block_text for c in "=<>/±∑∏√")
                has_sub_super = any(c in block_text for c in "_^")
                has_brackets = any(c in block_text for c in "[]{}|") 
                
                # Equation must have at least two math-like signals or be very sparse
                if (has_math_op and (has_sub_super or has_brackets)) or block_text.count('=') >= 1:
                    is_equation = True
                elif len(words) < 5 and any(c.isdigit() for c in block_text) and (has_math_op or has_sub_super):
                    is_equation = True
            
            if is_equation and latex_ocr_model is not None:
                rect = pymupdf.Rect(block["bbox"])
                if rect.width > 10 and rect.height > 10:
                    pix = page.get_pixmap(clip=rect, dpi=200)
                    img_bytes = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    
                    try:
                        latex_text = latex_ocr_model(img)
                        found_latex_symbols = latex_symbol_pattern.findall(latex_text)
                        for sys_match in set(found_latex_symbols):
                            if len(sys_match) == 1 and sys_match.isalpha():
                                continue
                            if sys_match in _LATEX_NON_SYMBOLS or sys_match in _COMMON_SYMBOLS:
                                continue
                            
                            cmd_name = sys_match[1:] if sys_match.startswith('\\') else sys_match
                            if len(cmd_name) > 15:
                                continue
                            if any(c in sys_match for c in ')]}>'):
                                continue
                            if sys_match.startswith('\\') and cmd_name[:1].isupper() and sys_match not in _VALID_UPPERCASE_LATEX:
                                continue
                            
                            symbols.append({
                                "text": sys_match,
                                "page": page_idx,
                                "column": column,
                                "block": block_idx,
                                "bbox": block["bbox"],
                                "context": _get_symbol_context(block_text, block_text.find(sys_match)),
                                "source": "ocr"
                            })
                    except Exception as e:
                        pass

            current_offset = 0
            for line_idx, line in enumerate(block["lines"]):
                for span in line["spans"]:
                    text = span["text"]
                    matches = list(unicode_pattern.finditer(text))
                    
                    if matches:
                        for match in matches:
                            sym_text = match.group(0).strip()
                            if sym_text and sym_text not in _COMMON_SYMBOLS:
                                symbols.append({
                                    "text": sym_text,
                                    "page": page_idx,
                                    "column": column,
                                    "block": block_idx,
                                    "line": line_idx,
                                    "bbox": span["bbox"],
                                    "context": _get_symbol_context(block_text, current_offset + match.start()),
                                    "source": "unicode"
                                })
                    current_offset += len(text)

    return symbols