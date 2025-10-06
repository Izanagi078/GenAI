import fitz
import re

def extract_pdf_text(path):
    doc = fitz.open(path)
    return [page.get_text() for page in doc]

def extract_abbreviations(pages):
    pattern = r'\b[A-Z]{2,}(?:-[A-Z]+)?\b'  # Matches AI, CNN, NLP, or ISO-IEC
    abbrevs = set()

    for page in pages:
        # Optional: remove headers/footers if needed
        matches = re.findall(pattern, page)
        abbrevs.update(matches)

    return sorted(abbrevs)
