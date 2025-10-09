from analyzer import extract_pdf_text, extract_abbreviations
from toughness import resolve_full_forms

pdf_path = "C:\\Users\\Lenovo\\Desktop\\GenAI\\Toughword_Analyser\\uploads\\TestFile.pdf"
pages = extract_pdf_text(pdf_path)
abbrevs = extract_abbreviations(pages)
abbr_map = resolve_full_forms(abbrevs, pages)

for abbr, full in abbr_map.items():
    print(f"{abbr} → {full}")
