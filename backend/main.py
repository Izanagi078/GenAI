import fitz
from analyzer import find_references, find_paper_info

if __name__ == "__main__":
    pdf_path = "uploads/twocolpaper.pdf"
    doc = fitz.open(pdf_path)
    
    # Find all references
    refs = find_references(doc)
    
    print(f"\nFound {len(refs)} references in {pdf_path}:\n")
    for ref in refs:
        print(f"\nReference {ref['text']:>5} | Page {ref['page']+1} | Column {ref['column']} | Block {ref['block']} | Line {ref['line']}")
        print(f"Title: {find_paper_info(ref['number'])}\n")
    
    doc.close()