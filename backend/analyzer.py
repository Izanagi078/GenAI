import fitz  # PyMuPDF
import re


def find_references(doc):
    """
    Find occurrences of references like "[n]" where n is any integer in a two-column research paper.
    
    Args:
        doc: A fitz.Document object representing the PDF document
        
    Returns:
        list: A list of dictionaries containing reference information:
            {
                'text': '[n]',          # The full reference text
                'number': n,            # The reference number as integer
                'page': page_number,    # Page number (0-based)
                'column': 1 or 2,       # Column number (1 for left, 2 for right)
                'block': block_number,  # Block number in the page
                'line': line_number,    # Line number within the block
                'bbox': (x0,y0,x1,y1)   # Bounding box coordinates
            }
    """
    pattern = r'\[(\d+)\]'
    refs = []
    
    for page_idx, page in enumerate(doc):
        # Get detailed text information including positions
        blocks = page.get_text("dict")["blocks"]
        page_width = page.rect.width
        
        # Process each text block
        for block_idx, block in enumerate(blocks):
            if "lines" not in block:
                continue
                
            # Determine which column the block belongs to
            block_center = (block["bbox"][0] + block["bbox"][2]) / 2
            column = 1 if block_center < page_width/2 else 2
            
            # Process each line in the block
            for line_idx, line in enumerate(block["lines"]):
                for span in line["spans"]:
                    text = span["text"]
                    matches = list(re.finditer(pattern, text))
                    
                    for match in matches:
                        refs.append({
                            'text': match.group(0),
                            'number': int(match.group(1)),
                            'page': page_idx,
                            'column': column,
                            'block': block_idx,
                            'line': line_idx,
                            'bbox': span["bbox"]  # (x0, y0, x1, y1)
                        })
    
    # Sort references by their position in the document
    sorted_refs = sorted(refs, key=lambda x: (x['page'], x['column'], x['block'], x['line']))
    
    # If there are no references, return empty list
    if not sorted_refs:
        return []
    
    # Find continuous sequence of references from the end
    # These are likely from the References section and should be removed
    last_idx = len(sorted_refs) - 1
    seq_start = last_idx
    
    # Work backwards to find where the sequence starts
    while seq_start > 0:
        if (sorted_refs[seq_start]['number'] != sorted_refs[seq_start-1]['number'] + 1 or
            sorted_refs[seq_start]['page'] != sorted_refs[seq_start-1]['page'] or
            abs(sorted_refs[seq_start]['block'] - sorted_refs[seq_start-1]['block']) > 1):
            break
        seq_start -= 1
    
    # Check if we found a sequence starting from 1 to some number
    if seq_start < last_idx and sorted_refs[seq_start]['number'] == 1:
        # Remove the sequence as it's likely the references section
        sorted_refs = sorted_refs[:seq_start]
    
    return sorted_refs


def find_paper_info(n, doc_path="uploads/twocolpaper.pdf"):
    """
    Find the title of a paper referenced by number n in the References section.
    
    Args:
        n: Reference number to search for
        doc_path: Path to the PDF document
        
    Returns:
        str: Title of the referenced paper, or None if not found
    """
    doc = fitz.open(doc_path)
    
    # First, find the References or Acknowledgements section
    references_page = None
    references_content = ""
    
    # Search through pages from back to front to find References section
    for page_num in range(doc.page_count - 1, -1, -1):
        page = doc[page_num]
        text = page.get_text()
        
        # Check for common section headers
        if "References" in text or "Bibliography" in text:
            references_page = page_num
            break
    
    if references_page is None:
        return None
        
    # Get text from the references section
    for page_num in range(references_page, doc.page_count):
        page = doc[page_num]
        references_content += page.get_text()
    
    # Look for the specific reference number
    # Common formats: [n], n., n), where n is the reference number
    patterns = [
        fr'\[{n}\]\s*(.*?)(?=\[\d+\]|\n\s*\[\d+\]|$)',  # [n] format
        fr'{n}\.\s*(.*?)(?=\d+\.|$)',                    # n. format
        fr'{n}\)\s*(.*?)(?=\d+\)|$)'                     # n) format
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, references_content, re.DOTALL)
        for match in matches:
            reference_text = match.group(1).strip()
            # Usually the title ends with a period and is followed by other information
            # We'll take the whole sentence as title for now
            # TODO: we need to make better logic to find title from whole reference sentence
            title_text = reference_text.strip()
            if title_text:
                return title_text
    
    return None


if __name__ == "__main__":
    # Test with a two-column paper
    doc = fitz.open("uploads/twocolpaper.pdf")
    
    refs = find_references(doc)
    
    for ref in refs:
        print(f"Reference {ref['text']} found on page {ref['page']+1}, column {ref['column']}, block {ref['block']}, line {ref['line']}")
    
    # Test paper info extraction
    if refs:
        # Test with the first reference found
        first_ref = refs[0]['number']
        print(f"\nTrying to find title for reference [{first_ref}]:")
        title = find_paper_info(first_ref)
        if title:
            print(f"Title: {title}")
        else:
            print("Title not found")
