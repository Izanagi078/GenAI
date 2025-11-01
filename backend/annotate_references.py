import fitz  # PyMuPDF
import re

def widen_pages(doc, extra_margin=200):
    widened_doc = fitz.open()
    for page in doc:
        original_rect = page.rect
        new_width = original_rect.width + extra_margin
        new_rect = fitz.Rect(0, 0, new_width, original_rect.height)

        # Create new page with expanded width
        new_page = widened_doc.new_page(width=new_width, height=original_rect.height)

        # Shift original content right by extra_margin
        matrix = fitz.Matrix(1, 1).pretranslate(extra_margin, 0)
        new_page.show_pdf_page(new_rect, doc, page.number, matrix=matrix)

    return widened_doc

def find_references(doc):
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
                            'text': match.group(0),
                            'number': int(match.group(1)),
                            'page': page_idx,
                            'column': column,
                            'block': block_idx,
                            'line': line_idx,
                            'bbox': span["bbox"]
                        })
    sorted_refs = sorted(refs, key=lambda x: (x['page'], x['column'], x['block'], x['line']))
    return sorted_refs

def find_paper_info(n, doc):
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
        references_content += doc[page_num].get_text()
    patterns = [
        fr'\[{n}\]\s*(.*?)(?=\[\d+\]\s|\n|$)',  # [n] format
        fr'{n}\.\s*(.*?)(?=\d+\.|$)',           # n. format
        fr'{n}\)\s*(.*?)(?=\d+\)|$)'            # n) format
    ]
    for pattern in patterns:
        matches = re.finditer(pattern, references_content, re.DOTALL)
        for match in matches:
            title_text = match.group(1).strip()
            if title_text:
                return title_text
    return None

def annotate_references_with_meanings(doc, refs, original_doc, output_path="annotated_output.pdf"):
    margin_width = 180
    font_size = 8
    padding = 10
    box_height = 60

    for ref in refs:
        page = doc[ref['page']]
        ref_num = ref['number']
        meaning = find_paper_info(ref_num, original_doc)

        if not meaning:
            print(f"⚠️ Reference [{ref_num}] not found in References section.")
            meaning = "Reference meaning not found."

        # Adjust margin placement
        if ref['column'] == 1:
            x_margin = padding
        else:
            x_margin = page.rect.width - margin_width - padding

        y_position = ref['bbox'][1]
        rect = fitz.Rect(x_margin, y_position, x_margin + margin_width, y_position + box_height)

        # Insert annotation
        page.insert_textbox(
            rect,
            f"[{ref_num}] {meaning}",
            fontsize=font_size,
            fontname="helv",
            color=(0, 0, 0),
            align=fitz.TEXT_ALIGN_LEFT
        )

    doc.save(output_path)
    print(f"✅ Annotated PDF saved as: {output_path}")

if __name__ == "__main__":
    input_pdf = "uploads/twocolpaper.pdf"
    original_doc = fitz.open(input_pdf)

    # Step 1: Widen pages
    widened_doc = widen_pages(original_doc, extra_margin=200)

    # Step 2: Find references from widened layout
    refs = find_references(widened_doc)

    # Step 3: Annotate widened document using original doc for meanings
    annotate_references_with_meanings(widened_doc, refs, original_doc)
