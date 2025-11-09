import fitz


def get_page_content_bbox(page: fitz.Page, padding=0) -> fitz.Rect:
    blocks = page.get_text("blocks")
    if not blocks:
        return page.cropbox

    x0, y0, x1, y1 = blocks[0][0:4]
    for b in blocks[1:]:
        x0 = min(x0, b[0])
        y0 = min(y0, b[1])
        x1 = max(x1, b[2])
        y1 = max(y1, b[3])

    return fitz.Rect(x0 - padding, y0, x1 + padding, y1)


def scale_content_horizontally(doc: fitz.Document, scaling_factor: float) -> tuple[fitz.Document, list]:
    new_doc = fitz.open()
    original_content_bboxes = []

    for source_page in doc:
        content_bbox = get_page_content_bbox(source_page)
        original_content_bboxes.append(content_bbox)

        new_page = new_doc.new_page(width=source_page.rect.width, height=source_page.rect.height)
        new_page.show_pdf_page(new_page.rect, doc, source_page.number)

        sx = scaling_factor
        content_center_x = content_bbox.x0 + content_bbox.width / 2
        tx = content_center_x * (1 - sx)

        transform_command = f"{sx} 0 0 1.0 {tx} 0 cm\n".encode("utf-8")

        original_content = new_page.read_contents()
        new_content = transform_command + original_content
        content_xref = new_page.get_contents()[0]
        new_doc.update_stream(content_xref, new_content)

    return new_doc, original_content_bboxes


def is_margin_space_occupied(page: fitz.Page, new_text_bbox: fitz.Rect, margin_area: fitz.Rect) -> bool:
    existing_blocks = page.get_text("blocks")

    for block in existing_blocks:
        existing_bbox = fitz.Rect(block[:4])
        if margin_area.contains(existing_bbox) and new_text_bbox.intersects(existing_bbox):
            return True

    return False


def add_definition_to_margin(doc: fitz.Document, scaling_factor: float, main_word: str, definition: str, original_location: dict, original_content_bboxes: list,  using_llm: bool = False,):
    try:
        page_num = original_location["page"]
        page = doc[page_num]

        original_bbox = fitz.Rect(original_location["bbox"])
        content_bbox = original_content_bboxes[page_num]
        content_center_x = content_bbox.x0 + content_bbox.width / 2

        scaled_content_x0 = content_center_x + (content_bbox.x0 - content_center_x) * scaling_factor
        scaled_content_x1 = content_center_x + (content_bbox.x1 - content_center_x) * scaling_factor

        y_position = original_bbox.y0
        padding = 5

        if original_location["column"] == 1:
            target_rect = fitz.Rect(padding, y_position, scaled_content_x0 - padding, page.rect.height - padding)
            margin_area_to_check = fitz.Rect(0, 0, scaled_content_x0, page.rect.height)
        elif original_location["column"] == 2:
            target_rect = fitz.Rect(scaled_content_x1 + padding, y_position, page.rect.width - padding, page.rect.height - padding)
            margin_area_to_check = fitz.Rect(scaled_content_x1, 0, page.rect.width, page.rect.height)
        else:
            return

        clean_def = " ".join(definition.replace('\r', ' ').replace('\n', ' ').split())
        full_text = f"{main_word}: {clean_def}"

        tw = fitz.TextWriter(page.rect)
        tw.fill_textbox(target_rect, full_text, fontsize=5)

        temp_doc = fitz.open()
        temp_page = temp_doc.new_page(width=page.rect.width, height=page.rect.height)
        tw.write_text(temp_page)
        blocks = temp_page.get_text("blocks")

        if not blocks:
            return
        new_text_bbox = fitz.Rect(blocks[0][:4])
        temp_doc.close()

        # --- Collision Check ----
        if is_margin_space_occupied(page, new_text_bbox, margin_area_to_check):
            print(f"  -> Skipping '{main_word}' due to detected overlap.")
            # TODO: Implement a better collision strategy, eg. shifting the new definition down until a free space is found
            return
        if using_llm:
            page.insert_textbox(target_rect, full_text, fontsize=5, fontname="helv", color=(0, 0.5, 0))  #Green
        else:
            page.insert_textbox(target_rect, full_text, fontsize=5, fontname="helv", color=(0.5, 0, 0))  #probably change color to grey using (0.2,0.2,0.2)

    except Exception as e:
        print(f"Error adding definition for '{main_word}': {e}")
