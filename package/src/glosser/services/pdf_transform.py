import pymupdf
import os
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from pylatexenc.latex2text import LatexNodes2Text
except ImportError:
    LatexNodes2Text = None

_unicode_font_path = None
_unicode_font_checked = False

def get_unicode_font_path():
    global _unicode_font_path, _unicode_font_checked
    if _unicode_font_checked:
        return _unicode_font_path
    _unicode_font_checked = True
    common_fonts = [
        "C:/Windows/Fonts/cambria.ttc",
        "C:/Windows/Fonts/seguiemj.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/times.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf"
    ]
    for f in common_fonts:
        if os.path.exists(f):
            _unicode_font_path = f
            return f
    return None

def get_page_content_bbox(page: pymupdf.Page, padding=0) -> pymupdf.Rect:
    blocks = page.get_text("blocks")
    if not blocks:
        return page.cropbox

    x0, y0, x1, y1 = blocks[0][0:4]
    for b in blocks[1:]:
        x0 = min(x0, b[0])
        y0 = min(y0, b[1])
        x1 = max(x1, b[2])
        y1 = max(y1, b[3])

    return pymupdf.Rect(x0 - padding, y0, x1 + padding, y1)


def scale_content_horizontally(doc: pymupdf.Document, scaling_factor: float, progress_callback: callable = None) -> tuple[pymupdf.Document, list]:
    """
      - increase the page width by `scaling_factor`.
      - Place original content unscaled and centered horizontally on the wider page.
      - Return the new document and a list of the content bboxes adjusted to the new page coordinates.
    """
    if scaling_factor <= 0:
        raise ValueError("scaling_factor must be positive.")
    new_doc = pymupdf.open()
    original_content_bboxes = []
    num_pages = len(doc)

    for i, source_page in enumerate(doc):
        if progress_callback:
            progress_callback(i, num_pages)
        src_w = source_page.rect.width
        src_h = source_page.rect.height

        new_w = src_w * scaling_factor
        new_h = src_h

        x_offset = (new_w - src_w) / 2.0
        dest_rect = pymupdf.Rect(x_offset, 0, x_offset + src_w, src_h)

        new_page = new_doc.new_page(width=new_w, height=new_h)

        new_page.show_pdf_page(dest_rect, doc, source_page.number)

        content_bbox = get_page_content_bbox(source_page)
        shifted_bbox = pymupdf.Rect(
            content_bbox.x0 + x_offset,
            content_bbox.y0,
            content_bbox.x1 + x_offset,
            content_bbox.y1,
        )
        original_content_bboxes.append(shifted_bbox)

    return new_doc, original_content_bboxes


def is_margin_space_occupied(page: pymupdf.Page, new_text_bbox: pymupdf.Rect, margin_area: pymupdf.Rect) -> bool:
    existing_blocks = page.get_text("blocks")

    for block in existing_blocks:
        existing_bbox = pymupdf.Rect(block[:4])
        if margin_area.contains(existing_bbox) and new_text_bbox.intersects(existing_bbox):
            return True

    return False


def add_definition_to_margin(
    doc: pymupdf.Document,
    scaling_factor: float,
    main_word: str,
    definition: str,
    original_location: dict,
    original_content_bboxes: list,
    using_llm: bool = False,
) -> bool:
    page_num = original_location["page"]
    page = doc[page_num]

    original_bbox = pymupdf.Rect(original_location["bbox"])
    content_bbox = original_content_bboxes[page_num]

    scaled_content_x0 = content_bbox.x0
    scaled_content_x1 = content_bbox.x1

    y_position = original_bbox.y0 + 3
    padding = 5

    if original_location["column"] == 1:
        target_rect = pymupdf.Rect(padding, y_position, scaled_content_x0 - padding, page.rect.height - padding)
        margin_area_to_check = pymupdf.Rect(0, 0, scaled_content_x0, page.rect.height)
    elif original_location["column"] == 2:
        target_rect = pymupdf.Rect(scaled_content_x1 + padding, y_position, page.rect.width - padding, page.rect.height - padding)
        margin_area_to_check = pymupdf.Rect(scaled_content_x1, 0, page.rect.width, page.rect.height)
    else:
        return False

    clean_def = " ".join(definition.replace('\r', ' ').replace('\n', ' ').split())
    full_text = f"{main_word}: {clean_def}"

    font_path = get_unicode_font_path()

    tw = pymupdf.TextWriter(page.rect)
    if font_path:
        tw.fill_textbox(target_rect, full_text, font=pymupdf.Font(fontfile=font_path), fontsize=5)
    else:
        tw.fill_textbox(target_rect, full_text, fontsize=5)

    temp_doc = pymupdf.open()
    temp_page = temp_doc.new_page(width=page.rect.width, height=page.rect.height)
    tw.write_text(temp_page)
    blocks = temp_page.get_text("blocks")

    if not blocks:
        temp_doc.close()
        return False
    new_text_bbox = pymupdf.Rect(blocks[0][:4])
    temp_doc.close()

    if is_margin_space_occupied(page, new_text_bbox, margin_area_to_check):
        return False

    if using_llm:
        if font_path:
            page.insert_textbox(target_rect, full_text, fontsize=5, fontfile=font_path, color=(0.5,0, 0))
        else:
            page.insert_textbox(target_rect, full_text, fontsize=5, fontname="helv", color=(0.5,0, 0))
    else:
        if font_path:
            page.insert_textbox(target_rect, full_text, fontsize=5, fontfile=font_path, color=(0, 0.5, 0))
        else:
            page.insert_textbox(target_rect, full_text, fontsize=5, fontname="helv", color=(0, 0.5, 0))

    return True


def add_symbol_definition_to_margin(
    doc: pymupdf.Document,
    scaling_factor: float,
    symbol: str,
    meaning: str,
    description: str,
    original_location: dict,
    original_content_bboxes: list,
    is_inferred: bool = False,
) -> bool:
    if meaning in ["NOT_FOUND", "", None]:
        return False

    page_num = original_location["page"]
    page = doc[page_num]

    original_bbox = pymupdf.Rect(original_location["bbox"])
    content_bbox = original_content_bboxes[page_num]

    scaled_content_x0 = content_bbox.x0
    scaled_content_x1 = content_bbox.x1

    y_position = original_bbox.y0 + 3
    padding = 5

    if original_location["column"] == 1:
        target_rect = pymupdf.Rect(padding, y_position, scaled_content_x0 - padding, page.rect.height - padding)
        margin_area_to_check = pymupdf.Rect(0, 0, scaled_content_x0, page.rect.height)
    elif original_location["column"] == 2:
        target_rect = pymupdf.Rect(scaled_content_x1 + padding, y_position, page.rect.width - padding, page.rect.height - padding)
        margin_area_to_check = pymupdf.Rect(scaled_content_x1, 0, page.rect.width, page.rect.height)
    else:
        return False

    clean_meaning = " ".join(meaning.replace('\r', ' ').replace('\n', ' ').split())
    has_desc = description and description.upper() != "NOT_FOUND" and description.lower() != "none"
    clean_desc = " ".join(description.replace('\r', ' ').replace('\n', ' ').split()) if has_desc else ""

    img_bytes = None
    pdf_img_w = 0
    pdf_img_h = 0
    
    try:
        import io
        from PIL import Image

        clean_symbol = re.sub(r'\\(?:bf|rm|it|cal|textbf|textit|mathrm|mathcal|mathbf)\s*', '', symbol).strip()
        math_str = f"${clean_symbol}$" if not clean_symbol.startswith('$') else clean_symbol

        fig = plt.figure(figsize=(0.1, 0.1))
        try:
            fig.text(0, 0, math_str, fontsize=10, ha='left', va='bottom', color='#008000')
            buf = io.BytesIO()
            plt.savefig(buf, format='png', transparent=True, bbox_inches='tight', pad_inches=0.01, dpi=300)
            img_bytes = buf.getvalue()
        finally:
            plt.close(fig)
        img_pil = Image.open(io.BytesIO(img_bytes))
        
        target_h = 7 
        aspect_ratio = img_pil.width / img_pil.height
        pdf_img_h = target_h
        pdf_img_w = target_h * aspect_ratio
        
    except Exception:
        pass
        
    font_path = get_unicode_font_path()
    tw = pymupdf.TextWriter(page.rect)
    
    x_offset = 4 
    y_offset = 1.8 
    
    if img_bytes:
        img_rect = pymupdf.Rect(
            target_rect.x0 + x_offset, 
            target_rect.y0 + y_offset, 
            target_rect.x0 + x_offset + pdf_img_w, 
            target_rect.y0 + y_offset + pdf_img_h
        )
        
        text_str = f"{clean_meaning}: {clean_desc}" if has_desc else f"{clean_meaning}"
        text_rect = pymupdf.Rect(img_rect.x1 + 3, target_rect.y0, target_rect.x1, target_rect.y1)
    else:
        display_symbol = symbol
        if LatexNodes2Text is not None and ('\\' in symbol):
            try:
                clean_sym = re.sub(r'\\(?:bf|rm|it|cal|textbf|textit|mathrm|mathcal|mathbf)\s*', '', symbol)
                result = LatexNodes2Text().latex_to_text(clean_sym).strip()
                display_symbol = result if result and '%' not in result else symbol
            except Exception:
                pass
        text_str = f"[{display_symbol}] {clean_meaning}: {clean_desc}" if has_desc else f"[{display_symbol}] {clean_meaning}"
        text_rect = target_rect

    if font_path:
        tw.fill_textbox(text_rect, text_str, font=pymupdf.Font(fontfile=font_path), fontsize=5)
    else:
        tw.fill_textbox(text_rect, text_str, fontsize=5)

    temp_doc = pymupdf.open()
    temp_page = temp_doc.new_page(width=page.rect.width, height=page.rect.height)
    tw.write_text(temp_page)
    blocks = temp_page.get_text("blocks")
    temp_doc.close()

    if not blocks:
        return False
        
    new_text_bbox = pymupdf.Rect(blocks[0][:4])
    total_bbox = (new_text_bbox | img_rect) if img_bytes else new_text_bbox

    if is_margin_space_occupied(page, total_bbox, margin_area_to_check):
        return False
        
    if img_bytes:
        page.insert_image(img_rect, stream=img_bytes)
        
    color = (0.5, 0, 0) if is_inferred else (0, 0.5, 0)

    if font_path:
        page.insert_textbox(text_rect, text_str, fontsize=5, fontfile=font_path, color=color)
    else:
        page.insert_textbox(text_rect, text_str, fontsize=5, fontname="helv", color=color)

    return True