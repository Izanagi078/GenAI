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

from .visual_design import ConfidenceVisualizer, TypographyOptimizer, LayoutOptimizer

_unicode_font_path = None
_unicode_font_checked = False

def get_unicode_font_path():
    global _unicode_font_path, _unicode_font_checked
    if _unicode_font_checked:
        return _unicode_font_path
    _unicode_font_checked = True
    common_fonts = [
        # Windows
        "C:/Windows/Fonts/cambria.ttc",
        "C:/Windows/Fonts/seguiemj.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/times.ttf",
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/opentype/noto/NotoSans-Regular.ttf",
        # macOS
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Geneva.ttf",
    ]
    for f in common_fonts:
        if os.path.exists(f):
            _unicode_font_path = f
            return f

    # Last-resort: ask matplotlib's font manager to locate any TrueType font
    # on the system – it searches platform-specific font directories for us.
    try:
        from matplotlib import font_manager as _fm
        candidates = _fm.findSystemFonts(fontext="ttf")
        # Prefer well-known Unicode-capable families
        preferred = ("dejavu", "liberation", "ubuntu", "freesans", "noto", "arial", "helvetica")
        for path in candidates:
            if any(n in os.path.basename(path).lower() for n in preferred):
                _unicode_font_path = path
                return path
        # Accept any TTF if nothing preferred was found
        if candidates:
            _unicode_font_path = candidates[0]
            return candidates[0]
    except Exception:
        pass

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
    using_llm: bool = False,  # Deprecated - kept for backward compatibility
    confidence: str = None,  # VIS: Use "HIGH", "MEDIUM", or "LOW"
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

    # VIS: Map legacy using_llm to confidence levels
    if confidence is None:
        confidence = "MEDIUM" if using_llm else "HIGH"

    clean_def = " ".join(definition.replace('\r', ' ').replace('\n', ' ').split())

    # VIS: Use ConfidenceVisualizer for multi-channel encoding
    full_text = ConfidenceVisualizer.format_annotation(
        main_word,
        clean_def,
        confidence,
        include_icon=True
    )

    # VIS: Dynamic font sizing based on text length
    margin_width = target_rect.width
    font_size = TypographyOptimizer.get_font_size(full_text, margin_width)

    font_path = get_unicode_font_path()

    tw = pymupdf.TextWriter(page.rect)
    if font_path:
        tw.fill_textbox(target_rect, full_text, font=pymupdf.Font(fontfile=font_path), fontsize=font_size)
    else:
        try:
            tw.fill_textbox(target_rect, full_text, font=pymupdf.Font(fontname="ubuntu"), fontsize=font_size)
        except Exception:
            tw.fill_textbox(target_rect, full_text, fontsize=font_size)

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

    # VIS: Multi-channel visual encoding using TextWriter (correct Unicode + native opacity)
    # TextWriter correctly sets up ToUnicode CMap, fixing icon extraction issues
    color = ConfidenceVisualizer.get_color(confidence)
    alpha = ConfidenceVisualizer.get_alpha(confidence)

    # Re-use the same TextWriter and write directly to page with color + opacity
    tw_render = pymupdf.TextWriter(page.rect)
    if font_path:
        tw_render.fill_textbox(target_rect, full_text, font=pymupdf.Font(fontfile=font_path), fontsize=font_size)
    else:
        try:
            tw_render.fill_textbox(target_rect, full_text, font=pymupdf.Font(fontname="ubuntu"), fontsize=font_size)
        except Exception:
            tw_render.fill_textbox(target_rect, full_text, fontsize=font_size)

    tw_render.write_text(page, color=color, opacity=alpha)

    return True


def add_confidence_legend(doc: pymupdf.Document) -> None:
    """
    Add a confidence legend box to the bottom-right corner of the first page.
    Explains the color/icon encoding used throughout the document.
    """
    if len(doc) == 0:
        return

    page = doc[0]
    font_path = get_unicode_font_path()
    font_size = 5.0

    entries = [
        ((0.0, 0.45, 0.0), "Green  Extracted from paper"),
        ((0.7, 0.35, 0.0), "≈  Orange  LLM-inferred"),
        ((0.55, 0.0, 0.0), "?  Red  Uncertain"),
    ]

    legend_w = 100
    legend_h = 36
    margin = 6
    x0 = page.rect.width - legend_w - margin
    y0 = page.rect.height - legend_h - margin

    box_rect = pymupdf.Rect(x0 - 3, y0 - 3, x0 + legend_w + 3, y0 + legend_h + 3)
    page.draw_rect(box_rect, color=(0.7, 0.7, 0.7), fill=(1.0, 1.0, 1.0), width=0.4)

    tw_title = pymupdf.TextWriter(page.rect)
    title_rect = pymupdf.Rect(x0, y0, x0 + legend_w, y0 + font_size + 2)
    if font_path:
        tw_title.fill_textbox(title_rect, "GlossVis Confidence:", font=pymupdf.Font(fontfile=font_path), fontsize=font_size - 0.5)
    else:
        tw_title.fill_textbox(title_rect, "GlossVis Confidence:", fontsize=font_size - 0.5)
    tw_title.write_text(page, color=(0.15, 0.15, 0.15))

    y_entry = y0 + font_size + 4
    for color, label in entries:
        tw = pymupdf.TextWriter(page.rect)
        entry_rect = pymupdf.Rect(x0, y_entry, x0 + legend_w, y_entry + font_size + 2)
        if font_path:
            tw.fill_textbox(entry_rect, label, font=pymupdf.Font(fontfile=font_path), fontsize=font_size)
        else:
            tw.fill_textbox(entry_rect, label, fontsize=font_size)
        tw.write_text(page, color=color)
        y_entry += font_size + 2


def add_symbol_definition_to_margin(
    doc: pymupdf.Document,
    scaling_factor: float,
    symbol: str,
    meaning: str,
    description: str,
    original_location: dict,
    original_content_bboxes: list,
    is_inferred: bool = False,  # Deprecated - kept for backward compatibility
    confidence: str = None,  # VIS: Use "HIGH", "MEDIUM", or "LOW"
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

    # VIS: Map legacy is_inferred to confidence levels
    if confidence is None:
        confidence = "MEDIUM" if is_inferred else "HIGH"

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
            # VIS: Match symbol color to text color based on confidence level
            symbol_color = ConfidenceVisualizer.get_color(confidence)
            fig.text(0, 0, math_str, fontsize=10, ha='left', va='bottom', color=symbol_color)
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
        
        # icon prefix: empty for HIGH (extracted), ≈/? for inferred
        icon = ConfidenceVisualizer.get_icon(confidence) if confidence else "≈"
        base_str = f"{clean_meaning}: {clean_desc}" if has_desc else f"{clean_meaning}"
        text_str = f"{icon} {base_str}".strip() if icon else base_str
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
        icon = ConfidenceVisualizer.get_icon(confidence) if confidence else "≈"
        base_str = f"[{display_symbol}] {clean_meaning}: {clean_desc}" if has_desc else f"[{display_symbol}] {clean_meaning}"
        text_str = f"{icon} {base_str}".strip() if icon else base_str
        text_rect = target_rect

    if font_path:
        tw.fill_textbox(text_rect, text_str, font=pymupdf.Font(fontfile=font_path), fontsize=5)
    else:
        try:
            tw.fill_textbox(text_rect, text_str, font=pymupdf.Font(fontname="ubuntu"), fontsize=5)
        except Exception:
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

    # VIS: Multi-channel visual encoding
    color = ConfidenceVisualizer.get_color(confidence)
    alpha = ConfidenceVisualizer.get_alpha(confidence)

    # VIS: Dynamic font sizing
    margin_width = text_rect.width
    font_size = TypographyOptimizer.get_font_size(text_str, margin_width)

    if img_bytes:
        page.insert_image(img_rect, stream=img_bytes)

    # Use TextWriter for correct Unicode rendering (icon glyphs + ToUnicode CMap)
    tw_render = pymupdf.TextWriter(page.rect)
    if font_path:
        tw_render.fill_textbox(text_rect, text_str, font=pymupdf.Font(fontfile=font_path), fontsize=font_size)
    else:
        try:
            tw_render.fill_textbox(text_rect, text_str, font=pymupdf.Font(fontname="ubuntu"), fontsize=font_size)
        except Exception:
            tw_render.fill_textbox(text_rect, text_str, fontsize=font_size)

    tw_render.write_text(page, color=color, opacity=alpha)

    return True