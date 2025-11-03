from typing import List, Tuple, Set

def extract_tough_words_from_page(page, known_vocab: Set[str]) -> List[Tuple[str, Tuple[float, float, float, float]]]:
    """
    Extract tough words and their bounding boxes from a page.
    Returns list of (word, bbox).
    Heuristic: words longer than 6 chars and not in known_vocab.
    """
    words = page.get_text("words")  # [(x0, y0, x1, y1, word, block_no, line_no, word_no)]
    tough: List[Tuple[str, Tuple[float, float, float, float]]] = []
    for x0, y0, x1, y1, w, *_ in words:
        lw = w.lower()
        if lw not in known_vocab and len(w) > 6:
            tough.append((w, (x0, y0, x1, y1)))
    return tough
