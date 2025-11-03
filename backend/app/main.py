from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
import shutil, time, fitz, traceback

from backend.services import parser, pdf_transform, definitions, tough_words, definition_tough_word
from backend.models.schemas import AnnotateResponse

UPLOAD_DIR = Path(__file__).resolve().parents[1] / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="PaperAnnotator")

@app.post("/annotate/", response_model=AnnotateResponse)
async def annotate(file: UploadFile = File(...), scaling: float = 0.9):
    timestamp = int(time.time())
    dest = UPLOAD_DIR / f"uploaded_{timestamp}.pdf"

    try:
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")

    try:
        original_doc = fitz.open(str(dest))
        scaled_doc, original_bboxes = pdf_transform.scale_content_horizontally(original_doc, scaling)

        # --- References (untouched) ---
        refs = parser.find_references(original_doc)
        if not refs:
            raise HTTPException(status_code=400, detail="No references found in the document.")

        processed = 0
        for ref in refs:
            try:
                title = definitions.get_definition_for_reference(ref, pdf_path=str(dest))
                if not title:
                    continue
                location_data = {"page": ref["page"], "column": ref["column"], "bbox": ref["bbox"]}
                pdf_transform.add_definition_to_margin(
                    doc=scaled_doc,
                    scaling_factor=scaling,
                    main_word=f"Ref {ref['text']}",
                    definition=title,
                    original_location=location_data,
                    original_content_bboxes=original_bboxes,
                )
                processed += 1
            except Exception as inner_error:
                print(f"Error processing reference {ref['text']}: {inner_error}")
                continue

        # === Tough Words ===
        full_text = "".join(page.get_text("text") for page in original_doc)
        inline_defs = definition_tough_word.build_inline_definitions(full_text)
        known_vocab: set[str] = set()

        # Collect all unique tough words
        all_tough = set()
        page_word_map = {}
        for page_num in range(len(original_doc)):
            page = original_doc[page_num]
            toughs = tough_words.extract_tough_words_from_page(page, known_vocab)
            page_word_map.setdefault(page_num, []).extend(toughs)
            for word, _ in toughs:
                if word.lower() not in inline_defs:
                    all_tough.add(word)

        # Batch resolve unresolved words with Gemini
        gemini_defs = definition_tough_word.batch_resolve_with_gemini(list(all_tough))

        # Merge maps
        definition_cache = {**inline_defs, **{k.lower(): v for k, v in gemini_defs.items()}}

        # Annotate
        for page_num, toughs in page_word_map.items():
            for word, bbox in toughs:
                try:
                    meaning = definition_cache.get(word.lower())
                    if not meaning:
                        continue
                    location_data = {"page": page_num, "column": 0, "bbox": bbox}
                    pdf_transform.add_definition_to_margin(
                        doc=scaled_doc,
                        scaling_factor=scaling,
                        main_word=word,
                        definition=meaning,
                        original_location=location_data,
                        original_content_bboxes=original_bboxes,
                    )
                    processed += 1
                except Exception as inner_error:
                    print(f"Error processing tough word {word}: {inner_error}")
                    continue

        out_path = UPLOAD_DIR / f"annotated_{timestamp}.pdf"
        scaled_doc.save(str(out_path))
        return AnnotateResponse(output_path=str(out_path), processed=processed)

    except Exception as e:
        print("=== FULL TRACEBACK ===")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
