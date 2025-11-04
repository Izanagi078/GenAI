from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import time
from .config import GOOGLE_API_KEY

from backend.services import parser, pdf_transform, definitions
from backend.models.schemas import AnnotateResponse

UPLOAD_DIR = Path(__file__).resolve().parents[1] / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="PaperAnnotator")


@app.post("/annotate/", response_model=AnnotateResponse)
async def annotate(file: UploadFile = File(...), scaling: float = 0.9):
    # Save the uploaded file
    timestamp = int(time.time())
    dest = UPLOAD_DIR / f"uploaded_{timestamp}.pdf"
    try:
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")

    try:
        import fitz

        original_doc = fitz.open(str(dest))

        # Scale to create margins
        scaled_doc, original_bboxes = pdf_transform.scale_content_horizontally(original_doc, scaling)

        refs = parser.find_references(original_doc)
        g_key= GOOGLE_API_KEY

        processed = 0
        for ref in refs:
            title = definitions.get_title_for_reference_langchain(ref, pdf_path=str(dest),google_api_key=g_key)
            if title:
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

        out_path = UPLOAD_DIR / f"annotated_{timestamp}.pdf"
        scaled_doc.save(str(out_path))

        return JSONResponse(status_code=200, content={"output_path": str(out_path), "processed": processed})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
