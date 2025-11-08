from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import time
from .config import GOOGLE_API_KEY
from typing import Annotated

from backend.services import parser, pdf_transform, definitions
from backend.models.schemas import AnnotateResponse

UPLOAD_DIR = Path(__file__).resolve().parents[1] / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="PaperAnnotator")


@app.post("/annotate/", response_model=AnnotateResponse)
async def annotate(file: UploadFile = File(...), scaling: float = 0.9, find_references: bool = True, find_abbreviation: bool = True):
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

        # Build references database
        refs_db = parser.build_references_db(original_doc, GOOGLE_API_KEY)

        # Scale to create margins
        scaled_doc, original_bboxes = pdf_transform.scale_content_horizontally(original_doc, scaling)
        if find_references:
            refs = parser.find_references(original_doc)

            processed = 0
            for ref in refs:
                print("Processing ref:", ref)
                ref_info = refs_db.get(ref['number'])
                if ref_info and (ref_info.get('title') or ref_info.get('year')):
                    title = ref_info.get('title') or ""
                    year = ref_info.get('year') or ""
                    definition = f"{title} ({year})".strip() if title or year else "Reference not found"
                else:
                    definition = "Reference not found"

                if definition != "Reference not found":
                    print("found definition: ", definition)
                    location_data = {"page": ref["page"], "column": ref["column"], "bbox": ref["bbox"]}
                    pdf_transform.add_definition_to_margin(
                        doc=scaled_doc,
                        scaling_factor=scaling,
                        main_word=f"Ref {ref['text']}",
                        definition=definition,
                        original_location=location_data,
                        original_content_bboxes=original_bboxes,
                    )
                    processed += 1

        if find_abbreviation:
            abbs = parser.find_abbreviations(original_doc)

            processed = 0
            # Using set to avoid duplicate lookups for the same abbreviation
            unique_abbreviations = {abbr["text"] for abbr in abbs}
            print("Unique abbreviations found:", len(unique_abbreviations))

            # Find all full forms first to avoid processing duplicates
            full_form_map = {}
            for abbr_text in list(unique_abbreviations)[:20]:  # Limit API calls
                print("API CALL for:", abbr_text)
                full_form = definitions.find_full_form(
                    pdf_path=str(dest),
                    abbr=abbr_text,
                    google_api_key=GOOGLE_API_KEY
                )
                if full_form:
                    full_form_map[abbr_text] = full_form

            # Track how many times each abbreviation has been seen
            abbr_occurrence_count = {}

            # Now iterate through the original list to place annotations
            for abbr in abbs:
                abbr_text = abbr["text"]
                abbr_occurrence_count[abbr_text] = abbr_occurrence_count.get(abbr_text, 0) + 1

                # Add the definition ONLY when it appears for the second time
                if (
                    abbr_text in full_form_map and 
                    abbr_occurrence_count[abbr_text] == 2  # second appearance
                ):
                    full_form = full_form_map[abbr_text]
                    
                    location_data = {
                        "page": abbr["page"],
                        "column": abbr["column"],
                        "bbox": abbr["bbox"]
                    }
                    pdf_transform.add_definition_to_margin(
                        doc=scaled_doc,
                        scaling_factor=scaling,
                        main_word=abbr_text,
                        definition=full_form,
                        original_location=location_data,
                        original_content_bboxes=original_bboxes,
                    )

                    processed += 1

        out_path = UPLOAD_DIR / f"annotated_{timestamp}.pdf"
        scaled_doc.save(str(out_path))

        return JSONResponse(status_code=200, content={"output_path": str(out_path), "processed": processed})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")


@app.post("/query/")
async def query(file: UploadFile = File(...), query: str = Form(...)):
    # Save the uploaded file
    timestamp = int(time.time())
    name = Path(file.filename).name
    dest = UPLOAD_DIR / f"{name}"
    try:
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")

    try:
        # Get the answer from the PDF
        answer = definitions.find_answer(pdf_path=str(dest), query=query, google_api_key=GOOGLE_API_KEY)

        return JSONResponse(status_code=200, content={"answer": answer})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
