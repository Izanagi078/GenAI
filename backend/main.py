"""Legacy shim - guidance for running the new FastAPI app.

The functionality of this script was migrated into the package-based API
at `backend.app.main` (FastAPI). To run the web API, use uvicorn (from root dir):

    uvicorn backend.app.main:app --reload

If you need to run the original CLI-style flow, import and call the
functions from `backend.services.parser` and `backend.services.pdf_transform`.
"""

from backend.app.main import app  # re-export the FastAPI app for uvicorn

__all__ = ["app"]