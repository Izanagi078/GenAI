from pydantic import BaseModel


class AnnotateResponse(BaseModel):
    output_path: str
    processed: int
