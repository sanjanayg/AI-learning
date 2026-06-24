from pydantic import BaseModel
from typing import Optional,List


class ImageTextResponse(BaseModel):
    success: bool
    extracted_text: str
    

class ErrorResponse(BaseModel):
    success: bool
    error: str

class ExtractedPage(BaseModel):
    page_number: int
    text: str
    extraction_method: str


class ExtractTextResponse(BaseModel):
    success: bool
    file_name: str
    file_type: str
    extraction_method: str
    extracted_text: str
    pages: Optional[List[ExtractedPage]] = None