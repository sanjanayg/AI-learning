from pydantic import BaseModel
from typing import Optional


class ImageTextResponse(BaseModel):
    success: bool
    extracted_text: str
    

class ExtractTextResponse(BaseModel):
    success: bool
    file_type: str
    extraction_method: str
    extracted_text: str

class ErrorResponse(BaseModel):
    success: bool
    error: str