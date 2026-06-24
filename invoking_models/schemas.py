from pydantic import BaseModel
from typing import Optional, List


class ImageTextResponse(BaseModel):
    success: bool
    extracted_text: str


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


class ErrorResponse(BaseModel):
    success: bool
    error: str


class DocumentChunk(BaseModel):
    chunk_id: str
    chat_id: str
    file_id: str
    file_name: str
    page_number: int
    element_type: str  # e.g., 'text_paragraph', 'structural_table', 'image_ocr'
    content: str
    token_count: int


class Citation(BaseModel):
    file_name: str
    page_number: int
    element_type: str
    content: str


class ChatQueryRequest(BaseModel):
    query: str


class ChatQueryResponse(BaseModel):
    success: bool
    answer: str
    citations: List[Citation]