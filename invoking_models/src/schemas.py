from datetime import datetime
from pydantic import BaseModel
from typing import Optional, List, Any



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


# ── DB-backed chat session schemas ─────────────────────────────────────────────

class ChatSummary(BaseModel):
    """Response item for GET /chats — one entry per registered chat session."""
    chat_id: str
    created_at: datetime
    last_active_at: datetime
    chat_name:str

    class Config:
        from_attributes = True


class ChatFileRecord(BaseModel):
    """Response item for GET /chats/{chat_id}/files."""
    file_id: str
    file_name: str
    uploaded_at: datetime
    chunk_count: int

    class Config:
        from_attributes = True


class ChatMessageRecord(BaseModel):
    """Response item for GET /chats/{chat_id}/messages."""
    id: str
    role: str
    content: str
    citations: List[Any]
    created_at: datetime

    class Config:
        from_attributes = True


class AppendMessageRequest(BaseModel):
    """Request body for POST /chats/{chat_id}/messages."""
    role: str                              # "user" | "assistant"
    content: str
    citations: Optional[List[Any]] = []


class CreateChatRequest(BaseModel):
    """Request body for POST /chats. chat_name is optional — auto-generated if blank."""
    chat_name: Optional[str] = None


class CreateChatResponse(BaseModel):
    """Response for POST /chats."""
    chat_id: str
    chat_name: str


# ── Upload Job schemas ────────────────────────────────────────────────────────

class UploadJobQueued(BaseModel):
    """Returned immediately per file after POST /chat/{chat_id}/upload."""
    job_id: str
    file_name: str
    status: str = "queued"


class UploadJobStatus(BaseModel):
    """Returned by GET /jobs/{job_id} and GET /chats/{chat_id}/jobs."""
    job_id: str
    chat_id: str
    file_id: str
    file_name: str
    file_type: str
    status: str
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True