import uuid
import magic
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from schemas import (
    ExtractTextResponse,
    ChatQueryRequest,
    ChatQueryResponse,
    Citation,
    ChatSummary,
    ChatFileRecord,
    ChatMessageRecord,
    AppendMessageRequest,
    CreateChatRequest,
    CreateChatResponse,
    UploadJobQueued,
    UploadJobStatus,
)
from services.extraction_service import ExtractionService
from services.chunking_service import RAGPipelineService
from services.llm_service import LLMService
from services.storage_service import StorageService
from services.sqs_service import publish_upload_job
from rag.chunking import LayoutAwareChunker
from rag.embeddings import EmbeddingService
from rag.vector_store import QdrantStore
from rag.retriever import RAGRetriever
from rag.guardrails import RAGGuardrails
from db.database import get_db
from db import crud
from db.models import JobStatus

extraction_router = APIRouter(tags=["Extraction"])
chunking_router = APIRouter(tags=["Chunking"])
chat_router = APIRouter(tags=["Chat RAG"], prefix="/chat")
chats_router = APIRouter(tags=["Chat Sessions"], prefix="/chats")
jobs_router = APIRouter(tags=["Upload Jobs"], prefix="/jobs")

# Dependency Injection for our service class instances
def get_extraction_service() -> ExtractionService:
    return ExtractionService()

def get_llm_service() -> LLMService:
    return LLMService()


# ── Extraction Endpoints ─────────────────────────────────────────────────────

@extraction_router.post("/extract-text", response_model=ExtractTextResponse)
async def extract_text(
    file: UploadFile = File(...),
    service: ExtractionService = Depends(get_extraction_service)
):
    try:
        return await service.extract_text_from_file(file)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Text extraction pipeline failed: {str(e)}"
        )


# ── Chunking Endpoints ───────────────────────────────────────────────────────

chunking_service = RAGPipelineService()

@chunking_router.post("/extract-text-convert-chunks")
async def extract_text_chunks(file: UploadFile = File(...)):
    try:
        return await chunking_service.extract_and_chunk(file)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Text extraction pipeline failed: {str(e)}"
        )


@chunking_router.post("/semantic-chunking")
async def semantic_chunking_api(file: UploadFile = File(...)):
    try:
        return await chunking_service.semantic_chunking(file)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Text extraction pipeline failed: {str(e)}"
        )


# ── Chat Session Registry Endpoints (/chats) ──────────────────────────────────

@chats_router.get("", response_model=list[ChatSummary])
async def list_chat_sessions(db: AsyncSession = Depends(get_db)):
    """
    List all registered chat sessions, ordered by most recently active first.
    Used by the Streamlit sidebar to populate the session selectbox.
    """
    chats = await crud.list_chats(db)
    return [
        ChatSummary(
            chat_id=c.id,
            chat_name=c.chat_name,
            created_at=c.created_at,
            last_active_at=c.last_active_at,
        )
        for c in chats
    ]


@chats_router.post("", status_code=201, response_model=CreateChatResponse)
async def create_chat_session(
    request: CreateChatRequest = CreateChatRequest(),
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new chat session.
    - If chat_name is provided and already exists, returns 409.
    - If chat_name is empty or not provided, auto-generates Chat 1, Chat 2, etc.
    """
    # Resolve name
    if request.chat_name and request.chat_name.strip():
        chat_name = request.chat_name.strip()
        if await crud.chat_name_exists(db, chat_name):
            raise HTTPException(
                status_code=409,
                detail=f"A chat named '{chat_name}' already exists."
            )
    else:
        chat_name = await crud.get_next_chat_name(db)

    new_id = str(uuid.uuid4())
    await crud.create_chat(db, chat_id=new_id, chat_name=chat_name)
    return CreateChatResponse(chat_id=new_id, chat_name=chat_name)


@chats_router.get("/{chat_id}/files", response_model=list[ChatFileRecord])
async def list_chat_files(chat_id: str, db: AsyncSession = Depends(get_db)):
    """
    Return all files successfully indexed under the given chat session.
    Source of truth for the Streamlit sidebar file badges.
    """
    files = await crud.list_files(db, chat_id)
    return [
        ChatFileRecord(
            file_id=f.file_id,
            file_name=f.file_name,
            uploaded_at=f.uploaded_at,
            chunk_count=f.chunk_count,
        )
        for f in files
    ]


@chats_router.get("/{chat_id}/messages", response_model=list[ChatMessageRecord])
async def list_chat_messages(chat_id: str, db: AsyncSession = Depends(get_db)):
    """
    Return the full conversation history for a chat session, chronologically.
    Used by Streamlit to restore history when switching into a session.
    """
    messages = await crud.list_messages(db, chat_id)
    return [
        ChatMessageRecord(
            id=str(m.id),
            role=m.role,
            content=m.content,
            citations=m.citations or [],
            created_at=m.created_at,
        )
        for m in messages
    ]


@chats_router.post("/{chat_id}/messages", status_code=201)
async def append_chat_message(
    chat_id: str,
    request: AppendMessageRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Persist a new message (user or assistant) to the chat history.
    Also bumps the parent chat's last_active_at timestamp.
    """
    # Ensure the chat row exists (graceful handling for orphaned uploads)
    await crud.upsert_chat(db, chat_id)
    await crud.append_message(
        db,
        chat_id=chat_id,
        role=request.role,
        content=request.content,
        citations=request.citations or [],
    )
    return {"success": True}


# ── Chat RAG Endpoints (/chat) ────────────────────────────────────────────────

# Allowed MIME types and their canonical labels (server-side validation)
_ALLOWED_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
    "image/jpeg",
    "image/png",
    "image/webp",
}


@chat_router.post("/{chat_id}/upload", status_code=202, response_model=list[UploadJobQueued])
async def upload_chat_files(
    chat_id: str,
    files: List[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Accepts one or more files, saves each to local disk, creates an upload_jobs
    row per file, publishes a message to SQS, and returns 202 immediately.
    Processing (extraction → chunking → embedding → Qdrant) happens in the worker.
    """
    await crud.upsert_chat(db, chat_id)
    queued_jobs = []

    for file in files:
        file_bytes = await file.read()

        if not file_bytes:
            raise HTTPException(status_code=400, detail=f"Empty file: {file.filename}")

        # Server-side MIME detection from actual bytes — don't trust client content_type
        detected_mime = magic.from_buffer(file_bytes, mime=True)
        if detected_mime not in _ALLOWED_MIME_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{detected_mime}' for file '{file.filename}'"
            )

        file_id = str(uuid.uuid4())
        storage_path = StorageService.save(file_id, file_bytes)

        job = await crud.create_upload_job(
            db,
            chat_id=chat_id,
            file_id=file_id,
            file_name=file.filename,
            file_type=detected_mime,
            storage_path=storage_path,
        )

        publish_upload_job(
            job_id=str(job.id),
            chat_id=chat_id,
            file_id=file_id,
            file_name=file.filename,
            file_type=detected_mime,
            storage_path=storage_path,
        )

        queued_jobs.append(UploadJobQueued(
            job_id=str(job.id),
            file_name=file.filename,
            status=JobStatus.QUEUED,
        ))

    return queued_jobs


# ── Job Status Endpoints (/jobs) ──────────────────────────────────────────────

@jobs_router.get("/{job_id}", response_model=UploadJobStatus)
async def get_job_status(job_id: str, db: AsyncSession = Depends(get_db)):
    """Returns current status of a single upload job."""
    job = await crud.get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return UploadJobStatus(
        job_id=str(job.id),
        chat_id=job.chat_id,
        file_id=job.file_id,
        file_name=job.file_name,
        file_type=job.file_type,
        status=job.status,
        error_message=job.error_message,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


@chats_router.get("/{chat_id}/jobs", response_model=list[UploadJobStatus])
async def list_chat_jobs(chat_id: str, db: AsyncSession = Depends(get_db)):
    """Returns status of all upload jobs for a chat session."""
    jobs = await crud.list_jobs_for_chat(db, chat_id)
    return [
        UploadJobStatus(
            job_id=str(j.id),
            chat_id=j.chat_id,
            file_id=j.file_id,
            file_name=j.file_name,
            file_type=j.file_type,
            status=j.status,
            error_message=j.error_message,
            created_at=j.created_at,
            updated_at=j.updated_at,
        )
        for j in jobs
    ]


@chat_router.post("/{chat_id}/query", response_model=ChatQueryResponse)
async def query_chat_session(
    chat_id: str,
    request: ChatQueryRequest,
    llm_service: LLMService = Depends(get_llm_service)
):
    """
    Queries a specific chat session with strict context isolation.
    Applies input injection checks, retrieves relevant chunks, enforces token limits,
    synthesizes a grounded response, and cleans up hallucinated citations.
    """
    try:
        # 1. Guardrail: Input Safety Validation
        RAGGuardrails.validate_query(request.query)

        # 2. Retrieve relevant chunks strictly filtered by chat_id
        retriever = RAGRetriever()
        retrieved_chunks = await retriever.retrieve_relevant_chunks(
            chat_id=chat_id,
            query=request.query,
            limit=5
        )

        # 3. Guardrail: Context Token Budgeting (truncates context if it exceeds budget)
        budget_chunks = RAGGuardrails.enforce_token_budget(retrieved_chunks, max_tokens=6000)

        # 4. Generate grounded response from LLM
        raw_answer = await llm_service.generate_grounded_response(request.query, budget_chunks)

        # 5. Guardrail: Validate generated citations and strip hallucinated ones
        clean_answer = RAGGuardrails.validate_and_clean_citations(raw_answer, budget_chunks)

        # 6. Guardrail: Standardize any refusal responses
        final_answer = RAGGuardrails.standardize_refusal(clean_answer)

        # 7. Package structured citations for granular attribution
        citations = [
            Citation(
                file_name=chunk.file_name,
                page_number=chunk.page_number,
                element_type=chunk.element_type,
                content=chunk.content
            )
            for chunk in budget_chunks
        ]

        return ChatQueryResponse(
            success=True,
            answer=final_answer,
            citations=citations
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Chat query failed: {str(e)}"
        )