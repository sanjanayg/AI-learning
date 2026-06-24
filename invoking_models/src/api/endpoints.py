import uuid
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
)
from services.extraction_service import ExtractionService
from services.chunking_service import RAGPipelineService
from services.llm_service import LLMService
from rag.chunking import LayoutAwareChunker
from rag.embeddings import EmbeddingService
from rag.vector_store import QdrantStore
from rag.retriever import RAGRetriever
from rag.guardrails import RAGGuardrails
from db.database import get_db
from db import crud

extraction_router = APIRouter(tags=["Extraction"])
chunking_router = APIRouter(tags=["Chunking"])
chat_router = APIRouter(tags=["Chat RAG"], prefix="/chat")
chats_router = APIRouter(tags=["Chat Sessions"], prefix="/chats")

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


@chats_router.post("", status_code=201)
async def create_chat_session(db: AsyncSession = Depends(get_db)):
    """
    Create a new chat session with a server-generated UUID.
    Returns the new chat_id so the client can immediately switch into it.
    """
    new_id = str(uuid.uuid4())
    await crud.create_chat(db, chat_id=new_id)
    return {"chat_id": new_id}


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

@chat_router.post("/{chat_id}/upload")
async def upload_chat_file(
    chat_id: str,
    file: UploadFile = File(...),
    extraction_service: ExtractionService = Depends(get_extraction_service),
    db: AsyncSession = Depends(get_db),
):
    """
    Ingests a document for a specific chat session.
    Extracts text/layout, chunks it, embeds chunks, and indexes them in Qdrant
    under the chat_id partition.

    DB write (chat_files row) happens ONLY after successful Qdrant indexing,
    ensuring the two stores stay consistent. If Qdrant fails, no DB row is written.
    """
    try:
        # 1. Extract text and layout
        extraction_res = await extraction_service.extract_text_from_file(file)

        # 2. Chunk document layout-aware
        file_id = str(uuid.uuid4())
        chunker = LayoutAwareChunker()
        chunks = chunker.chunk_document(extraction_res, chat_id=chat_id, file_id=file_id)

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No readable content could be chunked from this file."
            )

        # 3. Generate embeddings
        chunk_texts = [chunk.content for chunk in chunks]
        embeddings = await EmbeddingService.embed_documents(chunk_texts)

        # 4. Save to Qdrant — DB write only happens if this succeeds
        vector_store = QdrantStore()
        await vector_store.upsert_chunks(chunks, embeddings)

        # 5. Record in PostgreSQL (after Qdrant success — atomic within request)
        await crud.upsert_chat(db, chat_id)
        await crud.create_file(
            db,
            chat_id=chat_id,
            file_id=file_id,
            file_name=extraction_res.file_name,
            chunk_count=len(chunks),
        )

        return {
            "success": True,
            "chat_id": chat_id,
            "file_id": file_id,
            "file_name": extraction_res.file_name,
            "total_chunks": len(chunks)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"File upload and indexing failed: {str(e)}"
        )


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