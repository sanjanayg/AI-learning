import uuid
import logging
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
)
from services.extraction_service import ExtractionService
from services.chunking_service import RAGPipelineService
from services.llm_service import LLMService
from services.summary_service import ChatSummaryService
from services.report_service import ReportService
from rag.chunking import LayoutAwareChunker
from rag.embeddings import EmbeddingService
from rag.vector_store import QdrantStore
from rag.retriever import RAGRetriever
from rag.guardrails import RAGGuardrails
from db.database import get_db
from db import crud
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

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
            total_tokens_used=c.total_tokens_used,
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
        tokens_used=request.tokens_used,
    )
    return {"success": True}


# ── Chat RAG Endpoints (/chat) ────────────────────────────────────────────────
@chat_router.post("/{chat_id}/upload")
async def upload_chat_files(
    chat_id: str,
    files: list[UploadFile] = File(...),
    extraction_service: ExtractionService = Depends(get_extraction_service),
    db: AsyncSession = Depends(get_db),
):
    results = []

    try:
        existing_files = await crud.list_files(db, chat_id)
        existing_names = {f.file_name.lower() for f in existing_files}

        for file in files:
            if file.filename.lower() in existing_names:
                results.append({
                    "file_name": file.filename,
                    "success": False,
                    "error": "File has already been chunked."
                })
                continue

            extraction_res = await extraction_service.extract_text_from_file(file)

            file_id = str(uuid.uuid4())
            chunker = LayoutAwareChunker()
            chunks = chunker.chunk_document(
                extraction_res,
                chat_id=chat_id,
                file_id=file_id
            )

            if not chunks:
                results.append({
                    "file_name": file.filename,
                    "success": False,
                    "error": "No readable content could be chunked."
                })
                continue

            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = await EmbeddingService.embed_documents(chunk_texts)

            vector_store = QdrantStore()
            await vector_store.upsert_chunks(chunks, embeddings)

            await crud.upsert_chat(db, chat_id)
            await crud.create_file(
                db,
                chat_id=chat_id,
                file_id=file_id,
                file_name=extraction_res.file_name,
                chunk_count=len(chunks),
            )

            existing_names.add(file.filename.lower())

            results.append({
                "file_name": extraction_res.file_name,
                "success": True,
                "file_id": file_id,
                "total_chunks": len(chunks)
            })

        return {
            "success": True,
            "chat_id": chat_id,
            "files": results
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"File upload and indexing failed: {str(e)}"
        )

@chat_router.post("/{chat_id}/query", response_model=ChatQueryResponse)
async def query_chat_session(
    chat_id: str,
    request: ChatQueryRequest,
    llm_service: LLMService = Depends(get_llm_service),
    db: AsyncSession = Depends(get_db)
):
    """
    Queries a specific chat session with strict context isolation.
    Applies input injection checks, retrieves relevant chunks, enforces token limits,
    synthesizes a grounded response, and cleans up hallucinated citations.
    """
    try:
        # 1. Guardrail: Input Safety Validation
        RAGGuardrails.validate_query(request.query)
        history = await crud.get_recent_history(db, chat_id=chat_id, limit=3)
        formatted_history = [
            {"role": msg.role, "content": msg.content} for msg in history
        ]
        standalone_query = await llm_service.rewrite_query(
                history=formatted_history,
                query=request.query,
            )
        # 2. Retrieve relevant chunks strictly filtered by chat_id
        retriever = RAGRetriever()
        retrieved_chunks = await retriever.retrieve_relevant_chunks(
            chat_id=chat_id,
            query=standalone_query,
            limit=5,
            raw_query=request.query,   # pass original so extract_query_ids sees literal IDs
        )
        # 3. Guardrail: Context Token Budgeting (truncates context if it exceeds budget)
        budget_chunks = RAGGuardrails.enforce_token_budget(retrieved_chunks, max_tokens=6000)

        # 4. Generate grounded response from LLM
        model = await llm_service.select_model(request.intelligence, request.query)
        raw_answer = await llm_service.generate_grounded_response(request.query, budget_chunks, formatted_history, model)
        tokens_used = raw_answer.get("total_tokens", 0)

        # 5. Guardrail: Validate generated citations and strip hallucinated ones
        clean_answer = RAGGuardrails.validate_and_clean_citations(raw_answer["response"], budget_chunks)

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

        # 8. Persist assistant message with token count directly from the endpoint
        #    This is the authoritative write — Streamlit's fire-and-forget persist_message
        #    will hit the /chats/{chat_id}/messages endpoint which defaults tokens_used=0,
        #    so we write here first with the real count.
        await crud.upsert_chat(db, chat_id)
        await crud.append_message(
            db,
            chat_id=chat_id,
            role="assistant",
            content=final_answer,
            citations=[c.model_dump() for c in citations],
            tokens_used=tokens_used,
        )

        return ChatQueryResponse(
            success=True,
            answer=final_answer,
            citations=citations,
            intelligence=request.intelligence,
            model_used=model,
            tokens_used=tokens_used,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Chat query failed: {str(e)}"
        )
    

@chat_router.post("/{chat_id}/summary")
async def generate_chat_summary(
    chat_id: str,
    llm_service: LLMService = Depends(get_llm_service),
    db: AsyncSession = Depends(get_db),
):
    """
    Generate (or refresh) a rolling Chat Summary Report and return it as a
    downloadable PDF.

    Algorithm
    ---------
    1. Fetch the chat's display name for the PDF header.
    2. Look up the existing rolling summary in `chat_summaries`.
    3a. First time  → summarize ALL messages (with map-reduce if too long).
    3b. Returning   → fetch only messages AFTER last_message_id and ask the
                      LLM to update the existing summary incrementally.
    4. Upsert the updated summary + latest message_id into `chat_summaries`.
    5. Render a professional PDF via ReportService and stream it to the client.
    """
    logger.info("Chat summary report requested for chat_id='%s'", chat_id)

    try:
        # ── 1. Resolve chat name ───────────────────────────────────────────────
        chats = await crud.list_chats(db)
        chat_obj = next((c for c in chats if c.id == chat_id), None)
        chat_name = chat_obj.chat_name if chat_obj else chat_id

        # ── 2. Fetch existing rolling summary ─────────────────────────────────
        existing_row = await crud.get_chat_summary(db, chat_id)

        summary_service = ChatSummaryService()

        if existing_row is None or not existing_row.summary:
            # ── 3a. First-time: summarize full conversation ────────────────────
            logger.info("No existing summary for '%s' — running full summarization.", chat_id)
            all_messages = await crud.list_messages(db, chat_id)

            if not all_messages:
                raise HTTPException(
                    status_code=422,
                    detail="This chat has no messages yet. Send some messages before generating a summary.",
                )

            summary_json = await summary_service.summarize_full_conversation(all_messages)
            last_message_id = str(all_messages[-1].id)

        else:
            # ── 3b. Rolling update: only new messages ──────────────────────────
            logger.info(
                "Existing summary found for '%s' (last_message_id=%s) — fetching new messages.",
                chat_id,
                existing_row.last_message_id,
            )
            new_messages = await crud.get_messages_after(
                db,
                chat_id=chat_id,
                after_message_id=existing_row.last_message_id,
            )

            if not new_messages:
                # No new messages since last summary — serve the cached version
                logger.info("No new messages — serving cached summary for '%s'.", chat_id)
                summary_json = existing_row.summary
                last_message_id = existing_row.last_message_id
            else:
                logger.info(
                    "%d new messages found for '%s' — updating rolling summary.",
                    len(new_messages),
                    chat_id,
                )
                summary_json = await summary_service.update_summary_with_new_messages(
                    existing_summary=existing_row.summary,
                    new_messages=new_messages,
                )
                # Resolve the last message in the full list (not just new_messages)
                all_messages = await crud.list_messages(db, chat_id)
                last_message_id = str(all_messages[-1].id) if all_messages else existing_row.last_message_id

        # ── 4. Persist updated summary ─────────────────────────────────────────
        await crud.upsert_chat_summary(
            db,
            chat_id=chat_id,
            summary=summary_json,
            last_message_id=last_message_id,
        )

        # ── 5. Generate PDF and return as downloadable file ────────────────────
        pdf_buffer = ReportService.generate_pdf(
            chat_name=chat_name,
            summary_json=summary_json,
        )

        safe_name = chat_name.replace(" ", "-").lower()
        filename = f"chat-summary-{safe_name}.pdf"

        return StreamingResponse(
            pdf_buffer,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
            },
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Chat summary generation failed for chat_id='%s': %s", chat_id, exc)
        raise HTTPException(
            status_code=500,
            detail=f"Chat summary generation failed: {str(exc)}",
        )