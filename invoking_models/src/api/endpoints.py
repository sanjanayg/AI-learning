import uuid
import logging
import asyncio
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
    CacheMetadata,
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

# Semantic cache services
from cache.cache_config import cache_settings
from cache.query_normalizer import QueryNormalizer
from cache.cache_eligibility import CacheEligibilityService
from cache.cache_validator import CacheValidator
from cache.kb_version_tracker import KBVersionTracker
from cache.semantic_cache import SemanticCacheService
from cache.cache_metrics import metrics_service

logger = logging.getLogger(__name__)

extraction_router = APIRouter(tags=["Extraction"])
chunking_router = APIRouter(tags=["Chunking"])
chat_router = APIRouter(tags=["Chat RAG"], prefix="/chat")
chats_router = APIRouter(tags=["Chat Sessions"], prefix="/chats")
cache_router = APIRouter(tags=["Cache Services"], prefix="/cache")

_cache_service: SemanticCacheService | None = None

def get_cache_service() -> SemanticCacheService:
    global _cache_service
    if _cache_service is None:
        _cache_service = SemanticCacheService()
    return _cache_service

eligibility_service = CacheEligibilityService()
cache_validator = CacheValidator()


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
    cache_service = get_cache_service()
    normalized_query = QueryNormalizer.normalize(request.query)
    is_eligible, eligibility_reason = eligibility_service.is_eligible(normalized_query)
    
    # Track concurrency / caching variables
    is_creator = False
    kb_version = 0
    selected_llm_model = None
    intent = "general_qa"
    entities = {}
    query_embedding = []
    
    try:
        if cache_settings.CACHE_ENABLED and is_eligible:
            kb_version = await KBVersionTracker.get_version(db, chat_id)
            selected_llm_model = await llm_service.select_model(request.intelligence, request.query)
            intent, entities = await cache_validator.extract_intent_and_entities(normalized_query)
            query_embedding = await EmbeddingService.embed_query(normalized_query)
            
            event, is_creator = await cache_service.get_or_create_inflight_event(chat_id, normalized_query)
            
            if not is_creator:
                logger.info("Concurrency: Waiting for concurrent RAG generation for query '%s'...", normalized_query)
                await event.wait()
                # Try cache lookup after event is set
                cache_res = await cache_service.lookup(
                    query=normalized_query,
                    query_embedding=query_embedding,
                    tenant_id=chat_id,
                    kb_version=kb_version,
                    prompt_version=cache_settings.PROMPT_VERSION,
                    embedding_model=cache_settings.EMBEDDING_MODEL,
                    llm_model=selected_llm_model,
                    intent=intent,
                    entities=entities
                )
                if cache_res.hit and cache_res.entry:
                    # Append assistant message to DB
                    await crud.upsert_chat(db, chat_id)
                    await crud.append_message(
                        db,
                        chat_id=chat_id,
                        role="assistant",
                        content=cache_res.entry.answer,
                        citations=[],
                        tokens_used=0
                    )
                    return ChatQueryResponse(
                        success=True,
                        answer=cache_res.entry.answer,
                        citations=[],
                        intelligence=request.intelligence,
                        model_used=selected_llm_model,
                        tokens_used=0,
                        cache_metadata=CacheMetadata(
                            response_source="CACHE",
                            cache_hit=True,
                            similarity_score=cache_res.similarity_score,
                            cache_id=cache_res.entry.id,
                            lookup_time_ms=0.0
                        )
                    )
                # If still a miss after wait, fall through as a creator
                is_creator = True
            
            if is_creator:
                lookup_start = asyncio.get_event_loop().time()
                cache_res = await cache_service.lookup(
                    query=normalized_query,
                    query_embedding=query_embedding,
                    tenant_id=chat_id,
                    kb_version=kb_version,
                    prompt_version=cache_settings.PROMPT_VERSION,
                    embedding_model=cache_settings.EMBEDDING_MODEL,
                    llm_model=selected_llm_model,
                    intent=intent,
                    entities=entities
                )
                lookup_time_ms = (asyncio.get_event_loop().time() - lookup_start) * 1000.0
                
                if cache_res.hit and cache_res.entry:
                    # Release wait event since we got hit
                    await cache_service.release_inflight_event(chat_id, normalized_query)
                    is_creator = False
                    
                    # Append assistant message to DB
                    await crud.upsert_chat(db, chat_id)
                    await crud.append_message(
                        db,
                        chat_id=chat_id,
                        role="assistant",
                        content=cache_res.entry.answer,
                        citations=[],
                        tokens_used=0
                    )
                    return ChatQueryResponse(
                        success=True,
                        answer=cache_res.entry.answer,
                        citations=[],
                        intelligence=request.intelligence,
                        model_used=selected_llm_model,
                        tokens_used=0,
                        cache_metadata=CacheMetadata(
                            response_source="CACHE",
                            cache_hit=True,
                            similarity_score=cache_res.similarity_score,
                            cache_id=cache_res.entry.id,
                            lookup_time_ms=lookup_time_ms
                        )
                    )
        
        if not is_eligible:
            metrics_service.record_miss("not_cacheable")
            
        # 1. Guardrail: Input Safety Validation
        RAGGuardrails.validate_query(request.query)
        
        rag_start = asyncio.get_event_loop().time()
        
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
        raw_answer = await llm_service.generate_grounded_response(
            request.query,
            budget_chunks,
            formatted_history,
            model,
            mode=request.mode,
        )
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
        await crud.upsert_chat(db, chat_id)
        await crud.append_message(
            db,
            chat_id=chat_id,
            role="assistant",
            content=final_answer,
            citations=[c.model_dump() for c in citations],
            tokens_used=tokens_used,
        )

        rag_time_ms = (asyncio.get_event_loop().time() - rag_start) * 1000.0
        metrics_service.record_rag_latency(rag_time_ms)
        
        # Store in cache if eligible
        if (
            cache_settings.CACHE_ENABLED 
            and is_eligible 
            and final_answer 
            and "I'm sorry, but the uploaded documents do not contain the information required" not in final_answer
        ):
            # Non-blocking async store
            asyncio.create_task(
                cache_service.store(
                    query=normalized_query,
                    query_embedding=query_embedding,
                    answer=final_answer,
                    tenant_id=chat_id,
                    intent=intent,
                    entities=entities,
                    kb_version=kb_version,
                    llm_model=model
                )
            )

        return ChatQueryResponse(
            success=True,
            answer=final_answer,
            citations=citations,
            intelligence=request.intelligence,
            model_used=model,
            tokens_used=tokens_used,
            cache_metadata=CacheMetadata(
                response_source="LLM",
                cache_hit=False
            )
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Chat query failed: {str(e)}"
        )
    finally:
        # Guarantee release of concurrent requests waiting
        if is_creator:
            await cache_service.release_inflight_event(chat_id, normalized_query)

    

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


@cache_router.get("/stats")
async def get_cache_stats():
    """
    Exposes semantic cache stats: hits, misses, hit ratio, latencies, insertion count, etc.
    """
    return metrics_service.get_stats()


@cache_router.post("/clear")
async def clear_cache():
    metrics_service.clear()
    cache_service = get_cache_service()
    try:
        await asyncio.to_thread(
            cache_service.client.delete,
            collection_name=cache_service.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter()
            )
        )
        return {"success": True, "detail": "Cache and metrics cleared successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear cache collection: {str(e)}"
        )