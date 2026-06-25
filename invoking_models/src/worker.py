"""
worker.py — Long-polling SQS worker for async file processing.

Run from invoking_models/src/:
    python worker.py

Flow per message:
    1. Receive message from SQS (long-poll, WaitTimeSeconds=20)
    2. Mark job as processing in DB
    3. Load file bytes from local storage
    4. Route to correct extraction service by MIME type
    5. Chunk → Embed → Upsert into Qdrant
    6. Write chat_files row to Postgres
    7. Mark job as completed in DB
    8. Delete message from SQS (only on full success)
    On any failure: mark job as failed with error message, leave message
    in-flight for SQS retry (visibility timeout will re-deliver it).
"""

import asyncio
import json
import logging
import sys
import os

# Ensure src/ is on the path when run directly
sys.path.insert(0, os.path.dirname(__file__))

from config import settings
from db.database import AsyncSessionLocal
from db import crud
from db.models import JobStatus
from services.storage_service import StorageService
from services.sqs_service import receive_messages, delete_message
from services.extraction_service import ExtractionService
from services.file_service import FileService
from rag.chunking import LayoutAwareChunker
from rag.embeddings import EmbeddingService
from rag.vector_store import QdrantStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("worker")

# MIME type → canonical file type label used for routing
_MIME_TO_TYPE = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "text/plain": "txt",
    "image/jpeg": "image",
    "image/png": "image",
    "image/webp": "image",
}


async def _process_message(message: dict) -> None:
    """Full processing pipeline for one SQS message."""
    body = json.loads(message["Body"])
    receipt_handle = message["ReceiptHandle"]

    job_id = body["job_id"]
    chat_id = body["chat_id"]
    file_id = body["file_id"]
    file_name = body["file_name"]
    file_type = body["file_type"]       # MIME type
    storage_path = body["storage_path"]

    logger.info("Processing job=%s file=%s mime=%s", job_id, file_name, file_type)

    async with AsyncSessionLocal() as db:
        # Mark as processing so UI can show progress
        await crud.update_job_status(db, job_id, JobStatus.PROCESSING)

    try:
        # 1. Load raw bytes from disk
        file_bytes = await asyncio.to_thread(StorageService.load, storage_path)

        # 2. Build a minimal UploadFile-like object for ExtractionService
        from io import BytesIO
        from fastapi import UploadFile
        upload_file = UploadFile(
            filename=file_name,
            content_type=file_type,
            file=BytesIO(file_bytes),
        )

        # 3. Extract text using the existing ExtractionService
        #    (reuses all PDF/DOCX/image/txt routing internally)
        extraction_service = ExtractionService()
        extraction_res = await extraction_service.extract_text_from_file(upload_file)

        if not extraction_res or not extraction_res.extracted_text.strip():
            raise ValueError("Extraction returned empty text.")

        # 4. Chunk
        chunker = LayoutAwareChunker()
        chunks = chunker.chunk_document(extraction_res, chat_id=chat_id, file_id=file_id)

        if not chunks:
            raise ValueError("No chunks produced from extraction output.")

        # 5. Embed
        chunk_texts = [chunk.content for chunk in chunks]
        embeddings = await EmbeddingService.embed_documents(chunk_texts)

        # 6. Upsert into Qdrant
        #    Idempotent: deterministic UUID5 per chunk — re-running same file
        #    overwrites existing vectors, never duplicates.
        vector_store = QdrantStore()
        await vector_store.upsert_chunks(chunks, embeddings)

        # 7. Write chat_files row to Postgres + mark job completed
        async with AsyncSessionLocal() as db:
            await crud.upsert_chat(db, chat_id)
            await crud.create_file(
                db,
                chat_id=chat_id,
                file_id=file_id,
                file_name=file_name,
                chunk_count=len(chunks),
            )
            await crud.update_job_status(db, job_id, JobStatus.COMPLETED)

        # 8. Clean up raw file from disk after successful processing
        await asyncio.to_thread(StorageService.delete, storage_path)

        # 9. Delete from SQS — only reached on full success
        await asyncio.to_thread(delete_message, receipt_handle)

        logger.info("Completed job=%s file=%s chunks=%d", job_id, file_name, len(chunks))

    except Exception as exc:
        logger.exception("Failed job=%s file=%s error=%s", job_id, file_name, exc)
        async with AsyncSessionLocal() as db:
            await crud.update_job_status(
                db, job_id, JobStatus.FAILED, error_message=str(exc)
            )
        # Do NOT delete the SQS message — let visibility timeout expire so it retries


async def run_worker() -> None:
    """Main loop: long-poll SQS and process messages one at a time."""
    logger.info("Worker started. Polling SQS queue: %s", settings.SQS_QUEUE_URL)

    while True:
        try:
            messages = await asyncio.to_thread(
                receive_messages, 1, 20  # 1 message, 20s long-poll
            )
            for message in messages:
                await _process_message(message)

        except KeyboardInterrupt:
            logger.info("Worker shutting down.")
            break
        except Exception as exc:
            # Unexpected poll-level error — log and continue
            logger.exception("Unexpected worker loop error: %s", exc)
            await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(run_worker())
