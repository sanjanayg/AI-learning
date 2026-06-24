import asyncio
import logging

from fastapi import UploadFile, HTTPException
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import settings
from schemas import ExtractTextResponse, ExtractedPage
from services.file_service import FileService
from services.pdf_service import PDFService, PDFProcessingError
from services.docx_service import DocxService, DocxProcessingError
from services.txt_service import TxtService, TxtProcessingError
from services.llm_service import LLMService

logger = logging.getLogger(__name__)

# Exceptions that mean "bad input" — map to 400, not 500
_INPUT_ERRORS = (PDFProcessingError, DocxProcessingError, TxtProcessingError)

OCR_PAGE_TIMEOUT_SECONDS = getattr(settings, "PDF_OCR_PAGE_TIMEOUT_SECONDS", 60)


class ExtractionService:
    """
    Production RAG extraction flow:

    Image:
        Image → Base64 → LLM OCR → Text

    Text PDF:
        PDF → Direct text extraction → Text

    Scanned PDF:
        PDF → Page images → LLM OCR page-by-page → Text

    Mixed PDF:
        Text pages → direct extraction
        Scanned pages → OCR
        Merge in page order

    DOCX:
        Paragraphs + tables → Text

    TXT:
        Direct decode → Text
    """

    def __init__(self):
        self.llm_service = LLMService()
        self.semaphore = asyncio.Semaphore(settings.PDF_OCR_CONCURRENCY)

    async def extract_text_from_file(self, file: UploadFile) -> ExtractTextResponse:
        file_bytes, mime_type = await FileService.read_and_validate(file)
        file_name = file.filename or "uploaded_file"

        logger.info("Starting extraction: file=%s mime_type=%s size=%s",
                    file_name, mime_type, len(file_bytes))

        try:
            if mime_type in FileService.ALLOWED_IMAGE_TYPES:
                return await self._extract_from_image(file_name, file_bytes, mime_type)

            if mime_type == FileService.PDF_TYPE:
                return await self._extract_from_pdf(file_name, file_bytes)

            if mime_type in FileService.ALLOWED_DOCX_TYPES:
                return await self._extract_from_docx(file_name, file_bytes)

            if mime_type == FileService.TXT_TYPE:
                return await self._extract_from_txt(file_name, file_bytes)

            raise HTTPException(status_code=400, detail="Unsupported file type")

        except HTTPException:
            raise

        except _INPUT_ERRORS as exc:
            logger.warning("Extraction rejected bad input: file=%s error=%s", file_name, exc)
            raise HTTPException(status_code=400, detail=str(exc))

        except Exception as exc:
            logger.exception("Extraction failed unexpectedly: file=%s", file_name)
            raise HTTPException(
                status_code=500,
                detail="Text extraction failed due to an internal error.",
            )

    async def _extract_from_image(
        self,
        file_name: str,
        file_bytes: bytes,
        mime_type: str,
    ) -> ExtractTextResponse:
        base64_image = FileService.encode_to_base64(file_bytes)
        extracted_text = await self._ocr_with_retry(base64_image, mime_type)

        return ExtractTextResponse(
            success=True,
            file_name=file_name,
            file_type="image",
            extraction_method="llm_ocr",
            extracted_text=extracted_text.strip(),
            pages=[
                ExtractedPage(page_number=1, text=extracted_text.strip(), extraction_method="llm_ocr")
            ],
        )

    async def _extract_from_pdf(
        self,
        file_name: str,
        file_bytes: bytes,
    ) -> ExtractTextResponse:
        pdf_analysis = PDFService.analyze_pdf(file_bytes)
        pdf_type = pdf_analysis["pdf_type"]

        logger.info("PDF analyzed: file=%s pdf_type=%s total_pages=%s",
                    file_name, pdf_type, pdf_analysis["total_pages"])

        if pdf_type == "text_pdf":
            raw_pages = await asyncio.to_thread(PDFService.extract_text_pages, file_bytes)
            pages = [ExtractedPage(**page) for page in raw_pages]
            final_text = self._merge_pages(pages)

            return ExtractTextResponse(
                success=True,
                file_name=file_name,
                file_type="pdf",
                extraction_method="direct_text_extraction",
                extracted_text=final_text,
                pages=pages,
            )

        if pdf_type == "scanned_pdf":
            page_images = await asyncio.to_thread(PDFService.convert_pdf_pages_to_images, file_bytes)
            pages = await self._ocr_pdf_pages(page_images)
            final_text = self._merge_pages(pages)

            return ExtractTextResponse(
                success=True,
                file_name=file_name,
                file_type="pdf",
                extraction_method="scanned_pdf_llm_ocr",
                extracted_text=final_text,
                pages=pages,
            )

        direct_pages, ocr_page_images = await asyncio.to_thread(
            PDFService.extract_mixed_pdf_pages, file_bytes
        )

        direct_extracted_pages = [ExtractedPage(**page) for page in direct_pages]
        ocr_extracted_pages = await self._ocr_pdf_pages(ocr_page_images)

        all_pages = direct_extracted_pages + ocr_extracted_pages
        all_pages.sort(key=lambda page: page.page_number)

        final_text = self._merge_pages(all_pages)

        return ExtractTextResponse(
            success=True,
            file_name=file_name,
            file_type="pdf",
            extraction_method="mixed_pdf_hybrid_extraction",
            extracted_text=final_text,
            pages=all_pages,
        )

    async def _ocr_pdf_pages(self, page_images: list[dict]) -> list[ExtractedPage]:
        """
        Runs OCR across all page images concurrently (bounded by semaphore).
        Failures are isolated per-page: a failed page is returned with empty text
        and a distinct extraction_method, so the rest of the document still succeeds.
        """
        tasks = [self._process_single_pdf_page(page) for page in page_images]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        pages: list[ExtractedPage] = []
        failed_count = 0

        for page_image, result in zip(page_images, results):
            if isinstance(result, Exception):
                failed_count += 1
                logger.warning(
                    "OCR failed for page=%s: %s", page_image["page_number"], result
                )
                pages.append(
                    ExtractedPage(
                        page_number=page_image["page_number"],
                        text="",
                        extraction_method="llm_ocr_failed",
                    )
                )
            else:
                pages.append(result)

        if failed_count:
            logger.warning("%s/%s pages failed OCR", failed_count, len(page_images))

        return pages

    async def _process_single_pdf_page(self, page: dict) -> ExtractedPage:
        async with self.semaphore:
            base64_image = FileService.encode_to_base64(page["image_bytes"])
            text = await self._ocr_with_retry(base64_image, page["mime_type"])

            return ExtractedPage(
                page_number=page["page_number"],
                text=text.strip(),
                extraction_method="llm_ocr",
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type((asyncio.TimeoutError, ConnectionError, TimeoutError)),
        reraise=True,
    )
    async def _ocr_with_retry(self, base64_image: str, mime_type: str) -> str:
        """Wraps the LLM OCR call with a timeout and retry/backoff for transient failures."""
        return await asyncio.wait_for(
            asyncio.to_thread(self.llm_service.extract_text_from_image, base64_image, mime_type),
            timeout=OCR_PAGE_TIMEOUT_SECONDS,
        )

    async def _extract_from_docx(
        self,
        file_name: str,
        file_bytes: bytes,
    ) -> ExtractTextResponse:
        extracted_text = await asyncio.to_thread(DocxService.extract_text, file_bytes)

        return ExtractTextResponse(
            success=True,
            file_name=file_name,
            file_type="document",
            extraction_method="direct_docx_extraction",
            extracted_text=extracted_text,
            pages=[
                ExtractedPage(page_number=1, text=extracted_text, extraction_method="direct_docx_extraction")
            ],
        )

    async def _extract_from_txt(
        self,
        file_name: str,
        file_bytes: bytes,
    ) -> ExtractTextResponse:
        extracted_text = await asyncio.to_thread(TxtService.extract_text, file_bytes)

        return ExtractTextResponse(
            success=True,
            file_name=file_name,
            file_type="text",
            extraction_method="direct_txt_extraction",
            extracted_text=extracted_text,
            pages=[
                ExtractedPage(page_number=1, text=extracted_text, extraction_method="direct_txt_extraction")
            ],
        )

    @staticmethod
    def _merge_pages(pages: list[ExtractedPage]) -> str:
        return "\n\n".join(
            f"[Page {page.page_number}]\n{page.text}"
            for page in pages
            if page.text.strip()
        ).strip()