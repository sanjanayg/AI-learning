import asyncio
from fastapi import UploadFile, HTTPException
from schemas import ExtractTextResponse
from services.file_service import FileService
from services.pdf_service import PDFService
from services.llm_service import LLMService

class ExtractionService:

    """
    Image → Base64 → LLM OCR → Text

    Text PDF → Direct PDF text extraction → Text

    Scanned/Image PDF → Convert pages to images → LLM OCR page by page → Text
    """
    def __init__(self):
        self.llm_service = LLMService()

    async def extract_text_from_file(self, file: UploadFile) -> ExtractTextResponse:
        """
        Image → Base64 → LLM OCR → Text

        Text PDF → Direct PDF text extraction → Text

        Scanned/Image PDF → Convert pages to images → LLM OCR page by page → Text
        """
        # 1. Read and validate raw bytes asynchronously
        file_bytes, mime_type = await FileService.read_and_validate(file)

        # 2. Handle standard image files
        if mime_type in FileService.ALLOWED_IMAGE_TYPES:
            base64_image = FileService.encode_to_base64(file_bytes)
            
            # Offload heavy synchronous LLM call to a thread pool
            extracted_text = await asyncio.to_thread(
                self.llm_service.extract_text_from_image,
                base64_image=base64_image,
                mime_type=mime_type
            )

            return ExtractTextResponse(
                success=True,
                file_type="image",
                extraction_method="llm_ocr",
                extracted_text=extracted_text
            )

        # 3. Handle PDF files
        if mime_type == FileService.PDF_TYPE:
            pdf_analysis = PDFService.analyze_pdf(file_bytes)
            pdf_type = pdf_analysis["pdf_type"]

            # Native digital text PDF path
            if pdf_type == "text_pdf":
                extracted_text = PDFService.extract_text_from_pdf(file_bytes)
                return ExtractTextResponse(
                    success=True,
                    file_type="pdf",
                    extraction_method="direct_text_extraction",
                    extracted_text=extracted_text
                )

            # Scanned/Image PDF path (Optimized with Concurrency)
            page_images = PDFService.convert_pdf_to_images(file_bytes)
            
            # Fire all LLM API page extraction calls concurrently
            tasks = [
                self._process_single_pdf_page(page) 
                for page in page_images
            ]
            extracted_pages = await asyncio.gather(*tasks)

            return ExtractTextResponse(
                success=True,
                file_type="pdf",
                extraction_method=f"{pdf_type}_llm_ocr",
                extracted_text="\n".join(extracted_pages).strip()
            )
            # return ExtractTextResponse(
            #     success=True,
            #     file_type="pdf",
            #     extraction_method=f"{pdf_type}_llm_ocr",
            #     extracted_text=extracted_pages
            # )

        # 4. Fail-safe type checking
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type"
        )

    async def _process_single_pdf_page(self, page: dict) -> str:
        """Helper method to isolate and async-wrap individual page tasks."""
        base64_image = FileService.encode_to_base64(page["image_bytes"])
        
        page_text = await asyncio.to_thread(
            self.llm_service.extract_text_from_image,
            base64_image=base64_image,
            mime_type=page["mime_type"]
        )
        return f"\n--- Page {page['page_no']} ---\n{page_text}"
