import logging
from io import BytesIO

import fitz
from PIL import Image

from config import settings
logger = logging.getLogger(__name__)

# Tunable thresholds — move to settings if you want these env-configurable
TEXT_MIN_CHARS = 50
MAX_PAGES = 200
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25MB


class PDFProcessingError(Exception):
    """Raised for any recoverable PDF processing failure (bad file, too large, encrypted, etc.)."""


class PDFService:


    @staticmethod
    def _open_pdf(file_bytes: bytes) -> fitz.Document:
        """Validates and opens a PDF. Raises PDFProcessingError on any invalid input."""
        if not file_bytes:
            raise PDFProcessingError("Empty file provided.")

        if len(file_bytes) > MAX_FILE_SIZE_BYTES:
            raise PDFProcessingError(
                f"File size {len(file_bytes)} bytes exceeds max allowed "
                f"{MAX_FILE_SIZE_BYTES} bytes."
            )

        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
        except Exception as exc:
            logger.warning("Failed to open PDF: %s", exc)
            raise PDFProcessingError("File is not a valid or readable PDF.") from exc

        if doc.is_encrypted:
            doc.close()
            raise PDFProcessingError("PDF is password-protected/encrypted.")

        if doc.page_count == 0:
            doc.close()
            raise PDFProcessingError("PDF contains no pages.")

        if doc.page_count > MAX_PAGES:
            doc.close()
            raise PDFProcessingError(
                f"PDF has {doc.page_count} pages, exceeding max allowed {MAX_PAGES}."
            )

        return doc

    @staticmethod
    def _render_page_png(page: fitz.Page, matrix: fitz.Matrix) -> bytes:
        """Renders a page to PNG bytes directly — no PIL round-trip needed."""
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        return pix.tobytes("png")


    @staticmethod
    def analyze_pdf(file_bytes: bytes) -> dict:
        doc = PDFService._open_pdf(file_bytes)
        try:
            total_pages = doc.page_count
            text_pages = 0
            image_heavy_pages = 0

            for page in doc:
                try:
                    text = page.get_text("text").strip()
                    images = page.get_images(full=True)
                except Exception as exc:
                    logger.warning("Failed to read page %s during analyze: %s", page.number, exc)
                    continue

                if len(text) > TEXT_MIN_CHARS:
                    text_pages += 1
                if images and len(text) < TEXT_MIN_CHARS:
                    image_heavy_pages += 1

            if text_pages == total_pages:
                pdf_type = "text_pdf"
            elif text_pages == 0:
                pdf_type = "scanned_pdf"
            else:
                pdf_type = "mixed_pdf"

            logger.info(
                "PDF analyzed: type=%s total_pages=%s text_pages=%s image_heavy_pages=%s",
                pdf_type, total_pages, text_pages, image_heavy_pages,
            )

            return {
                "pdf_type": pdf_type,
                "total_pages": total_pages,
                "text_pages": text_pages,
                "image_heavy_pages": image_heavy_pages,
            }
        finally:
            doc.close()

    @staticmethod
    def extract_text_pages(file_bytes: bytes) -> list[dict]:
        doc = PDFService._open_pdf(file_bytes)
        try:
            pages = []
            for index, page in enumerate(doc, start=1):
                try:
                    text = page.get_text("text").strip()
                except Exception as exc:
                    logger.warning("Failed to extract text on page %s: %s", index, exc)
                    text = ""
                pages.append({
                    "page_number": index,
                    "text": text,
                    "extraction_method": "direct_text_extraction",
                })
            return pages
        finally:
            doc.close()

    @staticmethod
    def convert_pdf_pages_to_images(file_bytes: bytes) -> list[dict]:
        doc = PDFService._open_pdf(file_bytes)
        from config import settings
        try:
            page_images = []
            zoom = settings.PDF_OCR_DPI / 72
            matrix = fitz.Matrix(zoom, zoom)

            for index, page in enumerate(doc, start=1):
                try:
                    png_bytes = PDFService._render_page_png(page, matrix)
                except Exception as exc:
                    logger.warning("Failed to render page %s to image: %s", index, exc)
                    continue
                page_images.append({
                    "page_number": index,
                    "image_bytes": png_bytes,
                    "mime_type": "image/png",
                })
            return page_images
        finally:
            doc.close()

    @staticmethod
    def extract_mixed_pdf_pages(file_bytes: bytes) -> tuple[list[dict], list[dict]]:
        """
        Returns:
            direct_text_pages: pages with sufficient extractable text
            ocr_pages: pages rendered to images for vision/OCR processing
        """
        doc = PDFService._open_pdf(file_bytes)
        try:
            direct_pages = []
            ocr_pages = []
            zoom = settings.PDF_OCR_DPI / 72
            matrix = fitz.Matrix(zoom, zoom)

            for index, page in enumerate(doc, start=1):
                try:
                    text = page.get_text("text").strip()
                except Exception as exc:
                    logger.warning("Failed to read text on page %s: %s", index, exc)
                    text = ""

                if len(text) > TEXT_MIN_CHARS:
                    direct_pages.append({
                        "page_number": index,
                        "text": text,
                        "extraction_method": "direct_text_extraction",
                    })
                else:
                    try:
                        png_bytes = PDFService._render_page_png(page, matrix)
                    except Exception as exc:
                        logger.warning("Failed to render page %s for OCR: %s", index, exc)
                        continue
                    ocr_pages.append({
                        "page_number": index,
                        "image_bytes": png_bytes,
                        "mime_type": "image/png",
                    })

            return direct_pages, ocr_pages
        finally:
            doc.close()