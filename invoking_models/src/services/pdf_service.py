import logging
import tempfile
import os
from io import BytesIO
from typing import List, Tuple

import fitz
import pdfplumber

from config import settings

logger = logging.getLogger(__name__)

TEXT_MIN_CHARS = 50
MAX_PAGES = 200
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25MB


class PDFProcessingError(Exception):
    """Raised for recoverable PDF processing failures."""


class PDFService:
    @staticmethod
    def _open_pdf(file_bytes: bytes) -> fitz.Document:
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
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        return pix.tobytes("png")

    @staticmethod
    def _detect_columns(
        page: fitz.Page,
        min_gutter_height_ratio: float = 0.6,
        min_gutter_width_pts: float = 8.0,
    ) -> List[float]:
        blocks = page.get_text("blocks")
        text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]

        page_width = page.rect.width
        page_height = page.rect.height

        if not text_blocks:
            return [0, page_width]

        full_width_threshold = 0.7 * page_width
        candidate_blocks = [
            b for b in text_blocks if (b[2] - b[0]) < full_width_threshold
        ]

        if candidate_blocks:
            text_blocks = candidate_blocks

        x_bins = 120
        y_bins = 60
        bin_w = page_width / x_bins
        bin_h = page_height / y_bins

        occupancy = [[False] * y_bins for _ in range(x_bins)]

        for block in text_blocks:
            x0, y0, x1, y1 = block[0], block[1], block[2], block[3]

            xb0 = max(0, int(x0 / bin_w))
            xb1 = min(int(x1 / bin_w), x_bins - 1)
            yb0 = max(0, int(y0 / bin_h))
            yb1 = min(int(y1 / bin_h), y_bins - 1)

            for xi in range(xb0, xb1 + 1):
                for yi in range(yb0, yb1 + 1):
                    occupancy[xi][yi] = True

        rows_with_text = [
            yi for yi in range(y_bins)
            if any(occupancy[xi][yi] for xi in range(x_bins))
        ]

        if not rows_with_text:
            return [0, page_width]

        body_top = min(rows_with_text)
        body_bottom = max(rows_with_text)
        body_row_count = body_bottom - body_top + 1

        min_gutter_bins = max(1, int(min_gutter_width_pts / bin_w))

        empty_x_bins = []

        for xi in range(x_bins):
            empty_rows = sum(
                1
                for yi in range(body_top, body_bottom + 1)
                if not occupancy[xi][yi]
            )

            if body_row_count > 0:
                empty_ratio = empty_rows / body_row_count
                if empty_ratio >= min_gutter_height_ratio:
                    empty_x_bins.append(xi)

        if not empty_x_bins:
            return [0, page_width]

        gutter_runs = []
        run_start = empty_x_bins[0]
        previous = empty_x_bins[0]

        for xi in empty_x_bins[1:]:
            if xi == previous + 1:
                previous = xi
            else:
                gutter_runs.append((run_start, previous))
                run_start = xi
                previous = xi

        gutter_runs.append((run_start, previous))

        gutter_runs = [
            run for run in gutter_runs
            if (run[1] - run[0] + 1) >= min_gutter_bins
        ]

        gutter_runs = [
            run for run in gutter_runs
            if run[0] > 2 and run[1] < x_bins - 3
        ]

        if not gutter_runs:
            return [0, page_width]

        boundaries = [0]

        for run in gutter_runs:
            gutter_center = ((run[0] + run[1]) / 2 + 0.5) * bin_w
            boundaries.append(gutter_center)

        boundaries.append(page_width)

        return boundaries

    @staticmethod
    def _assign_column_index(bbox: tuple, column_boundaries: List[float]) -> int:
        x_center = (bbox[0] + bbox[2]) / 2

        for i in range(len(column_boundaries) - 1):
            if column_boundaries[i] <= x_center < column_boundaries[i + 1]:
                return i

        return max(0, len(column_boundaries) - 2)

    @staticmethod
    def _bbox_overlaps(b1: tuple, b2: tuple, threshold: float = 0.5) -> bool:
        x0 = max(b1[0], b2[0])
        y0 = max(b1[1], b2[1])
        x1 = min(b1[2], b2[2])
        y1 = min(b1[3], b2[3])

        if x1 <= x0 or y1 <= y0:
            return False

        overlap_area = (x1 - x0) * (y1 - y0)
        b1_area = max((b1[2] - b1[0]) * (b1[3] - b1[1]), 1)

        return (overlap_area / b1_area) > threshold

    @staticmethod
    def _table_to_markdown(data: list) -> str:
        if not data:
            return ""

        header = data[0] or []
        md_lines = []

        md_lines.append("| " + " | ".join(str(cell or "") for cell in header) + " |")
        md_lines.append("|" + "---|" * len(header))

        for row in data[1:]:
            md_lines.append("| " + " | ".join(str(cell or "") for cell in row) + " |")

        return "\n".join(md_lines)

    @staticmethod
    def _extract_tables_from_bytes(file_bytes: bytes, page_index: int) -> List[dict]:
        tables = []

        try:
            with pdfplumber.open(BytesIO(file_bytes)) as pdf:
                plumber_page = pdf.pages[page_index]
                found_tables = plumber_page.find_tables()

                for table in found_tables:
                    data = table.extract()

                    if not data:
                        continue

                    markdown = PDFService._table_to_markdown(data)

                    if not markdown.strip():
                        continue

                    tables.append({
                        "type": "table",
                        "content": f"\n[TABLE]\n{markdown}\n[/TABLE]\n",
                        "bbox": table.bbox,
                    })

        except Exception as exc:
            logger.warning(
                "Table extraction failed on page %s: %s",
                page_index + 1,
                exc,
            )

        return tables

    @staticmethod
    def _extract_layout_text_for_page(
        page: fitz.Page,
        table_bboxes: List[tuple],
    ) -> str:
        column_boundaries = PDFService._detect_columns(page)
        is_multi_column = len(column_boundaries) > 2

        page_width = page.rect.width
        full_width_threshold = 0.7 * page_width

        raw_blocks = page.get_text("blocks")

        text_blocks = []
        spanning_blocks = []

        for block in raw_blocks:
            x0, y0, x1, y1, text, block_no, block_type = block

            if block_type != 0 or not text.strip():
                continue

            bbox = (x0, y0, x1, y1)

            if any(PDFService._bbox_overlaps(bbox, tb) for tb in table_bboxes):
                continue

            column_index = PDFService._assign_column_index(bbox, column_boundaries)

            item = {
                "type": "text",
                "content": text.strip(),
                "bbox": bbox,
                "column_index": column_index,
                "spans_all_columns": (
                    is_multi_column and (x1 - x0) >= full_width_threshold
                ),
            }

            if item["spans_all_columns"]:
                spanning_blocks.append(item)
            else:
                text_blocks.append(item)

        spanning_blocks = sorted(spanning_blocks, key=lambda b: b["bbox"][1])

        separators_y = [b["bbox"][1] for b in spanning_blocks]

        bands = []
        previous_y = float("-inf")

        for separator_y in separators_y + [float("inf")]:
            band_blocks = [
                b for b in text_blocks
                if previous_y <= b["bbox"][1] < separator_y
            ]

            bands.append(band_blocks)
            previous_y = separator_y

        parts = []

        for index, band_blocks in enumerate(bands):
            sorted_band = sorted(
                band_blocks,
                key=lambda b: (b["column_index"], b["bbox"][1], b["bbox"][0]),
            )

            for block in sorted_band:
                parts.append(block["content"])

            if index < len(spanning_blocks):
                parts.append(spanning_blocks[index]["content"])

        return "\n".join(parts).strip()

    @staticmethod
    def _extract_layout_text_with_tables_for_page(
        file_bytes: bytes,
        page: fitz.Page,
        page_index: int,
    ) -> str:
        table_blocks = PDFService._extract_tables_from_bytes(file_bytes, page_index)
        table_bboxes = [table["bbox"] for table in table_blocks]

        text = PDFService._extract_layout_text_for_page(page, table_bboxes)

        combined_blocks = []

        if text.strip():
            combined_blocks.append({
                "type": "text",
                "content": text,
                "bbox": (0, 0, page.rect.width, page.rect.height),
                "column_index": 0,
            })

        for table in table_blocks:
            table["column_index"] = PDFService._assign_column_index(
                table["bbox"],
                PDFService._detect_columns(page),
            )
            combined_blocks.append(table)

        combined_blocks = sorted(
            combined_blocks,
            key=lambda b: (
                b.get("column_index", 0),
                b["bbox"][1],
                b["bbox"][0],
            ),
        )

        return "\n".join(block["content"] for block in combined_blocks).strip()

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
                    logger.warning(
                        "Failed to read page %s during analyze: %s",
                        page.number + 1,
                        exc,
                    )
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
                pdf_type,
                total_pages,
                text_pages,
                image_heavy_pages,
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
                    text = PDFService._extract_layout_text_with_tables_for_page(
                        file_bytes=file_bytes,
                        page=page,
                        page_index=index - 1,
                    )
                except Exception as exc:
                    logger.warning(
                        "Layout extraction failed on page %s, fallback to normal text: %s",
                        index,
                        exc,
                    )
                    try:
                        text = page.get_text("text").strip()
                    except Exception:
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

        try:
            page_images = []

            zoom = settings.PDF_OCR_DPI / 72
            matrix = fitz.Matrix(zoom, zoom)

            for index, page in enumerate(doc, start=1):
                try:
                    png_bytes = PDFService._render_page_png(page, matrix)
                except Exception as exc:
                    logger.warning(
                        "Failed to render page %s to image: %s",
                        index,
                        exc,
                    )
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
        doc = PDFService._open_pdf(file_bytes)

        try:
            direct_pages = []
            ocr_pages = []

            zoom = settings.PDF_OCR_DPI / 72
            matrix = fitz.Matrix(zoom, zoom)

            for index, page in enumerate(doc, start=1):
                try:
                    plain_text = page.get_text("text").strip()
                except Exception as exc:
                    logger.warning(
                        "Failed to read text on page %s: %s",
                        index,
                        exc,
                    )
                    plain_text = ""

                if len(plain_text) > TEXT_MIN_CHARS:
                    try:
                        layout_text = PDFService._extract_layout_text_with_tables_for_page(
                            file_bytes=file_bytes,
                            page=page,
                            page_index=index - 1,
                        )
                    except Exception as exc:
                        logger.warning(
                            "Layout extraction failed on mixed page %s: %s",
                            index,
                            exc,
                        )
                        layout_text = plain_text

                    direct_pages.append({
                        "page_number": index,
                        "text": layout_text,
                        "extraction_method": "direct_text_extraction",
                    })

                else:
                    try:
                        png_bytes = PDFService._render_page_png(page, matrix)
                    except Exception as exc:
                        logger.warning(
                            "Failed to render page %s for OCR: %s",
                            index,
                            exc,
                        )
                        continue

                    ocr_pages.append({
                        "page_number": index,
                        "image_bytes": png_bytes,
                        "mime_type": "image/png",
                    })

            return direct_pages, ocr_pages

        finally:
            doc.close()