"""
Layout-aware PDF extractor.
Handles: single-column, multi-column (2+), tables, and images.

Dependencies:
    pip install pymupdf pdfplumber --break-system-packages
"""
#### python pdf_extractor.py test_pdfs/test_two_column.pdf 
# python pdf_extractor.py D:/input_data_files_RAG/test_three_column_newsletter.pdf
import fitz  # PyMuPDF
import pdfplumber
from dataclasses import dataclass, field
from typing import List, Optional
import os


# ---------------------------------------------------------------------------
# Data model for extracted content
# ---------------------------------------------------------------------------

@dataclass
class ExtractedBlock:
    page_num: int
    block_type: str          # "text" | "table" | "image"
    content: str              # text or markdown-table string or image path
    bbox: tuple                # (x0, y0, x1, y1)
    column_index: int = 0     # which column this block belongs to (0 = left, 1 = right, ...)
    spans_all_columns: bool = False  # True for mastheads/headers/footers that cross every column


@dataclass
class PageExtraction:
    page_num: int
    num_columns: int
    blocks: List[ExtractedBlock] = field(default_factory=list)

    def get_reading_order_text(self) -> str:
        """
        Concatenate blocks in proper reading order.

        Full-width blocks (mastheads, section headers, footers) act as horizontal
        band separators: everything above a separator is read column-by-column,
        then the separator itself, then everything below it is read column-by-column
        again. This stops a masthead from being incorrectly assigned to a single
        column and appearing mid-stream between two columns.
        """
        spanning = sorted([b for b in self.blocks if b.spans_all_columns], key=lambda b: b.bbox[1])
        columned = [b for b in self.blocks if not b.spans_all_columns]

        # Build y-position band boundaries from spanning blocks
        separators_y = [b.bbox[1] for b in spanning]
        bands = []
        prev_y = float("-inf")
        for sep_y in separators_y + [float("inf")]:
            band_blocks = [b for b in columned if prev_y <= b.bbox[1] < sep_y]
            bands.append(band_blocks)
            prev_y = sep_y

        parts = []
        for i, band_blocks in enumerate(bands):
            sorted_band = sorted(band_blocks, key=lambda b: (b.column_index, b.bbox[1]))
            for b in sorted_band:
                if b.block_type == "table":
                    parts.append(f"\n[TABLE]\n{b.content}\n[/TABLE]\n")
                elif b.block_type == "image":
                    parts.append(f"\n[IMAGE: {b.content}]\n")
                else:
                    parts.append(b.content)
            if i < len(spanning):
                parts.append(spanning[i].content)
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------------

def detect_columns(page: fitz.Page, min_gutter_height_ratio: float = 0.6,
                    min_gutter_width_pts: float = 8.0) -> List[float]:
    """
    Detect column boundaries using a 2D occupancy grid (x bins x y bins) of text blocks.
    A real column gutter must be empty across most of the page's VERTICAL text extent,
    not just locally empty next to one paragraph or around a heading. This avoids false
    positives from natural whitespace in single-column text, and supports 2, 3, or more
    columns (e.g. newsletters).

    Returns column boundary x-values, e.g. [0, 280, 560] for 2 columns,
    or [0, 190, 380, 570] for 3 columns.
    """
    blocks = page.get_text("blocks")
    text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]

    page_width = page.rect.width
    page_height = page.rect.height

    if not text_blocks:
        return [0, page_width]

    # Exclude full-width blocks (mastheads, titles, footers spanning all columns) —
    # otherwise they paper over the gutters and the page looks single-column.
    full_width_threshold = 0.7 * page_width
    column_candidate_blocks = [b for b in text_blocks if (b[2] - b[0]) < full_width_threshold]
    if column_candidate_blocks:
        text_blocks = column_candidate_blocks

    x_bins, y_bins = 120, 60
    bin_w = page_width / x_bins
    bin_h = page_height / y_bins

    occupancy = [[False] * y_bins for _ in range(x_bins)]
    for b in text_blocks:
        x0, y0, x1, y1 = b[0], b[1], b[2], b[3]
        xb0, xb1 = int(x0 / bin_w), min(int(x1 / bin_w), x_bins - 1)
        yb0, yb1 = int(y0 / bin_h), min(int(y1 / bin_h), y_bins - 1)
        for xi in range(xb0, xb1 + 1):
            for yi in range(yb0, yb1 + 1):
                occupancy[xi][yi] = True

    rows_with_any_text = [yi for yi in range(y_bins) if any(occupancy[xi][yi] for xi in range(x_bins))]
    if not rows_with_any_text:
        return [0, page_width]
    body_top, body_bottom = min(rows_with_any_text), max(rows_with_any_text)
    body_row_count = body_bottom - body_top + 1
    min_gutter_bins = max(1, int(min_gutter_width_pts / bin_w))

    empty_x_bins = []
    for xi in range(x_bins):
        empty_rows = sum(1 for yi in range(body_top, body_bottom + 1) if not occupancy[xi][yi])
        if body_row_count > 0 and (empty_rows / body_row_count) >= min_gutter_height_ratio:
            empty_x_bins.append(xi)

    if not empty_x_bins:
        return [0, page_width]

    gutter_runs, run_start, prev = [], empty_x_bins[0], empty_x_bins[0]
    for xi in empty_x_bins[1:]:
        if xi == prev + 1:
            prev = xi
        else:
            gutter_runs.append((run_start, prev))
            run_start, prev = xi, xi
    gutter_runs.append((run_start, prev))

    gutter_runs = [r for r in gutter_runs if (r[1] - r[0] + 1) >= min_gutter_bins]
    gutter_runs = [r for r in gutter_runs if r[0] > 2 and r[1] < x_bins - 3]  # drop page-edge margins

    if not gutter_runs:
        return [0, page_width]

    boundaries = [0]
    for r in gutter_runs:
        boundaries.append(((r[0] + r[1]) / 2 + 0.5) * bin_w)
    boundaries.append(page_width)
    return boundaries

def assign_column_index(bbox: tuple, column_boundaries: List[float]) -> int:
    """Given a bbox and column boundary x-values, return which column it belongs to."""
    x_center = (bbox[0] + bbox[2]) / 2
    for i in range(len(column_boundaries) - 1):
        if column_boundaries[i] <= x_center < column_boundaries[i + 1]:
            return i
    return len(column_boundaries) - 2  # last column fallback


# ---------------------------------------------------------------------------
# Table extraction (via pdfplumber, more reliable for table grids than PyMuPDF)
# ---------------------------------------------------------------------------

def extract_tables_for_page(pdf_path: str, page_num: int) -> List[ExtractedBlock]:
    """Extract tables on a given page (0-indexed) as markdown strings with their bboxes."""
    blocks = []
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num]
        tables = page.find_tables()
        for t in tables:
            data = t.extract()
            if not data:
                continue
            # Convert table rows to markdown
            md_lines = []
            header = data[0] if data[0] else []
            md_lines.append("| " + " | ".join(str(c or "") for c in header) + " |")
            md_lines.append("|" + "---|" * len(header))
            for row in data[1:]:
                md_lines.append("| " + " | ".join(str(c or "") for c in row) + " |")
            blocks.append(
                ExtractedBlock(
                    page_num=page_num,
                    block_type="table",
                    content="\n".join(md_lines),
                    bbox=t.bbox,
                )
            )
    return blocks


# ---------------------------------------------------------------------------
# Image extraction
# ---------------------------------------------------------------------------

def extract_images_for_page(doc: fitz.Document, page: fitz.Page, page_num: int, out_dir: str) -> List[ExtractedBlock]:
    """Extract embedded images on a page, save to disk, return blocks with bbox + saved path."""
    blocks = []
    os.makedirs(out_dir, exist_ok=True)
    img_list = page.get_images(full=True)

    for img_index, img in enumerate(img_list):
        xref = img[0]
        try:
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]
            img_filename = f"page{page_num}_img{img_index}.{ext}"
            img_path = os.path.join(out_dir, img_filename)
            with open(img_path, "wb") as f:
                f.write(image_bytes)

            # Get bbox of this image on the page (approximate via get_image_rects)
            rects = page.get_image_rects(xref)
            bbox = tuple(rects[0]) if rects else (0, 0, 0, 0)

            blocks.append(
                ExtractedBlock(
                    page_num=page_num,
                    block_type="image",
                    content=img_path,
                    bbox=bbox,
                )
            )
        except Exception as e:
            print(f"  [warn] failed to extract image {img_index} on page {page_num}: {e}")

    return blocks


# ---------------------------------------------------------------------------
# Main per-page extraction (text blocks, excluding table regions to avoid duplication)
# ---------------------------------------------------------------------------

def bbox_overlaps(b1: tuple, b2: tuple, threshold: float = 0.5) -> bool:
    """Check if b1 overlaps significantly with b2 (used to skip text inside table regions)."""
    x0 = max(b1[0], b2[0])
    y0 = max(b1[1], b2[1])
    x1 = min(b1[2], b2[2])
    y1 = min(b1[3], b2[3])
    if x1 <= x0 or y1 <= y0:
        return False
    overlap_area = (x1 - x0) * (y1 - y0)
    b1_area = max((b1[2] - b1[0]) * (b1[3] - b1[1]), 1)
    return (overlap_area / b1_area) > threshold


def extract_text_blocks(page: fitz.Page, column_boundaries: List[float],
                         exclude_bboxes: List[tuple]) -> List[ExtractedBlock]:
    """Extract text blocks, tag with column index, skip anything overlapping excluded (table) regions."""
    blocks = []
    raw_blocks = page.get_text("blocks")
    page_width = page.rect.width
    full_width_threshold = 0.7 * page_width
    is_multi_column = len(column_boundaries) > 2

    for b in raw_blocks:
        x0, y0, x1, y1, text, block_no, block_type = b
        if block_type != 0 or not text.strip():
            continue

        bbox = (x0, y0, x1, y1)
        if any(bbox_overlaps(bbox, ex) for ex in exclude_bboxes):
            continue  # skip — this text is part of a table already extracted separately

        spans_all = is_multi_column and (x1 - x0) >= full_width_threshold
        col_idx = assign_column_index(bbox, column_boundaries)
        blocks.append(
            ExtractedBlock(
                page_num=page.number,
                block_type="text",
                content=text.strip(),
                bbox=bbox,
                column_index=col_idx,
                spans_all_columns=spans_all,
            )
        )
    return blocks


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def extract_pdf(pdf_path: str, image_out_dir: str = "./extracted_images") -> List[PageExtraction]:
    doc = fitz.open(pdf_path)
    results = []

    for page_num in range(len(doc)):
        page = doc[page_num]

        column_boundaries = detect_columns(page)
        num_columns = len(column_boundaries) - 1

        table_blocks = extract_tables_for_page(pdf_path, page_num)
        print("the table clocks is",table_blocks)
        table_bboxes = [t.bbox for t in table_blocks]

        image_blocks = extract_images_for_page(doc, page, page_num, image_out_dir)

        text_blocks = extract_text_blocks(page, column_boundaries, exclude_bboxes=table_bboxes)
        # assign column index to table/image blocks too, for correct reading order
        for blk in table_blocks + image_blocks:
            blk.column_index = assign_column_index(blk.bbox, column_boundaries)

        page_result = PageExtraction(
            page_num=page_num,
            num_columns=num_columns,
            blocks=text_blocks + table_blocks + image_blocks,
        )
        results.append(page_result)

        print(f"Page {page_num + 1}: {num_columns} column(s), "
              f"{len(text_blocks)} text block(s), {len(table_blocks)} table(s), "
              f"{len(image_blocks)} image(s)")

    doc.close()
    return results


def pdf_to_full_text(pdf_path: str, image_out_dir: str = "./extracted_images") -> str:
    """Convenience wrapper: returns the full document text in correct reading order."""
    pages = extract_pdf(pdf_path, image_out_dir)
    full_text = []
    for p in pages:
        full_text.append(f"\n--- Page {p.page_num + 1} ---\n")
        full_text.append(p.get_reading_order_text())
    return "\n".join(full_text)


# ---------------------------------------------------------------------------
# CLI usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pdf_extractor.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    text = pdf_to_full_text(pdf_path)
    print("\n\n========== EXTRACTED TEXT ==========\n")
    print(text)