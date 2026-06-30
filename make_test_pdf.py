"""
Generates 4 single-page test PDFs to exercise different extraction cases:
  1. test_single_column.pdf  - plain single-column text
  2. test_two_column.pdf     - two-column layout (like the IEEE template)
  3. test_table.pdf          - text + an actual grid table
  4. test_image.pdf          - text + an embedded image

Run: python make_test_pdfs.py
Outputs land in ./test_pdfs/
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import fitz  # PyMuPDF, used to create a quick image to embed

OUT_DIR = "test_pdfs"
os.makedirs(OUT_DIR, exist_ok=True)

LOREM = (
    "This is a sample paragraph used to test PDF text extraction. "
    "It contains several sentences so that the extraction script has "
    "enough content to work with. The quick brown fox jumps over the "
    "lazy dog. Extraction quality depends on correct reading order, "
    "especially when columns or tables are involved. "
)


def wrap_and_draw(c, x, y, max_width, content, font="Helvetica", size=11, leading=15):
    c.setFont(font, size)
    text = c.beginText(x, y)
    text.setLeading(leading)
    line = ""
    for word in content.split():
        test_line = line + word + " "
        if c.stringWidth(test_line, font, size) > max_width:
            text.textLine(line)
            line = word + " "
        else:
            line = test_line
    if line:
        text.textLine(line)
    c.drawText(text)


# ---------------------------------------------------------------------------
# 1. Single column
# ---------------------------------------------------------------------------
def make_single_column():
    path = os.path.join(OUT_DIR, "test_single_column.pdf")
    c = canvas.Canvas(path, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, height - 72, "Single Column Test Document")

    wrap_and_draw(c, 72, height - 110, width - 144, (LOREM * 6).strip())
    c.save()
    print(f"Created {path}")


# ---------------------------------------------------------------------------
# 2. Two column (mimics IEEE-style layout)
# ---------------------------------------------------------------------------
def make_two_column():
    path = os.path.join(OUT_DIR, "test_two_column.pdf")
    c = canvas.Canvas(path, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, height - 60, "Two Column Test Document")

    col_width = (width - 144 - 20) / 2  # two columns with a 20pt gutter
    left_x = 72
    right_x = 72 + col_width + 20
    top_y = height - 100

    left_content = ("LEFT COLUMN. " + LOREM * 8).strip()
    right_content = ("RIGHT COLUMN. " + LOREM * 8).strip()

    wrap_and_draw(c, left_x, top_y, col_width, left_content, size=10, leading=14)
    wrap_and_draw(c, right_x, top_y, col_width, right_content, size=10, leading=14)
    c.save()
    print(f"Created {path}")


# ---------------------------------------------------------------------------
# 3. Table
# ---------------------------------------------------------------------------
def make_table():
    path = os.path.join(OUT_DIR, "test_table.pdf")
    c = canvas.Canvas(path, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, height - 60, "Table Test Document")

    c.setFont("Helvetica", 11)
    c.drawString(72, height - 90, "Some intro text before the table appears below.")

    data = [
        ["Font Size", "Appearance", "Bold", "Italic"],
        ["8", "table caption", "-", "reference item"],
        ["9", "author email", "abstract", "abstract heading"],
        ["10", "level-1 heading", "-", "level-2 heading"],
        ["24", "title", "-", "-"],
    ]
    rows, cols = len(data), len(data[0])
    table_x, table_y = 72, height - 130
    row_h = 24
    col_w = (width - 144) / cols

    for r, row in enumerate(data):
        for col, val in enumerate(row):
            x = table_x + col * col_w
            y = table_y - r * row_h
            c.rect(x, y - row_h, col_w, row_h)
            c.setFont("Helvetica-Bold" if r == 0 else "Helvetica", 9)
            c.drawString(x + 4, y - row_h + 7, str(val))

    c.setFont("Helvetica", 11)
    c.drawString(72, table_y - rows * row_h - 30, "Some closing text after the table.")
    c.save()
    print(f"Created {path}")


# ---------------------------------------------------------------------------
# 4. Image
# ---------------------------------------------------------------------------
def make_image():
    img_path = os.path.join(OUT_DIR, "_sample_image.png")
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 300, 200))
    pix.set_rect(pix.irect, (70, 130, 180))
    pix.save(img_path)

    path = os.path.join(OUT_DIR, "test_image.pdf")
    c = canvas.Canvas(path, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, height - 60, "Image Test Document")

    c.setFont("Helvetica", 11)
    c.drawString(72, height - 90, "Text appears above the image below.")

    c.drawImage(img_path, 72, height - 320, width=300, height=200)

    c.setFont("Helvetica", 11)
    c.drawString(72, height - 340, "Text appears below the image as well.")
    c.save()
    print(f"Created {path}")
    os.remove(img_path)


# ---------------------------------------------------------------------------
# 5. Three column newsletter (mimics Hammond Elementary School layout)
# ---------------------------------------------------------------------------
def make_three_column_newsletter():
    path = os.path.join(OUT_DIR, "test_three_column_newsletter.pdf")
    c = canvas.Canvas(path, pagesize=letter)
    width, height = letter

    # Full-width masthead (spans all 3 columns)
    c.setFont("Helvetica-BoldOblique", 18)
    c.drawCentredString(width / 2, height - 50, "Hammond Elementary School")
    c.setFont("Helvetica", 9)
    c.drawRightString(width - 50, height - 50, "Fall 2000")
    c.line(50, height - 60, width - 50, height - 60)

    n_cols = 3
    margin = 50
    gutter = 14
    col_width = (width - 2 * margin - (n_cols - 1) * gutter) / n_cols
    top_y = height - 90

    col1_x = margin
    col2_x = margin + col_width + gutter
    col3_x = margin + 2 * (col_width + gutter)

    # Column 1
    c.setFont("Helvetica-Bold", 13)
    c.drawString(col1_x, top_y, "Kids' Right to Vote")
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(col1_x, top_y - 16, "by Lindsey")
    wrap_and_draw(c, col1_x, top_y - 34, col_width,
                  ("COLUMN ONE. " + LOREM * 6).strip(), size=9, leading=12)

    # Column 2
    c.setFont("Helvetica", 9)
    wrap_and_draw(c, col2_x, top_y, col_width,
                  ("COLUMN TWO. " + LOREM * 6).strip(), size=9, leading=12)

    # Column 3
    c.setFont("Helvetica-Bold", 13)
    c.drawString(col3_x, top_y, "Harry Potter")
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(col3_x, top_y - 16, "by Bryan")
    wrap_and_draw(c, col3_x, top_y - 34, col_width,
                  ("COLUMN THREE. " + LOREM * 6).strip(), size=9, leading=12)

    c.save()
    print(f"Created {path}")


if __name__ == "__main__":
    make_single_column()
    make_two_column()
    make_table()
    make_image()
    make_three_column_newsletter()
    print(f"\nAll test PDFs created in ./{OUT_DIR}/")