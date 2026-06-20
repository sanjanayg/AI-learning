import fitz


class PDFService:

    @staticmethod
    def extract_text_from_pdf(pdf_bytes: bytes) -> str:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        extracted_text = []

        for page_no, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()

            if text:
                extracted_text.append(f"\n--- Page {page_no} ---\n{text}")

        doc.close()

        return "\n".join(extracted_text).strip()

    @staticmethod
    def analyze_pdf(pdf_bytes: bytes) -> dict:
        """
        Returns PDF type:
        - text_pdf
        - scanned_pdf
        - hybrid_pdf
        """

        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as e:
            raise ValueError(f"Failed to open PDF: {e}")

        total_pages = len(doc)
        pages_with_text = 0
        pages_with_images = 0

        for page in doc:
            text = page.get_text("text").strip()
            images = page.get_images(full=True)

            if len(text) > 50:
                pages_with_text += 1

            if images:
                pages_with_images += 1

        doc.close()

        if pages_with_text == total_pages:
            pdf_type = "text_pdf"

        elif pages_with_text == 0 and pages_with_images > 0:
            pdf_type = "scanned_pdf"

        else:
            pdf_type = "hybrid_pdf"

        return {
            "pdf_type": pdf_type,
            "total_pages": total_pages,
            "pages_with_text": pages_with_text,
            "pages_with_images": pages_with_images
        }

    @staticmethod
    def convert_pdf_to_images(pdf_bytes: bytes) -> list[dict]:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        images = []

        for page_no, page in enumerate(doc, start=1):
            pix = page.get_pixmap(dpi=200)

            image_bytes = pix.tobytes("png")

            images.append({
                "page_no": page_no,
                "image_bytes": image_bytes,
                "mime_type": "image/png"
            })

        doc.close()
        
        return images