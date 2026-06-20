from providers.llm_providers import GroqProvider


class LLMService:

    def __init__(self):
        self.provider = GroqProvider()

    def extract_text_from_image(
        self,
        base64_image: str,
        mime_type: str
    ) -> str:

        prompt = """
        You are an OCR and document understanding assistant.

        Task:
        Extract all readable text from the image.

        Rules:
        - Do not add extra explanation.
        - Preserve line breaks as much as possible.
        - If the image contains a table, keep the table structure readable.
        - If text is unclear, write [unclear].
        - Return only the extracted text.
        """
       
        return self.provider.extract_text_from_image(
            base64_image=base64_image,
            mime_type=mime_type,
            prompt=prompt
        )