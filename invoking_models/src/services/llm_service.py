import httpx
from groq import GroqError, APIError, RateLimitError, InternalServerError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from providers.llm_providers import GroqProvider


class LLMService:

    def __init__(self):
        self.provider = GroqProvider()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        # Catching Groq-specific SDK errors alongside standard HTTPX fallback exceptions
        retry=retry_if_exception_type((
            GroqError, APIError,
            RateLimitError, 
            InternalServerError, 
            httpx.HTTPStatusError, 
            httpx.RequestError
        )),
        reraise=True  # Guarantees the original clean API error bubbles up to FastAPI for logging
    )
    def extract_text_from_image(
        self,
        base64_image: str,
        mime_type: str
    ) -> str:
        # Optimized prompt instructing the Vision LLM to render tables into Markdown
        # This increases structural accuracy for financial/tabular document extractions.
        prompt = """
        You are an expert OCR and document layout understanding engine.

        Task:
        Extract all readable text, tabular metrics, and key data points from the image.

        Rules:
        - Do not add any conversational remarks, introductions, or pleasantries.
        - Preserve semantic line breaks and paragraph structure precisely.
        - If the image contains a table, render it as a highly clean Markdown table.
        - Keep visual layout structure as consistent as possible.
        - If a specific text fragment is blurry or completely unreadable, mark it explicitly as [unclear].
        - Return only the raw extracted text.
        """.strip()
       
        return self.provider.extract_text_from_image(
            base64_image=base64_image,
            mime_type=mime_type,
            prompt=prompt
        )