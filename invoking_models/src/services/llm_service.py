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

    async def generate_grounded_response(self, query: str, context_chunks: list) -> str:
        """
        Synthesizes a grounded response from the provided context chunks.
        Instructs the LLM to rely ONLY on context, format citations strictly, 
        and refuse to answer if the context is insufficient.
        Runs in a separate thread pool to ensure non-blocking operation.
        """
        if not context_chunks:
            return "I am sorry, but the provided documents do not contain the information required to answer this question."

        context_blocks = []
        for idx, chunk in enumerate(context_chunks, start=1):
            block = (
                f"--- Document Chunk {idx} ---\n"
                f"Source File: {chunk.file_name}\n"
                f"Page Number: {chunk.page_number}\n"
                f"Element Type: {chunk.element_type}\n"
                f"Content:\n{chunk.content}\n"
            )
            context_blocks.append(block)
        
        context_text = "\n".join(context_blocks)

        system_prompt = """
        You are a highly precise, multi-tenant RAG synthesis engine. 
        Your task is to answer the user's query based ONLY on the provided document chunks.

        Strict Rules:
        1. Rely ONLY on the provided document chunks. Do NOT use any external, prior, or pre-trained knowledge.
        2. If the context does not contain the answer, or if the context is insufficient, you must respond with exactly: "I am sorry, but the provided documents do not contain the information required to answer this question." Do not attempt to explain why or speculate.
        3. You must provide strict inline citations for every statement you make that is derived from the documents. Format the citation exactly as: `[Source: <filename>, Page: <page_number>]`. 
           - The filename and page number must exactly match the source metadata provided in the chunk.
           - Place the citation directly at the end of the sentence or clause it supports.
        4. If you synthesize information from multiple files, clearly distinguish which information comes from which file and cite them separately.
        5. Do not make assumptions, extrapolate, or generalize beyond what is explicitly stated in the context.
        6. Return only the raw synthesized text with inline citations. Do not add any greetings, preambles, or conversational remarks.
        """.strip()

        user_content = f"""
        User Query: {query}

        Provided Document Chunks:
        {context_text}
        """.strip()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        import asyncio
        response_text = await asyncio.to_thread(self._call_groq_completion, messages)
        return response_text.strip()

    def _call_groq_completion(self, messages: list[dict]) -> str:
        try:
            completion = self.provider.client.chat.completions.create(
                model=self.provider.versatile_model,
                messages=messages,
                temperature=0.0
            )
            return completion.choices[0].message.content
        except Exception as exc:
            raise ValueError(f"LLM synthesis failed: {str(exc)}") from exc
