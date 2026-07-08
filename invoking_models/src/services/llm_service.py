import httpx
from groq import GroqError, APIError, RateLimitError, InternalServerError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import json
from providers.llm_providers import GroqProvider

MODEL_MAP = {
            "instant": "openai/gpt-oss-20b",
            "medium": "llama-3.3-70b-versatile",   
            "high": "qwen/qwen3.6-27b",
            "auto": "auto",
            }

ROUTER_SYSTEM_PROMPT = """You are a query complexity classifier for a RAG system.
                        Classify the user's query into exactly one tier based on reasoning difficulty:

                        - "instant": simple factual lookups, definitions, single-fact retrieval, greetings/small talk
                        - "medium": moderate reasoning, comparisons, summarization across a few chunks
                        - "high": multi-hop reasoning, ambiguous questions, synthesis across many sources, complex analysis, math/logic-heavy questions

                        Respond with ONLY a JSON object, no other text, no markdown fences:
                        {"tier": "instant" | "medium" | "high"}
                        """
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

    async def generate_grounded_response(self, query: str, context_chunks: list,history: list[dict] | None = None,model=None) -> str:
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

        messages = [{"role": "system", "content": system_prompt}]
    
        if history:
            messages.extend(history)  # prior user/assistant turns go in the middle
        
        messages.append({"role": "user", "content": user_content})

        import asyncio
        response_text = await asyncio.to_thread(self._call_groq_completion, messages,model)
        return response_text

    def _call_groq_completion(self, messages: list[dict], model=None) -> dict:
        if model:
            self.provider.versatile_model = model
        try:
            completion = self.provider.client.chat.completions.create(
                model=self.provider.versatile_model,
                messages=messages,
                temperature=0.0
            )
            return {
                "response": completion.choices[0].message.content.strip(),
                "total_tokens": completion.usage.total_tokens if completion.usage else 0,
            }
        except Exception as exc:
            raise ValueError(f"LLM synthesis failed: {str(exc)}") from exc
        
    async def rewrite_query(self, history: list[dict], query: str) -> str:
        """
        Rewrite a follow-up question into a standalone query for retrieval.
        """

        if not history:
            return query

        formatted_history = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in history[-6:]
        )

        prompt = f"""
                    You are a query rewriting assistant for a Retrieval-Augmented Generation (RAG) system.

                    Your task is to rewrite the user's latest question into a complete, self-contained question that can be understood without the conversation history.

                    Instructions:
                    - Rewrite ONLY the latest user question.
                    - Do NOT answer the question.
                    - Do NOT add explanations, assumptions, or extra information.
                    - Resolve all references such as "it", "its", "they", "them", "this", "that", "these", "those", "he", "she", and "there" using the conversation history.
                    - Replace omitted subjects or objects when they are clear from the conversation.
                    - Preserve the original meaning, intent, tone, and level of detail.
                    - If the latest question is already standalone and unambiguous, return it unchanged.
                    - If the conversation history does not provide enough information to resolve a reference, return the latest question unchanged.
                    - Return only the rewritten question. No quotation marks, labels, or additional text.

                    Conversation History:
                    {formatted_history}

                    Latest User Question:
                    {query}

                    Standalone Question:
                """

        messages = [
            {
                "role": "system",
                "content": (
                    "You rewrite follow-up questions into standalone search queries. "
                    "Never answer the question."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        try:
            result = self._call_groq_completion(messages)
            rewritten_query = result["response"]
            if not rewritten_query:
                return query
            return rewritten_query

        except Exception as e:
            return query
        
    async def route_intelligence(self, query: str, intelligence_mode: str = "auto") -> str:
        """
        Calls a fast, cheap model to classify query complexity.
        Returns one of: "instant", "medium", "high".
        Falls back to "medium" on any failure (safe middle ground).
        """
        try:
            import asyncio
            response = await asyncio.to_thread(
                self.provider.client.chat.completions.create,
                model="openai/gpt-oss-20b",
                messages=[
                    {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                temperature=0,
                max_tokens=20,
            )

            raw = response.choices[0].message.content.strip()
            parsed = json.loads(raw)
            tier = parsed.get("tier", "medium").lower()

            if tier not in ("instant", "medium", "high"):
                return "medium"

            return tier

        except (json.JSONDecodeError, KeyError, IndexError, AttributeError) as e:
            return "medium"
        except Exception as e:
            return "medium"
    async def select_model(self, intelligence_mode: str, query: str) -> str:
        resolved_tier = intelligence_mode.lower()
        if resolved_tier == "auto":
            resolved_tier = await self.route_intelligence(query)
        model = MODEL_MAP.get(resolved_tier, "llama-3.3-70b-versatile")
        return model

    async def generate_response(self, prompt: str) -> str:
        import asyncio
        messages = [{"role": "user", "content": prompt}]
        result = await asyncio.to_thread(self._call_groq_completion, messages, None)
        return result["response"]