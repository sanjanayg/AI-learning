"""
ChatSummaryService
==================
Handles all LLM-based summarization for the Chat Summary Report feature.

Design:
- First-time request  : summarize_full_conversation()
  - If the full conversation fits in the token budget  → single LLM call
  - If it exceeds the budget                          → map-reduce strategy
    (chunk → summarize each chunk → combine → final synthesis)
- Subsequent requests : update_summary_with_new_messages()
  - Sends only the NEW messages + the existing summary to the LLM
  - Avoids re-processing the full history every time

Token budget: TOKEN_BUDGET constant (conservative, matches RAGGuardrails limit).
"""

import asyncio
import json
import logging
from typing import Sequence

import tiktoken

from providers.llm_providers import GroqProvider

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Tokens below which we do a single pass; above triggers map-reduce.
TOKEN_BUDGET = 6000

# Characters per chunk in map-reduce phase (rough heuristic: ~4 chars/token).
CHUNK_CHAR_SIZE = TOKEN_BUDGET * 4

# Model used for summarization (matches the "medium" tier in llm_service.py).
SUMMARY_MODEL = "llama-3.3-70b-versatile"

# ── Prompt templates ─────────────────────────────────────────────────────────

_SYSTEM_PROMPT_FULL = """
You are an expert technical analyst tasked with creating a structured chat summary report.

Analyse the entire conversation and return a single JSON object with exactly these keys:
{
  "executive_summary": "A concise 3-5 sentence overview of the entire conversation.",
  "topics_discussed": ["topic 1", "topic 2", "..."],
  "key_user_questions": ["verbatim or paraphrased key questions asked by the user", "..."],
  "key_assistant_responses": ["key answers or insights provided by the assistant", "..."],
  "decisions_made": ["any decisions, conclusions, or agreements reached", "..."],
  "errors_issues": ["any errors, problems, or blockers discussed", "..."],
  "action_items": ["specific next steps or follow-up actions identified", "..."]
}

Rules:
- Return ONLY the JSON object. No markdown fences, no preamble, no trailing text.
- Keep array items concise (1-2 sentences each).
- If a section has nothing to report, use an empty array [] or a brief "None identified." string.
- Do not hallucinate facts not present in the conversation.
""".strip()

_SYSTEM_PROMPT_UPDATE = """
You are an expert technical analyst. You have an existing summary of a conversation and a set of new messages appended to that conversation.

Update the existing summary to incorporate the new messages and return a single JSON object with exactly these keys:
{
  "executive_summary": "...",
  "topics_discussed": [...],
  "key_user_questions": [...],
  "key_assistant_responses": [...],
  "decisions_made": [...],
  "errors_issues": [...],
  "action_items": [...]
}

Rules:
- Return ONLY the JSON object. No markdown fences, no preamble, no trailing text.
- Merge/update each section; do not throw away information from the existing summary unless it's been superseded.
- Keep array items concise (1-2 sentences each).
- If a section has nothing to report, use an empty array [].
""".strip()

_SYSTEM_PROMPT_CHUNK = """
You are summarising a segment of a longer conversation. Extract the most important points from this segment only.

Return a plain-text paragraph (not JSON) that captures:
- The main topic(s) discussed
- Key questions asked
- Key answers or decisions
- Any errors or action items mentioned

Be concise. 3-6 sentences maximum.
""".strip()

_SYSTEM_PROMPT_REDUCE = """
You are synthesising chunk-level summaries of a very long conversation into a final structured report.

Given the chunk summaries below, produce a single JSON object with exactly these keys:
{
  "executive_summary": "...",
  "topics_discussed": [...],
  "key_user_questions": [...],
  "key_assistant_responses": [...],
  "decisions_made": [...],
  "errors_issues": [...],
  "action_items": [...]
}

Rules:
- Return ONLY the JSON object. No markdown fences, no preamble, no trailing text.
- Synthesise across all chunks; do not repeat the same point multiple times.
- Keep array items concise (1-2 sentences each).
- If a section has nothing to report, use an empty array [].
""".strip()


# ── Service class ─────────────────────────────────────────────────────────────

class ChatSummaryService:
    """
    Orchestrates rolling and map-reduce LLM summarization for chat sessions.
    Instantiate once per request (lightweight — no shared state).
    """

    def __init__(self) -> None:
        self.provider = GroqProvider()
        # Use the cl100k_base encoder as a universal approximation; it's already
        # pulled in by tiktoken which is listed in requirements.txt.
        try:
            self._enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._enc = None

    # ── Public API ────────────────────────────────────────────────────────────

    async def summarize_full_conversation(
        self, messages: Sequence
    ) -> str:
        """
        Generate a structured JSON summary of the complete conversation.
        Automatically uses map-reduce if the conversation is too long.

        Parameters
        ----------
        messages : Sequence of ChatMessage ORM objects

        Returns
        -------
        str : JSON string matching the structured summary schema
        """
        if not messages:
            return json.dumps({
                "executive_summary": "No messages found in this conversation.",
                "topics_discussed": [],
                "key_user_questions": [],
                "key_assistant_responses": [],
                "decisions_made": [],
                "errors_issues": [],
                "action_items": [],
            })

        conversation_text = self._format_messages(messages)
        token_count = self._estimate_tokens(conversation_text)

        logger.info(
            "summarize_full_conversation: %d messages, ~%d tokens",
            len(messages),
            token_count,
        )

        if token_count <= TOKEN_BUDGET:
            logger.info("Token budget OK — using single-pass summarization.")
            return await self._single_pass_summarize(conversation_text)
        else:
            logger.info(
                "Conversation exceeds token budget (%d > %d) — using map-reduce.",
                token_count,
                TOKEN_BUDGET,
            )
            return await self._map_reduce_summarize(messages)

    async def update_summary_with_new_messages(
        self,
        existing_summary: str,
        new_messages: Sequence,
    ) -> str:
        """
        Incrementally update an existing structured summary with only the new
        messages since the last summary was generated.

        Parameters
        ----------
        existing_summary : str — JSON string of the previous summary
        new_messages     : Sequence of new ChatMessage ORM objects

        Returns
        -------
        str : Updated JSON string matching the structured summary schema
        """
        if not new_messages:
            logger.info("No new messages — returning cached summary unchanged.")
            return existing_summary

        new_text = self._format_messages(new_messages)

        user_content = (
            f"EXISTING SUMMARY:\n{existing_summary}\n\n"
            f"NEW MESSAGES:\n{new_text}"
        )

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT_UPDATE},
            {"role": "user", "content": user_content},
        ]

        logger.info(
            "update_summary_with_new_messages: %d new messages to incorporate.",
            len(new_messages),
        )
        result = await asyncio.to_thread(self._call_llm, messages)
        return self._parse_json_response(result)

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _single_pass_summarize(self, conversation_text: str) -> str:
        """Single LLM call for the full conversation text."""
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT_FULL},
            {"role": "user", "content": f"CONVERSATION:\n{conversation_text}"},
        ]
        result = await asyncio.to_thread(self._call_llm, messages)
        return self._parse_json_response(result)

    async def _map_reduce_summarize(self, messages: Sequence) -> str:
        """
        Map phase  : split conversation into character-based chunks; summarize each.
        Reduce phase: feed all chunk summaries to the LLM for a final structured JSON.
        """
        # ── Map ──────────────────────────────────────────────────────────────
        conversation_text = self._format_messages(messages)
        chunks = self._split_into_chunks(conversation_text, CHUNK_CHAR_SIZE)
        logger.info("map-reduce: %d chunks to process.", len(chunks))

        chunk_summary_tasks = [
            asyncio.to_thread(
                self._call_llm,
                [
                    {"role": "system", "content": _SYSTEM_PROMPT_CHUNK},
                    {"role": "user", "content": f"CONVERSATION SEGMENT:\n{chunk}"},
                ],
            )
            for chunk in chunks
        ]
        chunk_results = await asyncio.gather(*chunk_summary_tasks, return_exceptions=True)

        chunk_summaries = []
        for i, res in enumerate(chunk_results):
            if isinstance(res, Exception):
                logger.warning("Chunk %d summarization failed: %s", i, res)
                chunk_summaries.append(f"[Chunk {i+1} summary unavailable due to error]")
            else:
                chunk_summaries.append(res)

        # ── Reduce ────────────────────────────────────────────────────────────
        combined = "\n\n---\n\n".join(
            f"CHUNK {i+1} SUMMARY:\n{s}" for i, s in enumerate(chunk_summaries)
        )
        reduce_messages = [
            {"role": "system", "content": _SYSTEM_PROMPT_REDUCE},
            {"role": "user", "content": combined},
        ]
        final_result = await asyncio.to_thread(self._call_llm, reduce_messages)
        return self._parse_json_response(final_result)

    def _call_llm(self, messages: list[dict]) -> str:
        """
        Synchronous Groq completion call (run inside asyncio.to_thread).
        Returns raw response string.
        """
        try:
            completion = self.provider.client.chat.completions.create(
                model=SUMMARY_MODEL,
                messages=messages,
                temperature=0.0,
            )
            return completion.choices[0].message.content.strip()
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            raise ValueError(f"LLM summarization failed: {exc}") from exc

    def _parse_json_response(self, raw: str) -> str:
        """
        Attempt to parse the LLM response as JSON and re-serialise it
        (normalises formatting). Falls back to wrapping the raw text in an
        executive_summary field if parsing fails, so the PDF generator always
        receives a valid structure.
        """
        # Strip markdown code fences if the model wrapped it anyway
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            # Drop first and last fence lines
            cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned

        try:
            parsed = json.loads(cleaned)
            # Ensure all required keys exist, defaulting to [] or ""
            defaults = {
                "executive_summary": "",
                "topics_discussed": [],
                "key_user_questions": [],
                "key_assistant_responses": [],
                "decisions_made": [],
                "errors_issues": [],
                "action_items": [],
            }
            for key, default in defaults.items():
                parsed.setdefault(key, default)
            return json.dumps(parsed)
        except (json.JSONDecodeError, ValueError):
            logger.warning(
                "LLM response was not valid JSON — wrapping in fallback structure."
            )
            fallback = {
                "executive_summary": raw,
                "topics_discussed": [],
                "key_user_questions": [],
                "key_assistant_responses": [],
                "decisions_made": [],
                "errors_issues": [],
                "action_items": ["Review the raw summary above."],
            }
            return json.dumps(fallback)

    @staticmethod
    def _format_messages(messages: Sequence) -> str:
        """Format a list of ChatMessage ORM objects into a readable conversation string."""
        lines = []
        for msg in messages:
            role_label = msg.role.upper()
            lines.append(f"{role_label}: {msg.content}")
        return "\n".join(lines)

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count using tiktoken when available.
        Falls back to len(text) // 4 (the standard 4-chars-per-token heuristic).
        """
        if self._enc is not None:
            try:
                return len(self._enc.encode(text))
            except Exception:
                pass
        return len(text) // 4

    @staticmethod
    def _split_into_chunks(text: str, chunk_size: int) -> list[str]:
        """
        Split `text` into overlapping character-level chunks of `chunk_size`.
        Uses a 10% overlap to avoid cutting mid-thought.
        """
        overlap = chunk_size // 10
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            if end == len(text):
                break
            start = end - overlap
        return chunks
