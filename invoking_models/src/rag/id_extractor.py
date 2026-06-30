"""
src/rag/id_extractor.py
-----------------------
Standalone ID-extraction utility used at both ingestion time and query time.

Design rules (do NOT violate):
  - Completely independent of validate_query / RAGGuardrails.
    These are two entirely separate concerns and must never share a code path.
  - No LLM calls. Pure regex, O(n) in text length.
  - Patterns are sourced exclusively from settings.ID_PATTERNS so that
    production operators can retune them via config without a code change.
  - Compiled patterns are cached at module load (via _compile_patterns()).

Public surface:
    extract_candidate_ids(text: str) -> list[str]
        Called at ingestion time on each DocumentChunk's content.

    extract_query_ids(query: str) -> list[str]
        Called at query time on the raw user query string.
        Use the raw query, not the rewritten/standalone form, so user intent
        (the literal ID they typed) is never altered by the rewriter.
"""

from __future__ import annotations

import re
import logging
from functools import lru_cache
from typing import List

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _compile_patterns() -> list[tuple[str, re.Pattern]]:
    """
    Compile all patterns from settings.ID_PATTERNS exactly once.
    lru_cache ensures this never reruns even under concurrent requests.

    Returns a list of (name, compiled_pattern) tuples.
    """
    # Import here to avoid circular imports at module load time
    from config import settings

    compiled: list[tuple[str, re.Pattern]] = []
    for entry in settings.ID_PATTERNS:
        name = entry["name"]
        raw_pattern = entry["pattern"]
        try:
            pat = re.compile(raw_pattern, re.IGNORECASE)
            compiled.append((name, pat))
            logger.debug("Compiled ID pattern '%s': %s", name, raw_pattern)
        except re.error as exc:
            # A bad pattern in config must not silently swallow all ID extraction.
            # Log loudly and skip so the other patterns still run.
            logger.error(
                "Invalid regex in ID_PATTERNS entry '%s': %r — skipping. Error: %s",
                name, raw_pattern, exc
            )
    return compiled


def _extract(text: str) -> list[str]:
    """
    Core extraction logic shared by both public functions.

    Algorithm:
      1. Run each compiled pattern against `text`.
      2. Collect all matches across all patterns.
      3. Deduplicate while preserving first-seen order (dict trick).
      4. Strip surrounding whitespace from each match.
      5. Discard empty strings.

    Returns a list of unique matched strings, order-preserved.
    """
    if not text or not isinstance(text, str):
        return []

    seen: dict[str, None] = {}  # ordered set via dict (Python 3.7+)

    for name, pattern in _compile_patterns():
        for match in pattern.finditer(text):
            token = match.group(0).strip()
            if token and token not in seen:
                seen[token] = None
                logger.debug("ID pattern '%s' matched: %r", name, token)

    return list(seen.keys())


def extract_candidate_ids(text: str) -> List[str]:
    """
    Extract structured IDs from a document chunk's content at ingestion time.

    Called by LayoutAwareChunker on every chunk before upsert.
    Results are stored as the `extracted_ids` Qdrant payload field and
    indexed as a KEYWORD array for MatchAny scroll queries.

    Args:
        text: Raw chunk content string.

    Returns:
        Deduplicated list of matched ID strings. Empty list if none found.
    """
    ids = _extract(text)
    if ids:
        logger.debug("extract_candidate_ids: found %d IDs in chunk", len(ids))
    return ids


def extract_query_ids(query: str) -> List[str]:
    """
    Extract structured IDs from a user query at retrieval time.

    *** Intentionally kept separate from validate_query / RAGGuardrails. ***
    These are two independent concerns:
      - validate_query: LLM-based safety classifier (async, expensive, I/O-bound)
      - extract_query_ids: regex scan (sync, microseconds, pure CPU)
    They must never be merged into a single function or LLM call.

    Always called on the RAW user query, not the LLM-rewritten standalone query,
    so the literal ID the user typed is never altered.

    Args:
        query: Raw user query string as received by the endpoint.

    Returns:
        Deduplicated list of matched ID strings. Empty list triggers hybrid-only path.
    """
    ids = _extract(query)
    if ids:
        logger.info("extract_query_ids: detected %d ID(s) in query: %r", len(ids), ids)
    return ids
