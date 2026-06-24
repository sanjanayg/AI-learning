# class TxtService:

#     @staticmethod
#     def extract_text(file_bytes: bytes) -> str:
#         encodings = ["utf-8", "utf-16", "latin-1"]

#         for encoding in encodings:
#             try:
#                 return file_bytes.decode(encoding).strip()
#             except UnicodeDecodeError:
#                 continue

#         raise ValueError("Unable to decode TXT file")
    
import logging

from config import settings

logger = logging.getLogger(__name__)

MAX_FILE_SIZE_BYTES = getattr(settings, "TXT_MAX_FILE_SIZE_BYTES", 10 * 1024 * 1024)  # 10MB

# BOM signatures checked in order of specificity (longer/more specific first)
_BOM_ENCODINGS = [
    (b"\xef\xbb\xbf", "utf-8-sig"),
    (b"\xff\xfe\x00\x00", "utf-32-le"),
    (b"\x00\x00\xfe\xff", "utf-32-be"),
    (b"\xff\xfe", "utf-16-le"),
    (b"\xfe\xff", "utf-16-be"),
]


class TxtProcessingError(Exception):
    """Raised for any recoverable TXT processing failure (bad file, too large, undecodable, etc.)."""


class TxtService:

    @staticmethod
    def _detect_bom_encoding(file_bytes: bytes) -> str | None:
        for bom, encoding in _BOM_ENCODINGS:
            if file_bytes.startswith(bom):
                return encoding
        return None

    @staticmethod
    def extract_text(file_bytes: bytes) -> str:
        if not file_bytes:
            raise TxtProcessingError("Empty file provided.")

        if len(file_bytes) > MAX_FILE_SIZE_BYTES:
            raise TxtProcessingError(
                f"File size {len(file_bytes)} bytes exceeds max allowed "
                f"{MAX_FILE_SIZE_BYTES} bytes."
            )

        # 1. Trust a BOM if present — it's an explicit, unambiguous signal.
        bom_encoding = TxtService._detect_bom_encoding(file_bytes)
        if bom_encoding:
            try:
                text = file_bytes.decode(bom_encoding).strip()
                logger.info("TXT decoded via BOM-detected encoding=%s", bom_encoding)
                return text
            except UnicodeDecodeError as exc:
                logger.warning("BOM indicated %s but decode failed: %s", bom_encoding, exc)
                # fall through to strict attempts below

        # 2. Strict UTF-8 — the overwhelmingly common case, and unambiguous when it succeeds.
        try:
            text = file_bytes.decode("utf-8").strip()
            logger.info("TXT decoded via utf-8")
            return text
        except UnicodeDecodeError:
            pass

        # 3. Windows-1252 — common for legacy/Excel-exported text files, stricter than latin-1
        # (rejects a handful of byte values latin-1 would accept), reducing false "successes".
        try:
            text = file_bytes.decode("windows-1252").strip()
            logger.info("TXT decoded via windows-1252")
            return text
        except UnicodeDecodeError:
            pass

        # 4. Last resort: latin-1 always succeeds (every byte is a valid code point), so this
        # is a best-effort fallback, not a real validation — log it loudly so downstream
        # consumers know the content may be imperfect.
        text = file_bytes.decode("latin-1").strip()
        logger.warning(
            "TXT decoded via latin-1 fallback — content may be garbled (no encoding could "
            "be confidently determined)."
        )
        return text