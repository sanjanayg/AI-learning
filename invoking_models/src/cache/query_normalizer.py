import re

ARTICLES = re.compile(r"\b(a|an|the)\b")

class QueryNormalizer:
    @staticmethod
    def normalize(query: str) -> str:
        if not query:
            return ""

        normalized = query.strip().lower()
        normalized = re.sub(r"\s+", " ", normalized)
        normalized = re.sub(r"\?+", "?", normalized)
        normalized = re.sub(r"!+", "!", normalized)
        normalized = re.sub(r"\.+", ".", normalized)
        normalized = re.sub(r",+", ",", normalized)

        # Remove leading articles that don't affect meaning
        normalized = ARTICLES.sub("", normalized)

        # Collapse spaces introduced by article removal and strip trailing ?
        normalized = re.sub(r"\s+", " ", normalized).strip().rstrip("?")
        normalized = normalized.strip()

        return normalized
