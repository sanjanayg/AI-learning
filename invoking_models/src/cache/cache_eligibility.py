import re
from typing import Tuple, List, Callable

class CacheEligibilityService:
    def __init__(self):
        # List of rules that can block caching.
        # Each rule takes normalized query and returns (is_eligible, reason)
        self.rules: List[Callable[[str], Tuple[bool, str]]] = [
            self._check_empty_or_short,
            self._check_personal_financial_patterns,
            self._check_realtime_weather_patterns,
        ]

    def is_eligible(self, query: str) -> Tuple[bool, str]:
        """
        Check if the normalized query is eligible for semantic caching.
        Returns (is_eligible, reason).
        """
        for rule in self.rules:
            eligible, reason = rule(query)
            if not eligible:
                return False, reason
        return True, "Eligible"

    def _check_empty_or_short(self, query: str) -> Tuple[bool, str]:
        if not query or len(query) < 10:
            return False, "Query is too short or empty"
        return True, "Passed"

    def _check_personal_financial_patterns(self, query: str) -> Tuple[bool, str]:
        # Personal info or specific accounts
        personal_patterns = [
            r"\bmy\s+(?:account|balance|ticket|payslip|salary|wallet|profile|card|info|data|documents?)\b",
            r"\b(?:account\s+balance|payslip|salary|ticket|tickets|payslips)\b",
        ]
        for pattern in personal_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return False, f"Query contains personal or transaction-specific keyword matching: '{pattern}'"
        return True, "Passed"

    def _check_realtime_weather_patterns(self, query: str) -> Tuple[bool, str]:
        # Real-time or highly dynamic data
        realtime_patterns = [
            r"\b(?:weather|temperature|forecast)\b",
            r"\b(?:real-time|realtime|live|current|today|right\s+now|now)\b",
            r"\b(?:stock\s+price|market\s+price)\b",
        ]
        for pattern in realtime_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return False, f"Query requests real-time or weather data matching: '{pattern}'"
        return True, "Passed"
