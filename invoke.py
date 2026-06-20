import requests


class LLMClient:
    def __init__(self, provider: str, api_key: str):
        self.provider = provider.lower()
        self.api_key = api_key

    def invoke(self, model: str, prompt: str):
        if self.provider == "gemini":
            return self._gemini(model, prompt)

        if self.provider == "openai":
            return self._openai(model, prompt)

        if self.provider == "anthropic":
            return self._anthropic(model, prompt)

        if self.provider == "groq":
            return self._groq(model, prompt)

        raise ValueError("Unsupported provider")

    def _gemini(self, model, prompt):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

        headers = {
            "api-key": self.API_KEY,
            "Content-Type": "application/json"
        }

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ]
        }

        return self._post(url, headers, payload)

    def _openai(self, model, prompt):
        url = "https://api.openai.com/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        return self._post(url, headers, payload)

    def _anthropic(self, model, prompt):
        url = "https://api.anthropic.com/v1/messages"

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        return self._post(url, headers, payload)

    def _groq(self, model, prompt):
        url = "https://api.groq.com/openai/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        return self._post(url, headers, payload)

    def _post(self, url, headers, payload):
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=60
        )

        if response.status_code != 200:
            raise Exception(f"API Error {response.status_code}: {response.text}")

        return response.json()