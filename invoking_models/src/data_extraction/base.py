from abc import ABC, abstractmethod


class BaseLLMProvider(ABC):

    @abstractmethod
    def extract_text_from_image(
        self,
        base64_image: str,
        mime_type: str,
        prompt: str
    ) -> str:
        pass