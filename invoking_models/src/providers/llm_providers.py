from groq import Groq
from config import settings
from data_extraction.base import BaseLLMProvider


class GroqProvider(BaseLLMProvider):

    def __init__(self):
        self.api_key = settings.GROQ_API_KEY
        self.model = settings.GROQ_MODEL
        self.versatile_model=settings.GROQ_MODEL_VERSATILE

        if not self.api_key:
            raise ValueError("Groq API key is missing.")

        self.client = Groq(api_key=self.api_key)

    def extract_text_from_image(
        self,
        base64_image: str,
        mime_type: str,
        prompt: str
    ) -> str:

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime_type};base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                ],
                temperature=0
            )
            
            return completion.choices[0].message.content

        except Exception as e:
            raise ValueError(f"Groq image extraction failed: {str(e)}")