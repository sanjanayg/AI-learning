import base64
from io import BytesIO
from PIL import Image
from fastapi import UploadFile

from config import settings



ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp"
}


class ImageService:

    @staticmethod
    async def validate_and_encode_image(file: UploadFile) -> tuple[str, str]:
        if file.content_type not in ALLOWED_IMAGE_TYPES:
            raise ValueError("Only JPEG, PNG, and WEBP images are allowed.")

        image_bytes = await file.read()

        max_size_bytes = settings.MAX_IMAGE_SIZE_MB * 1024 * 1024  #10,485,760 bytes
        if len(image_bytes) > max_size_bytes:
            raise ValueError(
                f"Image size should not exceed {settings.MAX_IMAGE_SIZE_MB} MB."
            )

        try:
            image = Image.open(BytesIO(image_bytes))
            image.verify()
        except Exception:
            raise ValueError("Uploaded file is not a valid image.")

        encoded_image = base64.b64encode(image_bytes).decode("utf-8")

        return encoded_image, file.content_type