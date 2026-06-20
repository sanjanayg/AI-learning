import base64
from fastapi import UploadFile, HTTPException


class FileService:
    ALLOWED_IMAGE_TYPES = {
        "image/jpeg",
        "image/png",
        "image/webp"
    }

    PDF_TYPE = "application/pdf"

    MAX_FILE_SIZE_MB = 10

    @classmethod
    async def read_and_validate(cls, file: UploadFile) -> tuple[bytes, str]:
        content = await file.read()

        if not content:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        size_mb = len(content) / (1024 * 1024)

        if size_mb > cls.MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds {cls.MAX_FILE_SIZE_MB} MB"
            )

        if file.content_type not in cls.ALLOWED_IMAGE_TYPES and file.content_type != cls.PDF_TYPE:
            raise HTTPException(
                status_code=400,
                detail="Only JPG, PNG, WEBP, and PDF files are supported"
            )

        return content, file.content_type

    @staticmethod
    def encode_to_base64(file_bytes: bytes) -> str:
        return base64.b64encode(file_bytes).decode("utf-8")