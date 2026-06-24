import base64
from fastapi import UploadFile, HTTPException

from config import settings


class FileService:
    PDF_TYPE = "application/pdf"
    TXT_TYPE = "text/plain"

    ALLOWED_IMAGE_TYPES = {
        "image/jpeg",
        "image/png",
        "image/webp",
    }

    ALLOWED_DOCX_TYPES = {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    }

    ALLOWED_TYPES = (
        {PDF_TYPE, TXT_TYPE}
        | ALLOWED_IMAGE_TYPES
        | ALLOWED_DOCX_TYPES
    )

    @staticmethod
    async def read_and_validate(file: UploadFile) -> tuple[bytes, str]:
        if not file.filename:
            raise HTTPException(status_code=400, detail="File name is missing")

        file_bytes = await file.read()

        if not file_bytes:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        max_size = 15 * 1024 * 1024

        if len(file_bytes) > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max allowed size is 15 MB",
            )

        mime_type = file.content_type

        if not mime_type:
            raise HTTPException(status_code=400, detail="Unable to detect file type")

        if mime_type not in FileService.ALLOWED_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {mime_type}",
            )

        return file_bytes, mime_type

    @staticmethod
    def encode_to_base64(file_bytes: bytes) -> str:
        return base64.b64encode(file_bytes).decode("utf-8")