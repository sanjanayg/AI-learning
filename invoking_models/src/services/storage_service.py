import os
import logging
from config import settings

logger = logging.getLogger(__name__)


class StorageService:
    """Persists raw uploaded file bytes to local disk and reads them back for the worker."""

    @staticmethod
    def save(file_id: str, file_bytes: bytes) -> str:
        """
        Saves file bytes under FILE_STORAGE_PATH/{file_id}.
        Returns the absolute storage path stored in the DB / SQS message.
        """
        os.makedirs(settings.FILE_STORAGE_PATH, exist_ok=True)
        path = os.path.join(settings.FILE_STORAGE_PATH, file_id)
        with open(path, "wb") as f:
            f.write(file_bytes)
        logger.info("Saved file to storage: %s", path)
        return path

    @staticmethod
    def load(storage_path: str) -> bytes:
        """Reads file bytes back from disk for worker processing."""
        with open(storage_path, "rb") as f:
            return f.read()

    @staticmethod
    def delete(storage_path: str) -> None:
        """Removes the raw file after successful processing to free disk space."""
        try:
            os.remove(storage_path)
            logger.info("Deleted processed file: %s", storage_path)
        except FileNotFoundError:
            pass
