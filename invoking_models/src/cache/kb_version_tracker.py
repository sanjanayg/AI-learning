import zlib
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from db.models import ChatFile

class KBVersionTracker:
    @staticmethod
    async def get_version(db: AsyncSession, chat_id: str) -> int:
        """
        Compute a deterministic integer version for the knowledge base of a chat session.
        This is based on the set of files successfully uploaded/indexed.
        If no files exist, returns 0.
        Otherwise, returns a hash of the sorted file IDs, ensuring consistency across restarts.
        """
        result = await db.execute(
            select(ChatFile.file_id)
            .filter(ChatFile.chat_id == chat_id)
            .order_by(ChatFile.file_id)
        )
        file_ids = [row[0] for row in result.all()]
        if not file_ids:
            return 0
        
        # Compute a stable 32-bit hash of the sorted file IDs
        ids_str = ",".join(file_ids)
        # We mask with 0xffffffff to ensure it is a positive 32-bit integer
        return zlib.adler32(ids_str.encode("utf-8")) & 0xffffffff
