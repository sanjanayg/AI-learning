from services.extraction_service import ExtractionService
from rag.chunking import TokenChunkingService
import tiktoken
from rag.chunking import SemanticChunker

encoding = tiktoken.get_encoding("cl100k_base")

service = ExtractionService()
class RAGPipelineService:

    @staticmethod
    async def extract_and_chunk(file):
        
        extraction_response = await service.extract_text_from_file(file)
        chunks = TokenChunkingService.chunk_text(
            text=extraction_response.extracted_text,
            chunk_size=512,
            chunk_overlap=50
        )
        return {
            "success": True,
            "file_type": extraction_response.file_type,
            "extraction_method": extraction_response.extraction_method,
            "total_chunks": len(chunks),
            "chunks": chunks
        }
    async def semantic_chunking(self, file):
        extraction_response = await service.extract_text_from_file(file)
        print("the extraction",extraction_response)
        chunker = SemanticChunker(similarity_threshold=0.65, min_chunk_sentences=2)

        chunks = chunker.chunk_text(extraction_response.extracted_text)

        result = []

        for i, chunk in enumerate(chunks, start=1):
            token_count = len(encoding.encode(chunk))

            result.append({
                "chunk_no": i,
                "text": chunk,
                "tokens": token_count
            })

        return {
            "total_chunks": len(chunks),
            "chunks": result
        }