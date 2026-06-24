import re
import uuid
import logging
import tiktoken
from schemas import DocumentChunk, ExtractTextResponse

logger = logging.getLogger(__name__)


class LayoutAwareChunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Standard GPT-4 / Llama tokenizer encoder
        try:
            self.encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.encoder = tiktoken.get_encoding("gpt-4")

    def _token_count(self, text: str) -> int:
        return len(self.encoder.encode(text))

    def _split_sentences(self, text: str) -> list[str]:
        # Split by sentence boundaries, preserving punctuation
        sentence_endings = re.compile(r'(?<=[.!?]) +')
        return [s.strip() for s in sentence_endings.split(text) if s.strip()]

    def _parse_blocks(self, text: str) -> list[dict]:
        """
        Splits a page's text into table blocks and paragraph blocks.
        Identifies tables by looking for consecutive lines containing pipe (|) separators.
        """
        lines = text.split("\n")
        blocks = []
        in_table = False
        current_table_lines = []
        current_text_lines = []

        def flush_text():
            if current_text_lines:
                text_content = "\n".join(current_text_lines).strip()
                if text_content:
                    blocks.append({"type": "paragraph", "content": text_content})
                current_text_lines.clear()

        def flush_table():
            if current_table_lines:
                table_content = "\n".join(current_table_lines).strip()
                if table_content:
                    blocks.append({"type": "table", "content": table_content})
                current_table_lines.clear()

        for line in lines:
            # Check if line contains a pipe (used in Markdown and DOCX table formats)
            is_table_row = "|" in line
            
            if is_table_row:
                flush_text()
                in_table = True
                current_table_lines.append(line)
            else:
                if in_table:
                    flush_table()
                    in_table = False
                current_text_lines.append(line)

        flush_text()
        flush_table()
        return blocks

    def chunk_document(
        self, 
        extracted_doc: ExtractTextResponse, 
        chat_id: str, 
        file_id: str
    ) -> list[DocumentChunk]:
        """
        Processes an Extracted Document and returns a list of DocumentChunks.
        Maintains structural layouts (tables vs paragraphs) and propagates complete metadata.
        """
        chunks: list[DocumentChunk] = []
        file_name = getattr(extracted_doc, "file_name", "unknown_file")
        pages = getattr(extracted_doc, "pages", [])

        if not pages:
            # Fallback if pages list is missing but raw extracted_text is present
            raw_text = getattr(extracted_doc, "extracted_text", "")
            from schemas import ExtractedPage
            pages = [ExtractedPage(page_number=1, text=raw_text, extraction_method=extracted_doc.extraction_method)]

        for page in pages:
            page_number = page.page_number
            method = page.extraction_method.lower()
            
            # Determine base element type for text on this page
            # If the page was OCR-ed, non-table blocks represent OCR text
            is_ocr_page = "ocr" in method or "vision" in method
            base_text_type = "image_ocr" if is_ocr_page else "text_paragraph"

            # Parse page into tables and paragraphs
            blocks = self._parse_blocks(page.text)
            
            # Temporary accumulator for building text chunks
            accumulated_text = []
            accumulated_tokens = 0

            def flush_accumulator():
                nonlocal accumulated_text, accumulated_tokens
                if accumulated_text:
                    content = "\n\n".join(accumulated_text)
                    chunk_id = f"{file_id}_p{page_number}_{str(uuid.uuid4())[:8]}"
                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            chat_id=chat_id,
                            file_id=file_id,
                            file_name=file_name,
                            page_number=page_number,
                            element_type=base_text_type,
                            content=content,
                            token_count=accumulated_tokens
                        )
                    )
                    accumulated_text = []
                    accumulated_tokens = 0

            for block in blocks:
                block_type = block["type"]
                content = block["content"]
                block_tokens = self._token_count(content)

                if block_type == "table":
                    # Flush text accumulator before inserting a table chunk to preserve ordering
                    flush_accumulator()
                    
                    # Tables are treated as a single, layout-preserved chunk of type 'structural_table'
                    chunk_id = f"{file_id}_p{page_number}_tbl_{str(uuid.uuid4())[:8]}"
                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            chat_id=chat_id,
                            file_id=file_id,
                            file_name=file_name,
                            page_number=page_number,
                            element_type="structural_table",
                            content=content,
                            token_count=block_tokens
                        )
                    )
                else:
                    # Block is a paragraph. Check if it fits in current accumulator
                    if block_tokens > self.chunk_size:
                        # If a single paragraph is too large, split it into sentences and chunk them
                        flush_accumulator()
                        sentences = self._split_sentences(content)
                        
                        temp_chunk = []
                        temp_tokens = 0
                        
                        for sentence in sentences:
                            sent_tokens = self._token_count(sentence)
                            if temp_tokens + sent_tokens > self.chunk_size:
                                if temp_chunk:
                                    sent_content = " ".join(temp_chunk)
                                    chunk_id = f"{file_id}_p{page_number}_{str(uuid.uuid4())[:8]}"
                                    chunks.append(
                                        DocumentChunk(
                                            chunk_id=chunk_id,
                                            chat_id=chat_id,
                                            file_id=file_id,
                                            file_name=file_name,
                                            page_number=page_number,
                                            element_type=base_text_type,
                                            content=sent_content,
                                            token_count=temp_tokens
                                        )
                                    )
                                temp_chunk = [sentence]
                                temp_tokens = sent_tokens
                            else:
                                temp_chunk.append(sentence)
                                temp_tokens += sent_tokens
                        
                        if temp_chunk:
                            accumulated_text = [" ".join(temp_chunk)]
                            accumulated_tokens = temp_tokens
                    else:
                        # Paragraph fits or is smaller than chunk size.
                        # Check if adding it to accumulator exceeds chunk size.
                        if accumulated_tokens + block_tokens > self.chunk_size:
                            flush_accumulator()
                        
                        accumulated_text.append(content)
                        accumulated_tokens += block_tokens

            # Flush any remaining text at the end of the page
            flush_accumulator()

        logger.info(
            "Document chunked successfully: file=%s chunks=%d", 
            file_name, len(chunks)
        )
        return chunks


# ---------- Legacy Chunker Classes for Backward Compatibility ----------

class TokenChunkingService:
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 512, chunk_overlap: int = 50) -> list[str]:
        from langchain_text_splitters import TokenTextSplitter
        splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return splitter.split_text(text)


class SemanticChunker:
    def __init__(self, similarity_threshold: float = 0.65, min_chunk_sentences: int = 2):
        self.similarity_threshold = similarity_threshold
        self.min_chunk_sentences = min_chunk_sentences
        self._model = None

    @property
    def model(self):
        # Lazy-load to avoid slowing down startup if not used
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._model

    def chunk_text(self, text: str) -> list[str]:
        import numpy as np
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        if not sentences:
            return [text]

        embeddings = self.model.encode(sentences)
        chunks, current = [], [sentences[0]]

        for i in range(1, len(sentences)):
            sim = np.dot(embeddings[i - 1], embeddings[i]) / (
                np.linalg.norm(embeddings[i - 1]) * np.linalg.norm(embeddings[i]) + 1e-8
            )
            if sim < self.similarity_threshold and len(current) >= self.min_chunk_sentences:
                chunks.append(". ".join(current) + ".")
                current = []
            current.append(sentences[i])

        if current:
            chunks.append(". ".join(current) + ".")

        return chunks