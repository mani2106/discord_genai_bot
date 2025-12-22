"""
Document processing with contextual retrieval for the Discord RAG System.
"""

import os
from typing import List, Optional
from datetime import datetime

from .config import OpenRouterConfig
from .data_models import ContextualChunk, EmbeddedChunk, ProcessedDocument
from .llama_integration import OpenRouterLLMIntegration


class DocumentProcessor:
    """Processes documents using contextual chunking and embedding generation."""

    def __init__(self, openrouter_config: OpenRouterConfig):
        """Initialize document processor.

        Args:
            openrouter_config: Configuration for OpenRouter API
        """
        self.config = openrouter_config
        self.llm_integration = OpenRouterLLMIntegration(openrouter_config)
        self.chunk_size = 512
        self.chunk_overlap = 50

    async def process_document(self, file_path: str, filename: str) -> ProcessedDocument:
        """Process a document file into contextual chunks with embeddings.

        Args:
            file_path: Path to the document file
            filename: Name of the document file

        Returns:
            ProcessedDocument: Processed document with embedded chunks

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not a supported text format
            Exception: If processing fails
        """
        # Validate file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read and validate file content
        text_content = self._read_text_file(file_path, filename)

        # Chunk the document
        text_chunks = self._chunk_document(text_content)

        # Generate contextual chunks
        contextual_chunks = await self._generate_contextual_chunks(
            text_chunks, filename, text_content
        )

        # Generate embeddings for contextual chunks
        embedded_chunks = await self._embed_chunks(contextual_chunks)

        return ProcessedDocument(
            filename=filename,
            original_text=text_content,
            chunks=embedded_chunks,
            upload_timestamp=datetime.now()
        )

    def _read_text_file(self, file_path: str, filename: str) -> str:
        """Read and validate a text file.

        Args:
            file_path: Path to the file
            filename: Name of the file

        Returns:
            str: File content as text

        Raises:
            ValueError: If file is not a supported text format or is empty
        """
        # Check file extension
        supported_extensions = {'.txt', '.md', '.markdown', '.text'}
        file_ext = os.path.splitext(filename)[1].lower()

        if file_ext not in supported_extensions:
            raise ValueError(
                f"Unsupported file format: {file_ext}. "
                f"Supported formats: {', '.join(supported_extensions)}"
            )

        # Read file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
            except Exception as e:
                raise ValueError(f"Failed to read file {filename}: {str(e)}")

        # Validate content
        if not content.strip():
            raise ValueError(f"File {filename} is empty or contains only whitespace")

        return content

    def _chunk_document(self, text: str) -> List[str]:
        """Split document into manageable chunks.

        Args:
            text: Document text to chunk

        Returns:
            List[str]: List of text chunks
        """
        if not text.strip():
            return []

        chunks = []
        start = 0

        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size

            # If this isn't the last chunk, try to break at a sentence or word boundary
            if end < len(text):
                # Look for sentence boundary (. ! ?) within overlap distance
                sentence_end = -1
                for i in range(end - self.chunk_overlap, end):
                    if i >= 0 and i < len(text) and text[i] in '.!?':
                        # Check if next character is whitespace or end of text
                        if i + 1 >= len(text) or text[i + 1].isspace():
                            sentence_end = i + 1
                            break

                # If no sentence boundary found, look for word boundary
                if sentence_end == -1:
                    word_end = -1
                    for i in range(end - 1, end - self.chunk_overlap, -1):
                        if i >= 0 and i < len(text) and text[i].isspace():
                            word_end = i
                            break

                    if word_end != -1:
                        end = word_end
                else:
                    end = sentence_end

            # Extract chunk
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start position with overlap
            start = max(start + 1, end - self.chunk_overlap)

        return chunks

    async def _generate_contextual_chunks(
        self,
        chunks: List[str],
        filename: str,
        full_document: str
    ) -> List[ContextualChunk]:
        """Generate contextual information for each chunk.

        Args:
            chunks: List of text chunks
            filename: Document filename
            full_document: Complete document text for context

        Returns:
            List[ContextualChunk]: Chunks with contextual information
        """
        contextual_chunks = []

        # Create document context for prompt
        document_context = f"Document: {filename}\nType: Text document"
        if len(full_document) > 2000:
            document_summary = full_document[:2000] + "..."
        else:
            document_summary = full_document

        for i, chunk in enumerate(chunks):
            try:
                # Generate context using OpenRouter LLM
                context_prompt = f"""<document>
{document_summary}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the context, nothing else."""

                context = await self.llm_integration.generate_text(context_prompt)
                context = context.strip()

                # Create contextual text by prepending context to original chunk
                contextual_text = f"{context} {chunk}"

                contextual_chunk = ContextualChunk(
                    original_text=chunk,
                    contextual_text=contextual_text,
                    chunk_index=i,
                    document_filename=filename
                )

                contextual_chunks.append(contextual_chunk)

            except Exception as e:
                # If context generation fails, use original chunk with basic context
                basic_context = f"This chunk is from the document '{filename}'."
                contextual_text = f"{basic_context} {chunk}"

                contextual_chunk = ContextualChunk(
                    original_text=chunk,
                    contextual_text=contextual_text,
                    chunk_index=i,
                    document_filename=filename
                )

                contextual_chunks.append(contextual_chunk)

        return contextual_chunks

    async def _embed_chunks(self, contextual_chunks: List[ContextualChunk]) -> List[EmbeddedChunk]:
        """Generate embeddings for contextual chunks.

        Args:
            contextual_chunks: List of contextual chunks

        Returns:
            List[EmbeddedChunk]: Chunks with embeddings
        """
        if not contextual_chunks:
            return []

        # Extract contextual texts for batch embedding
        contextual_texts = [chunk.contextual_text for chunk in contextual_chunks]

        try:
            # Generate embeddings in batch
            embeddings = await self.llm_integration.generate_embeddings(contextual_texts)

            # Create embedded chunks
            embedded_chunks = []
            for chunk, embedding in zip(contextual_chunks, embeddings):
                embedded_chunk = EmbeddedChunk(
                    contextual_chunk=chunk,
                    embedding=embedding
                )
                embedded_chunks.append(embedded_chunk)

            return embedded_chunks

        except Exception as e:
            raise Exception(f"Failed to generate embeddings for chunks: {str(e)}")