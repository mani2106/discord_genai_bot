"""
File-based document storage system for the Discord RAG System.
"""

import os
import json
import pickle
from typing import List, Optional, Dict, Any
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .data_models import ProcessedDocument, EmbeddedChunk, RelevantChunk


class DocumentStorage:
    """Simple file-based storage for processed documents and embeddings."""

    def __init__(self, base_path: str = "./filestore"):
        """Initialize document storage.

        Args:
            base_path: Base directory for file storage
        """
        self.base_path = base_path
        self.docs_path = os.path.join(base_path, "docs")
        self.embeddings_path = os.path.join(base_path, "embeds")

        # Ensure directories exist
        os.makedirs(self.docs_path, exist_ok=True)
        os.makedirs(self.embeddings_path, exist_ok=True)

    def store_document(self, user_id: str, document: ProcessedDocument) -> None:
        """Store a processed document for a user.

        Args:
            user_id: Discord user ID
            document: Processed document to store

        Raises:
            Exception: If storage fails
        """
        try:
            # Create user directory
            user_docs_path = os.path.join(self.docs_path, user_id)
            user_embeds_path = os.path.join(self.embeddings_path, user_id)
            os.makedirs(user_docs_path, exist_ok=True)
            os.makedirs(user_embeds_path, exist_ok=True)

            # Generate safe filename
            safe_filename = self._make_safe_filename(document.filename)
            timestamp = document.upload_timestamp.strftime("%Y%m%d_%H%M%S")
            base_name = f"{timestamp}_{safe_filename}"

            # Store document metadata and text
            doc_file = os.path.join(user_docs_path, f"{base_name}.json")
            doc_data = {
                "filename": document.filename,
                "original_text": document.original_text,
                "upload_timestamp": document.upload_timestamp.isoformat(),
                "chunk_count": len(document.chunks)
            }

            with open(doc_file, 'w', encoding='utf-8') as f:
                json.dump(doc_data, f, indent=2, ensure_ascii=False)

            # Store embeddings and chunks
            embed_file = os.path.join(user_embeds_path, f"{base_name}.pkl")
            embed_data = {
                "filename": document.filename,
                "chunks": document.chunks,
                "upload_timestamp": document.upload_timestamp
            }

            with open(embed_file, 'wb') as f:
                pickle.dump(embed_data, f)

        except Exception as e:
            raise Exception(f"Failed to store document {document.filename}: {str(e)}")

    def get_user_documents(self, user_id: str) -> List[ProcessedDocument]:
        """Retrieve all documents for a user.

        Args:
            user_id: Discord user ID

        Returns:
            List[ProcessedDocument]: List of user's documents
        """
        documents = []
        user_embeds_path = os.path.join(self.embeddings_path, user_id)

        if not os.path.exists(user_embeds_path):
            return documents

        try:
            for filename in os.listdir(user_embeds_path):
                if filename.endswith('.pkl'):
                    embed_file = os.path.join(user_embeds_path, filename)

                    with open(embed_file, 'rb') as f:
                        embed_data = pickle.load(f)

                    # Reconstruct ProcessedDocument
                    # For listing, we need at least some text to satisfy validation
                    document = ProcessedDocument(
                        filename=embed_data["filename"],
                        original_text="[Content available - use get_document_by_filename for full text]",
                        chunks=embed_data["chunks"],
                        upload_timestamp=embed_data["upload_timestamp"]
                    )
                    documents.append(document)

        except Exception as e:
            # Log error but don't fail completely
            print(f"Error loading documents for user {user_id}: {str(e)}")

        return documents

    def get_document_by_filename(self, user_id: str, filename: str) -> Optional[ProcessedDocument]:
        """Retrieve a specific document by filename.

        Args:
            user_id: Discord user ID
            filename: Document filename

        Returns:
            Optional[ProcessedDocument]: Document if found, None otherwise
        """
        user_docs_path = os.path.join(self.docs_path, user_id)
        user_embeds_path = os.path.join(self.embeddings_path, user_id)

        if not os.path.exists(user_docs_path) or not os.path.exists(user_embeds_path):
            return None

        try:
            # Find matching files
            safe_filename = self._make_safe_filename(filename)

            doc_file = None
            embed_file = None

            for file in os.listdir(user_docs_path):
                if file.endswith('.json') and safe_filename in file:
                    doc_file = os.path.join(user_docs_path, file)
                    break

            for file in os.listdir(user_embeds_path):
                if file.endswith('.pkl') and safe_filename in file:
                    embed_file = os.path.join(user_embeds_path, file)
                    break

            if not doc_file or not embed_file:
                return None

            # Load document data
            with open(doc_file, 'r', encoding='utf-8') as f:
                doc_data = json.load(f)

            with open(embed_file, 'rb') as f:
                embed_data = pickle.load(f)

            # Reconstruct ProcessedDocument
            document = ProcessedDocument(
                filename=doc_data["filename"],
                original_text=doc_data["original_text"],
                chunks=embed_data["chunks"],
                upload_timestamp=datetime.fromisoformat(doc_data["upload_timestamp"])
            )

            return document

        except Exception as e:
            print(f"Error loading document {filename} for user {user_id}: {str(e)}")
            return None

    def clear_user_documents(self, user_id: str) -> None:
        """Clear all documents for a user.

        Args:
            user_id: Discord user ID
        """
        user_docs_path = os.path.join(self.docs_path, user_id)
        user_embeds_path = os.path.join(self.embeddings_path, user_id)

        # Remove all files in user directories
        for path in [user_docs_path, user_embeds_path]:
            if os.path.exists(path):
                try:
                    for filename in os.listdir(path):
                        file_path = os.path.join(path, filename)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                except Exception as e:
                    print(f"Error clearing documents for user {user_id}: {str(e)}")

    def search_chunks(
        self,
        query_embedding: List[float],
        user_id: str,
        top_k: int = 5
    ) -> List[RelevantChunk]:
        """Search for relevant chunks using semantic similarity.

        Args:
            query_embedding: Query embedding vector
            user_id: Discord user ID
            top_k: Number of top results to return

        Returns:
            List[RelevantChunk]: Ranked list of relevant chunks
        """
        relevant_chunks = []
        documents = self.get_user_documents(user_id)

        if not documents:
            return relevant_chunks

        # Collect all chunks with their embeddings
        all_chunks = []
        all_embeddings = []

        for document in documents:
            for chunk in document.chunks:
                all_chunks.append(chunk)
                all_embeddings.append(chunk.embedding)

        if not all_embeddings:
            return relevant_chunks

        try:
            # Calculate cosine similarities
            query_embedding_array = np.array(query_embedding).reshape(1, -1)
            chunk_embeddings_array = np.array(all_embeddings)

            similarities = cosine_similarity(query_embedding_array, chunk_embeddings_array)[0]

            # Get top-k most similar chunks
            top_indices = np.argsort(similarities)[::-1][:top_k]

            for idx in top_indices:
                similarity_score = float(similarities[idx])
                # Only include chunks with reasonable similarity (> 0.1)
                if similarity_score > 0.1:
                    relevant_chunk = RelevantChunk(
                        chunk=all_chunks[idx],
                        similarity_score=similarity_score
                    )
                    relevant_chunks.append(relevant_chunk)

        except Exception as e:
            print(f"Error searching chunks for user {user_id}: {str(e)}")

        return relevant_chunks

    def _make_safe_filename(self, filename: str) -> str:
        """Create a safe filename for storage.

        Args:
            filename: Original filename

        Returns:
            str: Safe filename for filesystem
        """
        # Remove or replace unsafe characters
        safe_chars = []
        for char in filename:
            if char.isalnum() or char in '.-_':
                safe_chars.append(char)
            else:
                safe_chars.append('_')

        safe_filename = ''.join(safe_chars)

        # Ensure it's not too long
        if len(safe_filename) > 100:
            name, ext = os.path.splitext(safe_filename)
            safe_filename = name[:95] + ext

        return safe_filename