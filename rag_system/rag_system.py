"""
Main Discord RAG System orchestrator.
"""

import os
from typing import Optional, List

from .config import OpenRouterConfig
from .document_processor import DocumentProcessor
from .document_storage import DocumentStorage
from .query_engine import QueryEngine
from .data_models import ProcessedDocument


class DiscordRAGSystem:
    """Main orchestrator for the Discord RAG System."""

    def __init__(self, config: Optional[OpenRouterConfig] = None, storage_path: str = "./filestore"):
        """Initialize the Discord RAG System.

        Args:
            config: OpenRouter configuration. If None, loads from environment.
            storage_path: Path for document storage
        """
        # Load configuration
        if config is None:
            config = OpenRouterConfig.from_env()

        config.validate()
        self.config = config

        # Initialize components
        self.storage = DocumentStorage(storage_path)
        self.document_processor = DocumentProcessor(config)
        self.query_engine = QueryEngine(config, self.storage)

    async def process_file_upload(self, file_path: str, filename: str, user_id: str) -> str:
        """Process an uploaded file and store it for the user.

        Args:
            file_path: Path to the uploaded file
            filename: Original filename
            user_id: Discord user ID

        Returns:
            str: Success message with document summary

        Raises:
            Exception: If processing fails
        """
        try:
            # Process the document
            processed_doc = await self.document_processor.process_document(file_path, filename)

            # Store the processed document
            self.storage.store_document(user_id, processed_doc)

            # Generate success message
            chunk_count = len(processed_doc.chunks)
            word_count = len(processed_doc.original_text.split())

            success_message = (
                f"✅ **Document processed successfully!**\n\n"
                f"**File:** {filename}\n"
                f"**Size:** {word_count:,} words\n"
                f"**Chunks:** {chunk_count} chunks created\n"
                f"**Status:** Ready for querying\n\n"
                f"You can now ask questions about this document using `/ask_docs`!"
            )

            return success_message

        except FileNotFoundError as e:
            raise Exception(f"File not found: {str(e)}")
        except ValueError as e:
            raise Exception(f"Invalid file: {str(e)}")
        except Exception as e:
            raise Exception(f"Processing failed: {str(e)}")

    async def query_documents(self, query: str, user_id: str) -> str:
        """Query the user's documents and generate a response.

        Args:
            query: User's question
            user_id: Discord user ID

        Returns:
            str: Generated response with sources

        Raises:
            Exception: If query processing fails
        """
        try:
            # Check if user has any documents
            user_docs = self.storage.get_user_documents(user_id)
            if not user_docs:
                return (
                    "❌ **No documents found!**\n\n"
                    "You haven't uploaded any documents yet. "
                    "Use `/upload_doc` to upload a text file first."
                )

            # Process the query
            response = await self.query_engine.query(query, user_id)

            # Format the response
            if response.source_chunks:
                confidence_emoji = "🟢" if response.confidence_score > 0.7 else "🟡" if response.confidence_score > 0.4 else "🔴"
                formatted_response = (
                    f"{confidence_emoji} **Query Response** (Confidence: {response.confidence_score:.1%})\n\n"
                    f"{response.answer}"
                )
            else:
                formatted_response = f"❌ {response.answer}"

            return formatted_response

        except ValueError as e:
            raise Exception(f"Invalid query: {str(e)}")
        except Exception as e:
            raise Exception(f"Query failed: {str(e)}")

    def clear_user_documents(self, user_id: str) -> str:
        """Clear all documents for a user.

        Args:
            user_id: Discord user ID

        Returns:
            str: Confirmation message
        """
        try:
            # Get document count before clearing
            user_docs = self.storage.get_user_documents(user_id)
            doc_count = len(user_docs)

            if doc_count == 0:
                return "ℹ️ No documents to clear. Your document collection is already empty."

            # Clear documents
            self.storage.clear_user_documents(user_id)

            return (
                f"✅ **Documents cleared successfully!**\n\n"
                f"Removed {doc_count} document{'s' if doc_count != 1 else ''} from your collection."
            )

        except Exception as e:
            return f"❌ Failed to clear documents: {str(e)}"

    def list_user_documents(self, user_id: str) -> str:
        """List all documents for a user.

        Args:
            user_id: Discord user ID

        Returns:
            str: Formatted list of documents
        """
        try:
            user_docs = self.storage.get_user_documents(user_id)

            if not user_docs:
                return (
                    "📄 **Your Document Collection**\n\n"
                    "No documents uploaded yet.\n"
                    "Use `/upload_doc` to add your first document!"
                )

            # Format document list
            doc_list = ["📄 **Your Document Collection**\n"]

            for i, doc in enumerate(user_docs, 1):
                upload_date = doc.upload_timestamp.strftime("%Y-%m-%d %H:%M")
                chunk_count = len(doc.chunks)

                doc_list.append(
                    f"{i}. **{doc.filename}**\n"
                    f"   • Uploaded: {upload_date}\n"
                    f"   • Chunks: {chunk_count}\n"
                )

            doc_list.append(f"\n**Total:** {len(user_docs)} document{'s' if len(user_docs) != 1 else ''}")

            return "\n".join(doc_list)

        except Exception as e:
            return f"❌ Failed to list documents: {str(e)}"

    def get_system_status(self) -> str:
        """Get system status information.

        Returns:
            str: System status message
        """
        try:
            status_parts = [
                "🤖 **RAG System Status**\n",
                f"• **LLM Model:** {self.config.llm_model}",
                f"• **Embedding Model:** {self.config.embedding_model}",
                f"• **API Base:** {self.config.api_base}",
                f"• **Storage Path:** {self.storage.base_path}",
                "\n✅ System is operational and ready to process documents!"
            ]

            return "\n".join(status_parts)

        except Exception as e:
            return f"❌ System status check failed: {str(e)}"