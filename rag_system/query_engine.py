"""
Query engine for processing user queries and generating responses.
"""

from typing import List, Optional
import asyncio

from .config import OpenRouterConfig
from .data_models import QueryResponse, RelevantChunk
from .document_storage import DocumentStorage
from .llama_integration import OpenRouterLLMIntegration


class QueryEngine:
    """Handles user queries and retrieves relevant information."""

    def __init__(self, openrouter_config: OpenRouterConfig, storage: DocumentStorage):
        """Initialize query engine.

        Args:
            openrouter_config: Configuration for OpenRouter API
            storage: Document storage instance
        """
        self.config = openrouter_config
        self.storage = storage
        self.llm_integration = OpenRouterLLMIntegration(openrouter_config)
        self.max_context_chunks = 5
        self.min_similarity_threshold = 0.2

    async def query(self, query_text: str, user_id: str) -> QueryResponse:
        """Process a user query and generate a response.

        Args:
            query_text: User's question or query
            user_id: Discord user ID

        Returns:
            QueryResponse: Generated response with sources

        Raises:
            Exception: If query processing fails
        """
        if not query_text.strip():
            raise ValueError("Query text cannot be empty")

        try:
            # Generate embedding for the query
            query_embedding = await self.llm_integration.generate_single_embedding(query_text)

            # Retrieve relevant chunks
            relevant_chunks = await self._retrieve_relevant_chunks(query_embedding, user_id)

            # Generate response based on retrieved chunks
            if relevant_chunks:
                answer = await self._generate_response(query_text, relevant_chunks)
                confidence_score = self._calculate_confidence_score(relevant_chunks)
            else:
                answer = self._generate_no_results_response(query_text)
                confidence_score = 0.0

            return QueryResponse(
                answer=answer,
                source_chunks=relevant_chunks,
                confidence_score=confidence_score
            )

        except Exception as e:
            raise Exception(f"Failed to process query: {str(e)}")

    async def _retrieve_relevant_chunks(
        self,
        query_embedding: List[float],
        user_id: str
    ) -> List[RelevantChunk]:
        """Retrieve relevant document chunks for a query.

        Args:
            query_embedding: Query embedding vector
            user_id: Discord user ID

        Returns:
            List[RelevantChunk]: Relevant chunks sorted by similarity
        """
        # Search for relevant chunks
        relevant_chunks = self.storage.search_chunks(
            query_embedding=query_embedding,
            user_id=user_id,
            top_k=self.max_context_chunks * 2  # Get more to filter
        )

        # Filter by similarity threshold and limit results
        filtered_chunks = [
            chunk for chunk in relevant_chunks
            if chunk.similarity_score >= self.min_similarity_threshold
        ]

        # Return top chunks
        return filtered_chunks[:self.max_context_chunks]

    async def _generate_response(
        self,
        query: str,
        context_chunks: List[RelevantChunk]
    ) -> str:
        """Generate a response using retrieved context chunks.

        Args:
            query: User's query
            context_chunks: Relevant context chunks

        Returns:
            str: Generated response
        """
        if not context_chunks:
            return self._generate_no_results_response(query)

        # Prepare context from chunks
        context_parts = []
        source_info = []

        for i, relevant_chunk in enumerate(context_chunks, 1):
            chunk = relevant_chunk.chunk
            context_parts.append(f"[Context {i}]: {chunk.contextual_chunk.original_text}")
            source_info.append(f"- {chunk.contextual_chunk.document_filename} (similarity: {relevant_chunk.similarity_score:.2f})")

        context_text = "\n\n".join(context_parts)
        sources_text = "\n".join(source_info)

        # Create prompt for response generation
        response_prompt = f"""Based on the following context from uploaded documents, please answer the user's question. If the context doesn't contain enough information to answer the question, say so clearly.

Context:
{context_text}

Question: {query}

Please provide a helpful and accurate answer based on the context above. If you reference specific information, mention which document it came from."""

        try:
            # Generate response
            response = await self.llm_integration.generate_text(response_prompt)

            # Add source information
            full_response = f"{response}\n\n**Sources:**\n{sources_text}"

            return full_response

        except Exception as e:
            return f"I found relevant information in your documents, but encountered an error generating the response: {str(e)}\n\n**Sources:**\n{sources_text}"

    def _generate_no_results_response(self, query: str) -> str:
        """Generate a response when no relevant documents are found.

        Args:
            query: User's query

        Returns:
            str: No results response
        """
        return (
            f"I couldn't find any relevant information in your uploaded documents to answer: \"{query}\"\n\n"
            "This could mean:\n"
            "• You haven't uploaded any documents yet\n"
            "• Your documents don't contain information related to this question\n"
            "• Try rephrasing your question or using different keywords\n\n"
            "Use `/upload_doc` to add documents or `/list_docs` to see what's available."
        )

    def _calculate_confidence_score(self, relevant_chunks: List[RelevantChunk]) -> float:
        """Calculate confidence score based on retrieved chunks.

        Args:
            relevant_chunks: Retrieved relevant chunks

        Returns:
            float: Confidence score between 0.0 and 1.0
        """
        if not relevant_chunks:
            return 0.0

        # Use average similarity score as confidence
        avg_similarity = sum(chunk.similarity_score for chunk in relevant_chunks) / len(relevant_chunks)

        # Apply some scaling to make confidence more meaningful
        # High similarity (>0.8) = high confidence
        # Medium similarity (0.4-0.8) = medium confidence
        # Low similarity (<0.4) = low confidence
        if avg_similarity >= 0.8:
            confidence = 0.8 + (avg_similarity - 0.8) * 0.5  # Scale 0.8-1.0 to 0.8-0.9
        elif avg_similarity >= 0.4:
            confidence = 0.4 + (avg_similarity - 0.4) * 1.0  # Scale 0.4-0.8 to 0.4-0.8
        else:
            confidence = avg_similarity  # Keep low similarities as-is

        return min(confidence, 1.0)