from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self):
        """
        Load cross-encoder model for re-ranking retrieved chunks.
        Scores each (query, chunk) pair for better accuracy.
        """
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, query: str, chunks: list, top_k: int = 3) -> list:
        """
        Re-rank retrieved chunks using cross-encoder scoring.

        Args:
            query: User question
            chunks: List of retrieved document chunks
            top_k: Number of top chunks to return

        Returns:
            Top-k re-ranked chunks
        """
        if not chunks:
            return []

        # Create (query, chunk_text) pairs for scoring
        pairs = [(query, chunk.page_content) for chunk in chunks]

        # Score all pairs
        scores = self.model.predict(pairs)

        # Sort chunks by score (highest first)
        scored_chunks = sorted(
            zip(scores, chunks),
            key=lambda x: x[0],
            reverse=True
        )

        # Return top-k chunks
        return [chunk for _, chunk in scored_chunks[:top_k]]
