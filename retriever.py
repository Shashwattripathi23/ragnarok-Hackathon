import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Minimum cosine similarity for a chunk to be included in context.
# Chunks below this threshold are considered irrelevant and excluded.
# This prevents noise from poisoning the LLM context and causing hallucinations.
_MIN_SIMILARITY = 0.30


class VectorStore:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initialize the VectorStore with a sentence-transformer model.
        all-MiniLM-L6-v2 is small, fast, and good for general text.
        """
        # This will download the model weights on the first run (~90MB)
        self.model = SentenceTransformer(model_name)
        self.chunks = []
        self.embeddings = None

    def index(self, chunks):
        """
        Compute embeddings for a list of document chunks and store them.
        chunks: list of dicts [{"text": "...", "metadata": {...}}]
        """
        self.chunks = chunks
        if not chunks:
            self.embeddings = np.array([])
            return

        texts = [chunk["text"] for chunk in chunks]
        # Encode returns a numpy array of shape (len(texts), embedding_dim)
        self.embeddings = self.model.encode(texts, show_progress_bar=True)

    def search(self, query, top_k=5, file_filters=None, min_similarity=_MIN_SIMILARITY):
        """
        Search for the most relevant chunks given a query.

        Parameters
        ----------
        query : str
            The search query string.
        top_k : int
            Maximum number of results to return.
        file_filters : list[str] | None
            List of filenames to restrict the search to. None = no filter.
        min_similarity : float
            Cosine similarity threshold (0–1). Chunks below this score are
            excluded from results to prevent irrelevant context from reaching
            the LLM. Default: 0.30.

        Returns
        -------
        list[dict]
            Matching chunks in descending similarity order, each with
            {"text": ..., "metadata": ..., "score": float}.
        """
        if self.embeddings is None or len(self.embeddings) == 0:
            return []

        # Get query embedding
        query_embedding = self.model.encode([query])

        # Compute cosine similarities between query and all indexed chunks
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]

        # Get the sorted indices from highest similarity to lowest
        sorted_indices = np.argsort(similarities)[::-1]

        results = []
        for idx in sorted_indices:
            # ── Hard stop: skip chunks below the relevance threshold ──
            score = float(similarities[idx])
            if score < min_similarity:
                break  # sorted descending, so all remaining are also below threshold

            chunk = self.chunks[idx]

            # Apply file filter if specified
            if file_filters is not None:
                if chunk["metadata"]["file"] not in file_filters:
                    continue

            # Attach the similarity score for potential downstream use / debugging
            results.append({**chunk, "score": round(score, 4)})

            if len(results) >= top_k:
                break

        return results
