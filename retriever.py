import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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

    def search(self, query, top_k=5, file_filters=None):
        """
        Search for the most relevant chunks given a query.
        file_filters: list of filenames to restrict the search to.
        Returns a list of matching chunks.
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
            chunk = self.chunks[idx]
            
            # Apply file filter if specified
            if file_filters is not None:
                if chunk["metadata"]["file"] not in file_filters:
                    continue
                    
            results.append(chunk)
            if len(results) >= top_k:
                break

        return results
