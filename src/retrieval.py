"""
Retrieval Module — FAISS-based Similar Thread Search
=====================================================
Builds a FAISS index over customer messages and retrieves the most similar
historical threads for grounding reply generation.

Usage:
    from src.retrieval import ThreadRetriever
    retriever = ThreadRetriever()
    retriever.build_index()
    similar = retriever.search("my iPhone won't charge", k=5)
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

# ─── Config ────────────────────────────────────────────────────────────────────

EMBEDDINGS_FILE = Path("data/processed/customer_embeddings.npy")
MESSAGES_FILE = Path("data/processed/customer_messages.json")
THREADS_FILE = Path("data/processed/apple_threads.jsonl")
INDEX_FILE = Path("data/processed/faiss_index.bin")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class ThreadRetriever:
    """
    Retrieves the most similar historical support threads given a new
    customer message. Uses FAISS for fast nearest-neighbor search over
    sentence-transformer embeddings.
    
    Falls back to cosine similarity with numpy if FAISS is not installed.
    """
    
    def __init__(self):
        self.model = None
        self.index = None
        self.messages = None
        self.threads = None
        self.embeddings = None
        self._thread_lookup = None
    
    def build_index(self):
        """Build or load the search index."""
        print("Building retrieval index...")
        
        # Load embeddings and messages
        self.embeddings = np.load(EMBEDDINGS_FILE).astype('float32')
        with open(MESSAGES_FILE) as f:
            self.messages = json.load(f)
        
        # Load full threads for context
        self.threads = []
        self._thread_lookup = {}
        with open(THREADS_FILE) as f:
            for line in f:
                thread = json.loads(line)
                self.threads.append(thread)
                self._thread_lookup[thread['thread_id']] = thread
        
        # Build FAISS index
        dim = self.embeddings.shape[1]
        
        if HAS_FAISS:
            self.index = faiss.IndexFlatIP(dim)  # Inner product (cosine after normalization)
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(self.embeddings)
            self.index.add(self.embeddings)
            print(f"  FAISS index built: {self.index.ntotal} vectors, dim={dim}")
        else:
            # Normalize for cosine similarity
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            self.embeddings = self.embeddings / (norms + 1e-8)
            print(f"  NumPy fallback index: {len(self.embeddings)} vectors, dim={dim}")
        
        # Load embedding model
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"  Embedding model loaded: {EMBEDDING_MODEL}")
    
    def search(self, query: str, k: int = 5) -> list[dict]:
        """
        Search for the most similar threads given a query message.
        
        Args:
            query: Customer message text
            k: Number of results to return
            
        Returns:
            List of dicts with keys: thread_id, customer_message, agent_response,
            similarity_score, full_thread
        """
        if self.model is None:
            self.build_index()
        
        # Embed query
        query_embedding = self.model.encode([query]).astype('float32')
        
        if HAS_FAISS:
            faiss.normalize_L2(query_embedding)
            scores, indices = self.index.search(query_embedding, k)
            scores = scores[0]
            indices = indices[0]
        else:
            # Cosine similarity via dot product (embeddings are normalized)
            query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
            similarities = np.dot(self.embeddings, query_norm.T).squeeze()
            indices = np.argsort(similarities)[::-1][:k]
            scores = similarities[indices]
        
        results = []
        for idx, score in zip(indices, scores):
            idx = int(idx)
            if idx < 0 or idx >= len(self.messages):
                continue
            
            msg = self.messages[idx]
            thread_id = msg.get('thread_id', '')
            full_thread = self._thread_lookup.get(thread_id, {})
            
            results.append({
                "thread_id": thread_id,
                "customer_message": msg['text'],
                "agent_response": msg.get('agent_response', ''),
                "similarity_score": float(score),
                "full_thread": full_thread,
            })
        
        return results
    
    def get_thread(self, thread_id: str) -> dict:
        """Get a specific thread by ID."""
        if self._thread_lookup is None:
            self.build_index()
        return self._thread_lookup.get(thread_id, {})


# Quick test
if __name__ == "__main__":
    retriever = ThreadRetriever()
    retriever.build_index()
    
    test_queries = [
        "My iPhone screen is cracked, what should I do?",
        "I can't sign into my Apple ID",
        "I was charged twice for an app purchase",
        "How do I update my iPhone to the latest iOS?",
        "My AirPods won't connect to my Mac",
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = retriever.search(query, k=3)
        for i, r in enumerate(results):
            print(f"  [{i+1}] (score={r['similarity_score']:.3f}) {r['customer_message'][:80]}...")
            print(f"       Agent: {r['agent_response'][:80]}...")
