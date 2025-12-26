import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pickle
from pathlib import Path

from src.config import EMBEDDING_MODEL_NAME, EMBEDDINGS_DIR


class EmbeddingModel:
    """
    wrapper for sentence-transformers models
    
    This class handles:
    - Loading embedding models
    - Generating embeddings for single or batch texts
    - Caching embeddings to disk
    - Computing similarity between texts
    """
    
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        """
        Initialize the embedding model
        
        Args:
            model_name: Name of the sentence-transformers model
                       Default: all-MiniLM-L6-v2 (fast, 384 dimensions)
        """
        print(f" Loading embedding model: {model_name}...")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f" Model loaded successfully!")
        print(f"   • Dimensions: {self.embedding_dim}")
        print(f"   • Max sequence length: {self.model.max_seq_length}")
    
    def encode(
        self, 
        texts: Union[str, List[str]], 
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for text(s)
        
        Args:
            texts: Single text string or list of texts
            batch_size: Number of texts to process at once (higher = faster but more RAM)
            show_progress: Whether to show progress bar
            
        Returns:
            numpy array of embeddings
            - Shape: (embedding_dim,) for single text
            - Shape: (n_texts, embedding_dim) for list of texts
        """
        # convert single text to list
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        else:
            single_text = False
        
        # generate embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        # return single embedding if single text was provided
        if single_text:
            return embeddings[0]
        
        return embeddings
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        calculate cosine similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between -1 and 1 (1 = identical meaning)
        """
        emb1 = self.encode(text1)
        emb2 = self.encode(text2)
        
        # cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)
    
    def batch_similarity(self, query: str, texts: List[str]) -> np.ndarray:
        """
        Calculate similarity between one query and multiple texts
        Useful for ranking documents by relevance
        
        Args:
            query: Query text
            texts: List of texts to compare against
            
        Returns:
            Array of similarity scores (same length as texts)
        """
        query_emb = self.encode(query)
        texts_embs = self.encode(texts, show_progress=False)
        
        # Calculate cosine similarity for each text
        similarities = np.dot(texts_embs, query_emb) / (
            np.linalg.norm(texts_embs, axis=1) * np.linalg.norm(query_emb)
        )
        
        return similarities
    
    def save_embeddings(
        self, 
        embeddings: np.ndarray, 
        filename: str,
        metadata: dict = None
    ) -> None:
        """
        Save embeddings to disk
        
        Args:
            embeddings: Numpy array of embeddings
            filename: Name of file (without path, .pkl will be added)
            metadata: Optional dict with additional info (e.g., model name, date)
        """
        filepath = EMBEDDINGS_DIR / f"{filename}.pkl"
        
        data = {
            'embeddings': embeddings,
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'metadata': metadata or {}
        }
        
        print(f"Saving embeddings to {filepath}...")
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        size_mb = filepath.stat().st_size / 1024 / 1024
        print(f"Saved successfully! ({size_mb:.2f} MB)")
    
    def load_embeddings(self, filename: str) -> dict:
        """
        Load embeddings from disk
        
        Args:
            filename: Name of file (without path or extension)
            
        Returns:
            Dict with 'embeddings', 'model_name', 'embedding_dim', 'metadata'
        """
        filepath = EMBEDDINGS_DIR / f"{filename}.pkl"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Embeddings file not found: {filepath}")
        
        print(f"Loading embeddings from {filepath}...")
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Loaded {data['embeddings'].shape[0]} embeddings")
        return data


def generate_document_embeddings(df, text_column: str = 'answer') -> np.ndarray:
    """
    Generate embeddings for all documents in a dataframe
    
    This is a convenience function that:
    1. Loads the embedding model
    2. Generates embeddings for specified column
    3. Saves embeddings to disk
    4. Returns the embeddings array
    
    Args:
        df: Pandas dataframe with text data
        text_column: Name of column containing text to embed
        
    Returns:
        Numpy array of embeddings (shape: n_docs x embedding_dim)
    """
    print(f"\nGenerating embeddings for {len(df)} documents...")
    print(f"   Column: '{text_column}'")
    
    # initialize model
    model = EmbeddingModel()
    
    # get texts
    texts = df[text_column].tolist()
    
    # generate embeddings (with progress bar)
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress=True
    )
    
    # save embeddings
    metadata = {
        'num_documents': len(df),
        'text_column': text_column,
        'avg_text_length': df[text_column].str.len().mean()
    }
    
    model.save_embeddings(
        embeddings,
        filename='document_embeddings',
        metadata=metadata
    )
    
    return embeddings


if __name__ == "__main__":
    # test the embedding model
    print("Testing Embedding Model\n")
    
    # initialize model
    model = EmbeddingModel()
    
    # test 1: Single text embedding
    print("\n" + "="*60)
    print("Test 1: Single Text Embedding")
    print("="*60)
    text = "What are the symptoms of diabetes?"
    embedding = model.encode(text)
    print(f"Text: {text}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"First 10 values: {embedding[:10]}")
    
    # test 2: Similarity between texts
    print("\n" + "="*60)
    print("Test 2: Text Similarity")
    print("="*60)
    text1 = "diabetes symptoms and treatment"
    text2 = "signs and therapy for diabetes"
    text3 = "how to bake a chocolate cake"
    
    sim_12 = model.similarity(text1, text2)
    sim_13 = model.similarity(text1, text3)
    
    print(f"Text 1: {text1}")
    print(f"Text 2: {text2}")
    print(f"Text 3: {text3}")
    print(f"\nSimilarity (1 vs 2): {sim_12:.4f} (high - similar topics)")
    print(f"Similarity (1 vs 3): {sim_13:.4f} (low - different topics)")
    
    # test 3: Batch similarity (ranking)
    print("\n" + "="*60)
    print("Test 3: Ranking Documents by Relevance")
    print("="*60)
    query = "diabetes treatment"
    documents = [
        "Diabetes can be treated with insulin and medication",
        "Common symptoms of diabetes include thirst and fatigue",
        "Heart disease prevention through exercise",
        "Managing blood sugar levels in diabetic patients"
    ]
    
    similarities = model.batch_similarity(query, documents)
    
    # sort by similarity
    ranked_indices = np.argsort(similarities)[::-1]
    
    print(f"Query: {query}\n")
    print("Ranked documents:")
    for i, idx in enumerate(ranked_indices, 1):
        print(f"{i}. [{similarities[idx]:.3f}] {documents[idx]}")