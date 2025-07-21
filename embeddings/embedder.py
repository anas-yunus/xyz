import os
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# Load the embedding model once
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_chunks(chunks):
    """Generate embeddings for a list of text chunks."""
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    return embeddings

def save_embeddings(chunks, vectors, doc_name, store_dir="storage/vector_store"):
    """Save vectors to FAISS index and chunks to pickle file."""
    os.makedirs(store_dir, exist_ok=True)

    # FAISS index
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    # Save index
    index_path = os.path.join(store_dir, f"{doc_name}.index")
    faiss.write_index(index, index_path)

    # Save chunks
    chunks_path = os.path.join(store_dir, f"{doc_name}_chunks.pkl")
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    print(f"✅ Embeddings saved to: {index_path}, {chunks_path}")
