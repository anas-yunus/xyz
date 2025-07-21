import os
import numpy as np
import faiss
import pickle
import requests
import json
from sentence_transformers import SentenceTransformer

# Load local embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Proxy base URL for chat only
CHAT_BASE_URL = "https://bfhldevapigw.healthrx.co.in/sp-gw/api/openai/v1"
API_KEY = os.getenv("OPENAI_API_KEY") or "sk-spgw-api01-7725518cecd9cb0663448e4489e7693f"

HEADERS = {
    "Content-Type": "application/json",
    "x-subscription-key": API_KEY
}

def load_embeddings(doc_name, store_dir="storage/vector_store"):
    index_path = os.path.join(store_dir, f"{doc_name}.index")
    chunks_path = os.path.join(store_dir, f"{doc_name}_chunks.pkl")

    index = faiss.read_index(index_path)
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    return index, chunks

# ✅ Use local embedder instead of broken proxy
def get_embedding(text):
    embedding = embedder.encode([text], convert_to_numpy=True)
    return embedding.reshape(1, -1)

def get_completion(prompt, model="gpt-4o"):
    url = f"{CHAT_BASE_URL}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    response = requests.post(url, headers=HEADERS, data=json.dumps(payload))

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"Chat API error: {response.status_code} - {response.text}")

def search_and_respond(query, index, chunks, top_k=3):
    query_vec = get_embedding(query)
    D, I = index.search(query_vec, top_k)
    selected_chunks = [chunks[i] for i in I[0]]

    context = "\n\n---\n\n".join(selected_chunks)
    prompt = f"""Answer the question using only the following context:\n\n{context}\n\nQuestion: {query}\nAnswer:"""

    response = get_completion(prompt)
    return response, selected_chunks
