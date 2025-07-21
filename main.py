from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import tempfile
import os

from ocr.ocr_engine import smart_pdf_ocr, extract_text_from_image
from chunking.chunker import chunk_text
from embeddings.embedder import embed_chunks, save_embeddings
from retrieval.query_engine import load_embeddings, search_and_respond

app = FastAPI(title="IQRS API")

# Optional: Enable CORS (useful for frontend calls)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domains for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    documents: str
    questions: list[str]

@app.post("/hackrx/run")
async def process_document(payload: QueryRequest):
    # 🔽 Download the document
    url = payload.documents
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        response = requests.get(url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download document")
        tmp.write(response.content)
        tmp_path = tmp.name

    try:
        # 📄 OCR
        extracted_text = smart_pdf_ocr(tmp_path)

        # 🧩 Chunking
        chunks = chunk_text(extracted_text, chunk_size=500, overlap=50)

        # 🔗 Embeddings
        vectors = embed_chunks(chunks)
        doc_id = os.path.basename(tmp_path)
        save_embeddings(chunks, vectors, doc_id)

        # ❓ Q&A Loop
        index, stored_chunks = load_embeddings(doc_id)
        answers = []
        for question in payload.questions:
            answer, _ = search_and_respond(question, index, stored_chunks)
            answers.append(answer)

        return { "answers": answers }

    finally:
        os.remove(tmp_path)  # Clean up temp file after processing

