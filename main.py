from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import tempfile
import os

from ocr.ocr_engine import smart_pdf_ocr
from chunking.chunker import chunk_text
from embeddings.embedder import embed_chunks, save_embeddings
from retrieval.query_engine import load_embeddings, search_and_respond

# --- FastAPI App Initialization ---
app = FastAPI(title="IQRS - HackRx API")

# CORS Middleware (if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Authorization Token ---
TEAM_TOKEN = "ada5416b78f70c66bd0b4d0d6c387422749b71a07e44411c64d14e30ffb7a4e5"

# --- Pydantic Request Model ---
class QueryRequest(BaseModel):
    documents: str
    questions: list[str]

# --- Endpoint ---
@app.post("/hackrx/run")
async def run_document_pipeline(
    payload: QueryRequest,
    authorization: str = Header(..., alias="Authorization")
):
    # Validate Bearer Token
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization header format.")
    token = authorization.split(" ")[1]
    if token != TEAM_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized access.")

    # Step 1: Download the document
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        response = requests.get(payload.documents)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download document.")
        tmp.write(response.content)
        tmp_path = tmp.name

    try:
        # Step 2: Run OCR
        extracted_text = smart_pdf_ocr(tmp_path)

        # Step 3: Chunk text
        chunks = chunk_text(extracted_text, chunk_size=500, overlap=50)

        # Step 4: Embed and Save
        vectors = embed_chunks(chunks)
        doc_id = os.path.basename(tmp_path)
        save_embeddings(chunks, vectors, doc_id)

        # Step 5: Answer Questions
        index, stored_chunks = load_embeddings(doc_id)
        answers = []
        for q in payload.questions:
            answer, _ = search_and_respond(q, index, stored_chunks)
            answers.append(answer)

        return { "answers": answers }

    finally:
        os.remove(tmp_path)  # Clean up temp file
