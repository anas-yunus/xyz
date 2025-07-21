import streamlit as st
import os
from ocr.ocr_engine import smart_pdf_ocr, extract_text_from_image
from chunking.chunker import chunk_text
from embeddings.embedder import embed_chunks, save_embeddings
from retrieval.query_engine import load_embeddings, search_and_respond

# Set up directories
DOCS_DIR = "documents"
os.makedirs(DOCS_DIR, exist_ok=True)

st.set_page_config(page_title="IQRS - Document OCR", layout="centered")
st.title("📄 IQRS Document Uploader + QA")

uploaded_file = st.file_uploader("Upload a PDF or Image", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    file_path = os.path.join(DOCS_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"📁 Uploaded: {uploaded_file.name}")

    # Step 1: OCR
    st.info("🔍 Running OCR...")
    if uploaded_file.name.endswith(".pdf"):
        extracted_text = smart_pdf_ocr(file_path)
    else:
        extracted_text = extract_text_from_image(file_path)

    st.success("✅ Text Extracted:")
    st.text_area("📜 OCR Output", extracted_text, height=300)

    # Step 2: Chunking
    st.subheader("🧩 Overlapping Chunks")
    chunks = chunk_text(extracted_text, chunk_size=500, overlap=50)
    st.write(f"Total Chunks: {len(chunks)}")

    for idx, chunk in enumerate(chunks[:5]):  # Show first 5 chunks
        st.markdown(f"**Chunk {idx + 1}:**")
        st.text_area(f"chunk_{idx}", chunk, height=200)

    # Step 3: Embedding
    if st.button("🔗 Generate & Save Embeddings"):
        st.info("🔄 Generating embeddings and saving to FAISS...")
        vectors = embed_chunks(chunks)
        save_embeddings(chunks, vectors, uploaded_file.name)
        st.success("✅ Embeddings saved successfully!")

    # Step 4: QA Input
    st.subheader("❓ Ask a Question About This Document")

    question = st.text_input("Enter your question")

    index_path = f"storage/vector_store/{uploaded_file.name}.index"
    chunks_path = f"storage/vector_store/{uploaded_file.name}_chunks.pkl"

    if question:
        if os.path.exists(index_path) and os.path.exists(chunks_path):
            st.info("🔍 Retrieving context and generating answer...")
            index, stored_chunks = load_embeddings(uploaded_file.name)
            answer, used_chunks = search_and_respond(question, index, stored_chunks)

            st.success("💡 Answer:")
            st.markdown(answer)

            with st.expander("📂 Context Chunks Used"):
                for i, chunk in enumerate(used_chunks):
                    st.markdown(f"**Chunk {i + 1}:**")
                    st.text_area(f"context_{i}", chunk, height=150)
        else:
            st.warning("⚠️ Please generate embeddings before asking a question.")
