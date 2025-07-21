# Chunking logic here
def chunk_text(text, chunk_size=500, overlap=50):
    """
    Splits text into overlapping chunks.
    Chunk 1: words 0-499
    Chunk 2: words 450-949
    Chunk 3: words 900-1399, etc.

    :param text: The full input text (string)
    :param chunk_size: Number of words per chunk
    :param overlap: Number of words to overlap between chunks
    :return: List of chunk strings
    """
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        start += chunk_size - overlap  # Slide forward

    return chunks
