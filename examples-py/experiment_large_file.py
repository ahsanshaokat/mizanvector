import os
from mizanvector import HFEmbedder, MizanMemoryStore


def load_large_file(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def chunk_text(text, chunk_size=800, overlap=150):
    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]

        chunks.append(chunk)
        start += (chunk_size - overlap)

    return chunks

def main():
    # Path to your large file
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(BASE_DIR, "testdata.txt")
    text = load_large_file(path)

    print("Loaded file:", path)
    print("Length:", len(text))

    # Chunk the file
    chunks = chunk_text(text, chunk_size=800, overlap=150)
    print("Total chunks:", len(chunks))

    # Embedder
    embedder = HFEmbedder("intfloat/e5-large-v2")
    embeddings = embedder.encode(chunks)

    # Create store
    store = MizanMemoryStore(dim=len(embeddings[0]))

    # Add chunks to store
    for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        store.add_document(
            id=f"chunk_{idx}",
            content=chunk,
            embedding=emb,
            metadata={"chunk_id": idx}
        )

    # Query
    query = "who is ahsan shaokat?"
    q_emb = embedder.encode_one(query)

    # Retrieve using Mizan vs Cosine
    print("\nTOP RESULTS — MIZAN")
    for r in store.search(q_emb, top_k=3, metric="mizan"):
        print(f"[MIZAN] {r.score:.4f} | Chunk ID: {r.id}")

    print("\nTOP RESULTS — COSINE")
    for r in store.search(q_emb, top_k=3, metric="cosine"):
        print(f"[COS]   {r.score:.4f} | Chunk ID: {r.id}")

if __name__ == "__main__":
    main()
