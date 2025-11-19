"""Basic example: in-memory search with Mizan vs cosine."""

from mizanvector import MizanMemoryStore


def main() -> None:
    store = MizanMemoryStore(dim=3)

    docs = {
        "doc_a": [1, 2, 3],
        "doc_b": [2, 4, 6],   # scaled version
        "doc_c": [1, 3, 2],   # permuted
    }

    for content, emb in docs.items():
        store.add_document(content=content, embedding=emb)

    query = [1, 2, 3]

    print("Query:", query)

    print("\nCosine ranking:")
    for r in store.search(query, top_k=3, metric="cosine"):
        print(f"  {r.content}: score={r.score:.4f}")

    print("\nMizan ranking:")
    for r in store.search(query, top_k=3, metric="mizan"):
        print(f"  {r.content}: score={r.score:.4f}")


if __name__ == "__main__":
    main()
