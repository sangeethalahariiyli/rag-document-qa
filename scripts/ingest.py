import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.ingestor import ingest_pdfs_from_folder
from app.utils.embedder import store_embeddings


def main():
    print("=" * 50)
    print("RAG Ingestion Pipeline")
    print("=" * 50)

    # Step 1 — Load and chunk PDFs
    print("\n[Step 1] Loading and chunking PDFs...")
    chunks = ingest_pdfs_from_folder()

    # Step 2 — Generate embeddings and store in ChromaDB
    print("\n[Step 2] Generating embeddings and storing in ChromaDB...")
    vectorstore = store_embeddings(chunks)

    # Step 3 — Test a quick search
    print("\n[Step 3] Testing semantic search...")
    results = vectorstore.similarity_search("What is machine learning?", k=2)
    print(f"Found {len(results)} relevant chunks for test query")
    print(f"\nSample result:\n{results[0].page_content[:300]}")

    print("\n✅ Pipeline complete! Ready for Q&A.")


if __name__ == "__main__":
    main()