import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.ingestor import ingest_pdfs_from_folder


def main():
    print("=" * 50)
    print("Testing Document Ingestion & Chunking")
    print("=" * 50)

    chunks = ingest_pdfs_from_folder()

    print("\n--- Sample Chunk ---")
    print(f"Content: {chunks[0].page_content[:300]}")
    print(f"Metadata: {chunks[0].metadata}")
    print("\n✅ Ingestion successful!")


if __name__ == "__main__":
    main()