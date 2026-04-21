import time
import uuid
from typing import List

import chromadb
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.core.config import settings


def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model=settings.gemini_embedding_model,
        google_api_key=settings.gemini_api_key,
    )


def store_embeddings(chunks: List[Document]) -> Chroma:
    """Embed and store one by one to avoid rate limit and length issues."""

    # Filter out empty chunks
    chunks = [c for c in chunks if len(c.page_content.strip()) > 20]
    print(f"Generating embeddings for {len(chunks)} chunks...")

    embeddings_model = get_embeddings()

    # Create ChromaDB client directly
    client = chromadb.PersistentClient(path=settings.vectorstore_path)
    collection = client.get_or_create_collection(name="documents")

    for i, chunk in enumerate(chunks):
        print(f"Embedding chunk {i+1}/{len(chunks)}")

        # Embed one chunk at a time
        vector = embeddings_model.embed_query(chunk.page_content)

        collection.add(
            embeddings=[vector],
            documents=[chunk.page_content],
            metadatas=[chunk.metadata],
            ids=[str(uuid.uuid4())],
        )

        time.sleep(1)  # 1 second delay between each

    print(f"✅ Embeddings stored in ChromaDB at: {settings.vectorstore_path}")

    vectorstore = Chroma(
        client=client,
        collection_name="documents",
        embedding_function=embeddings_model,
    )
    return vectorstore


def load_vectorstore() -> Chroma:
    embeddings = get_embeddings()
    client = chromadb.PersistentClient(path=settings.vectorstore_path)
    vectorstore = Chroma(
        client=client,
        collection_name="documents",
        embedding_function=embeddings,
    )
    print(f"✅ Loaded vectorstore from: {settings.vectorstore_path}")
    return vectorstore


def search_similar(query: str, k: int = None) -> List[Document]:
    if k is None:
        k = settings.retriever_k
    vectorstore = load_vectorstore()
    results = vectorstore.similarity_search(query, k=k)
    return results