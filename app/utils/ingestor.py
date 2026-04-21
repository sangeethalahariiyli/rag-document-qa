import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from app.core.config import settings


def load_pdf(file_path: str) -> List[Document]:
    print(f"Loading PDF: {file_path}")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")
    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def ingest_pdfs_from_folder(folder_path: str = None) -> List[Document]:
    if folder_path is None:
        folder_path = settings.upload_dir

    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Upload folder not found: {folder_path}")

    pdf_files = list(folder.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"No PDF files found in {folder_path}")

    print(f"Found {len(pdf_files)} PDF file(s): {[f.name for f in pdf_files]}")

    all_chunks = []
    for pdf_file in pdf_files:
        documents = load_pdf(str(pdf_file))
        chunks = split_documents(documents)
        all_chunks.extend(chunks)

    print(f"Total chunks ready for embedding: {len(all_chunks)}")
    return all_chunks