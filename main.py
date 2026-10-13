
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
import os

from app.core.config import settings
from app.utils.qa_chain import ask_question
from app.utils.ingestor import ingest_pdfs_from_folder
from app.utils.embedder import store_embeddings, load_vectorstore


app = FastAPI(
    title="RAG Document Q&A",
    description="Natural language Q&A over PDF documents using RAG",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory="static"), name="static")


class QuestionRequest(BaseModel):
    question: str


@app.on_event("startup")
def restore_knowledge_base():
    """Rebuild the sample knowledge base if ChromaDB is empty."""
    try:
        vectorstore = load_vectorstore()
        document_count = vectorstore._collection.count()

        if document_count > 0:
            print(
                f"Knowledge base already contains "
                f"{document_count} chunks."
            )
            return

        sample_dir = Path(__file__).resolve().parent / "sample_docs"

        if not sample_dir.exists() or not list(sample_dir.glob("*.pdf")):
            print("No sample PDF found. Upload a PDF using /ingest.")
            return

        print("Empty knowledge base detected. Rebuilding from sample PDF...")

        chunks = ingest_pdfs_from_folder(str(sample_dir))
        store_embeddings(chunks)

        print(f"Knowledge base restored with {len(chunks)} chunks.")

    except Exception as e:
        print(f"Knowledge-base recovery failed: {e}")


@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "RAG Q&A API is running"}


@app.post("/ask")
def ask(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    try:
        return ask_question(request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest")
def ingest_document(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )

    try:
        os.makedirs(settings.upload_dir, exist_ok=True)

        upload_path = os.path.join(settings.upload_dir, file.filename)

        with open(upload_path, "wb") as buffer:
            buffer.write(file.file.read())

        chunks = ingest_pdfs_from_folder()
        store_embeddings(chunks)

        return {
            "message": f"Successfully ingested {file.filename}",
            "chunks": len(chunks),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

