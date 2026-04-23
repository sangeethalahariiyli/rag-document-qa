from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import shutil
import os

from app.core.config import settings
from app.utils.qa_chain import ask_question
from app.utils.ingestor import ingest_pdfs_from_folder
from app.utils.embedder import store_embeddings

app = FastAPI(
    title="RAG Document Q&A",
    description="Natural language Q&A over PDF documents using RAG",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory="static"), name="static")


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "RAG Q&A API is running"}


@app.post("/ask")
def ask(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        result = ask_question(request.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest")
def ingest_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    upload_path = os.path.join(settings.upload_dir, file.filename)
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        chunks = ingest_pdfs_from_folder()
        store_embeddings(chunks)
        return {
            "message": f"Successfully ingested {file.filename}",
            "chunks": len(chunks),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))