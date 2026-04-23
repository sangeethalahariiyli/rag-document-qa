# 📚 RAG Document Q&A System

A production-ready **Retrieval-Augmented Generation (RAG)** pipeline that enables natural language Q&A over PDF documents. Built with LangChain, ChromaDB, Google Gemini API, and FastAPI.

---

## ✨ Features

- 📄 **PDF Ingestion** — Upload and process any PDF document
- 🧠 **Vector Embeddings** — Semantic search using Google Gemini embeddings
- 🔍 **Smart Retrieval** — ChromaDB vector store for fast similarity search
- 🤖 **AI Answers** — Gemini 2.5 Flash generates accurate, context-aware answers
- 🚀 **REST API** — FastAPI endpoint for easy integration
- 💬 **Chat UI** — Beautiful web interface to interact with your documents
- 🐳 **Docker Ready** — Containerized for consistent deployment

---

##---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python 3.11+ | Core language |
| LangChain | RAG orchestration |
| ChromaDB | Vector store |
| Google Gemini API | Embeddings + LLM |
| FastAPI | REST API |
| Docker | Containerization |

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/sangeethalahariiyli/rag-document-qa.git
cd rag-document-qa
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
pip install google-generativeai langchain-google-genai
```

### 4. Configure environment
```bash
cp .env.example .env
```
Edit `.env` and add your Gemini API key: 🏗️ Architecture

Get a free key at 👉 [aistudio.google.com](https://aistudio.google.com/apikey)

### 5. Add your PDF
```bash
cp your-document.pdf data/uploads/
```

### 6. Ingest documents
```bash
python scripts/ingest.py
```

### 7. Start the server
```bash
uvicorn main:app --reload
```
Open 👉 **http://127.0.0.1:8000**

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Chat UI frontend |
| `GET` | `/health` | Health check |
| `POST` | `/ask` | Ask a question |
| `POST` | `/ingest` | Upload & ingest PDF |
| `GET` | `/docs` | Interactive API docs |

### Example Request
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is machine learning?"}'
```

### Example Response
```json
{
  "question": "What is machine learning?",
  "answer": "Machine learning is a branch of artificial intelligence...",
  "sources": [{"content": "...", "page": 0, "source": "ml_test.pdf"}]
}
```

---

## 🐳 Docker Deployment

```bash
docker-compose up --build
```

---

## 🔧 Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | required | Your Gemini API key |
| `GEMINI_MODEL` | `gemini-2.5-flash` | LLM model |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-2-preview` | Embedding model |
| `CHUNK_SIZE` | `500` | Text chunk size |
| `CHUNK_OVERLAP` | `50` | Chunk overlap |
| `RETRIEVER_K` | `4` | Chunks to retrieve per query |

---

## 🙏 Acknowledgements

- [LangChain](https://langchain.com) — RAG framework
- [ChromaDB](https://chromadb.com) — Vector database
- [Google Gemini](https://ai.google.dev) — AI models
- [FastAPI](https://fastapi.tiangolo.com) — API framework
