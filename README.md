# RAG Document Q&A System

A Retrieval-Augmented Generation (RAG) pipeline for natural language Q&A over PDF documents.  
Built with **LangChain**, **ChromaDB**, **OpenAI API**, and **FastAPI**. Deployable via **Docker**.

---

## Architecture

```
PDF files  →  Ingestion & Chunking  →  Embeddings  →  ChromaDB
                                                            ↓
User query  →  Embed query  →  Semantic search  →  Retrieved chunks
                                                            ↓
                                              GPT (with context)  →  Answer
```

---

## Project structure

```
rag-document-qa/
├── app/
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py          # FastAPI endpoints
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py          # Centralized settings
│   └── utils/
│       ├── __init__.py
│       ├── ingestor.py        # PDF loading & chunking
│       ├── embedder.py        # Embedding + ChromaDB storage
│       └── qa_chain.py        # RAG chain (retriever + LLM)
├── data/
│   └── uploads/               # Place your PDFs here
├── vectorstore/               # ChromaDB persisted index
├── tests/
├── scripts/
│   └── ingest.py              # CLI script to ingest PDFs
├── main.py                    # FastAPI app entrypoint
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quickstart

### 1. Clone and set up environment

```bash
git clone <repo-url>
cd rag-document-qa

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Open .env and add your OPENAI_API_KEY
```

### 3. Add PDF documents

```bash
cp your-document.pdf data/uploads/
```

### 4. Ingest documents into ChromaDB

```bash
python scripts/ingest.py
```

### 5. Run the API server

```bash
uvicorn main:app --reload
```

API available at `http://localhost:8000`  
Interactive docs at `http://localhost:8000/docs`

---

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ask` | Ask a question over ingested documents |
| `POST` | `/ingest` | Upload and ingest a PDF via API |
| `GET`  | `/health` | Health check |

### Example request

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of the document?"}'
```

---

## Docker deployment

```bash
docker-compose up --build
```

---

## Technologies

- **Python 3.11**
- **LangChain** — RAG orchestration
- **ChromaDB** — Vector store
- **OpenAI API** — Embeddings + GPT responses
- **FastAPI** — REST API
- **Docker** — Containerization
