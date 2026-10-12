"""
FastAPI Server for RAG Application
Provides endpoints for document ingestion, querying, and real-time streaming chat.
"""

import os
import json
import tempfile
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag_engine import RAGEngine

app = FastAPI(
    title="RAG Backend API",
    description="High-performance backend for document ingestion, semantic search, and streaming RAG.",
    version="1.0.0"
)

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG Engine instance
engine = RAGEngine()


class ChatRequest(BaseModel):
    query: str
    model: Optional[str] = None
    top_k: Optional[int] = 3


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]


@app.get("/api/status")
def get_status():
    """Return database statistics, Ollama availability, and system state."""
    stats = engine.get_stats()
    return {
        "status": "healthy",
        "total_chunks": stats["total_chunks"],
        "unique_documents": stats["unique_documents"],
        "collection_name": stats["collection_name"],
        "ollama_online": stats["ollama_online"],
        "default_model": engine.llm_model,
        "embedding_model": engine.embedding_model,
    }


@app.post("/api/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload one or more documents and index them into ChromaDB."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    total_chunks = 0
    processed_files = []

    for file in files:
        filename = file.filename or "unknown_file"
        suffix = Path(filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{filename}") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            chunks = engine.ingest_file(tmp_path)
            total_chunks += chunks
            processed_files.append({"filename": filename, "chunks": chunks})
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    return {
        "message": f"Successfully indexed {len(processed_files)} document(s)",
        "total_chunks_added": total_chunks,
        "files": processed_files
    }


@app.post("/api/ingest-sample")
def ingest_samples():
    """Ingest sample documents from ./sample_docs directory."""
    sample_dir = Path("./sample_docs")
    if not sample_dir.exists():
        raise HTTPException(status_code=404, detail="sample_docs directory not found")

    total_chunks = 0
    files_processed = []

    for file_path in sample_dir.glob("*.*"):
        if file_path.is_file():
            count = engine.ingest_file(str(file_path))
            total_chunks += count
            files_processed.append({"filename": file_path.name, "chunks": count})

    return {
        "message": f"Successfully ingested {len(files_processed)} sample document(s)",
        "total_chunks_added": total_chunks,
        "files": files_processed
    }


@app.post("/api/clear")
def clear_knowledge_base():
    """Clear all indexed documents from ChromaDB."""
    engine.clear_index()
    return {"message": "Knowledge base cleared successfully."}


@app.post("/api/query", response_model=QueryResponse)
def query_documents(req: ChatRequest):
    """Synchronous non-streaming query endpoint."""
    res = engine.query(query=req.query, top_k=req.top_k or 3, model_override=req.model)
    return QueryResponse(answer=res["answer"], sources=res["sources"])


@app.post("/api/chat")
async def chat_stream(req: ChatRequest):
    """
    Streaming Server-Sent Events (SSE) chat endpoint.
    Streams sources first, followed by incremental tokens, then completion flag.
    """
    def event_generator():
        stream = engine.generate_response_stream(
            query=req.query,
            top_k=req.top_k or 3,
            model_override=req.model
        )
        for chunk in stream:
            payload = json.dumps(chunk)
            yield f"data: {payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
