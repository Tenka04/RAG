"""
FastAPI Server for RAG Application
Provides endpoints for document ingestion, querying, real-time streaming chat,
and extensive Developer Mode debugging, telemetry, embedding tests, and log tracking.
"""

import os
import sys
import time
import json
import uuid
import tempfile
import traceback
from pathlib import Path
from datetime import datetime
from collections import deque
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag_engine import RAGEngine

app = FastAPI(
    title="Assist Me RAG Backend API",
    description="High-performance backend for document ingestion, semantic search, streaming RAG, and Developer Telemetry.",
    version="1.1.0"
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

# Developer Mode Passcode (Can be overridden via environment)
DEV_PASSCODE = os.environ.get("DEV_PASSCODE", "dev123")

# Ring buffer for developer mode logs and bug tracking (last 250 items)
DEV_LOGS = deque(maxlen=250)


def log_dev_event(level: str, message: str, endpoint: Optional[str] = None, duration_ms: Optional[float] = None, traceback_str: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
    """Record an event or bug into the developer diagnostics log buffer."""
    entry = {
        "id": f"log_{uuid.uuid4().hex[:8]}",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "level": level.upper(),
        "message": message,
        "endpoint": endpoint,
        "duration_ms": duration_ms,
        "traceback": traceback_str,
        "details": details or {}
    }
    DEV_LOGS.appendleft(entry)


# Request logging middleware
@app.middleware("http")
async def developer_logging_middleware(request: Request, call_next):
    start_time = time.time()
    endpoint = request.url.path
    method = request.method

    try:
        response = await call_next(request)
        duration_ms = round((time.time() - start_time) * 1000, 2)
        status_code = response.status_code

        # Log requests to dev logger
        level = "ERROR" if status_code >= 400 else "INFO"
        if endpoint.startswith("/api/"):
            log_dev_event(
                level=level,
                message=f"{method} {endpoint} -> HTTP {status_code} ({duration_ms}ms)",
                endpoint=endpoint,
                duration_ms=duration_ms,
                details={"status_code": status_code, "method": method}
            )
        return response

    except Exception as exc:
        duration_ms = round((time.time() - start_time) * 1000, 2)
        tb_str = traceback.format_exc()
        log_dev_event(
            level="ERROR",
            message=f"Unhandled exception in {method} {endpoint}: {str(exc)}",
            endpoint=endpoint,
            duration_ms=duration_ms,
            traceback_str=tb_str,
            details={"error_type": type(exc).__name__}
        )
        raise exc


# ==============================================================================
# Pydantic Schemas
# ==============================================================================

class ChatRequest(BaseModel):
    query: str
    model: Optional[str] = None
    top_k: Optional[int] = 3


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]


class DevAuthRequest(BaseModel):
    passcode: str


class EmbedTestRequest(BaseModel):
    text: str


# ==============================================================================
# Standard Endpoints
# ==============================================================================

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
        "persist_dir": stats["persist_dir"]
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
            log_dev_event(
                level="INFO",
                message=f"Ingested file '{filename}' into ChromaDB with {chunks} chunks.",
                endpoint="/api/upload",
                details={"filename": filename, "chunks": chunks}
            )
        except Exception as e:
            tb_str = traceback.format_exc()
            log_dev_event(
                level="ERROR",
                message=f"Error parsing file '{filename}': {str(e)}",
                endpoint="/api/upload",
                traceback_str=tb_str,
                details={"filename": filename}
            )
            raise HTTPException(status_code=500, detail=f"Failed to process {filename}: {str(e)}")
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
        sample_dir = Path(__file__).parent / "sample_docs"
    if not sample_dir.exists():
        sample_dir = Path(__file__).parent.parent / "sample_docs"
    if not sample_dir.exists():
        log_dev_event(level="WARN", message="sample_docs directory not found on server.", endpoint="/api/ingest-sample")
        raise HTTPException(status_code=404, detail="sample_docs directory not found")

    total_chunks = 0
    files_processed = []

    for file_path in sample_dir.glob("*.*"):
        if file_path.is_file():
            try:
                count = engine.ingest_file(str(file_path))
                total_chunks += count
                files_processed.append({"filename": file_path.name, "chunks": count})
            except Exception as e:
                log_dev_event(
                    level="ERROR",
                    message=f"Failed sample ingestion for '{file_path.name}': {str(e)}",
                    endpoint="/api/ingest-sample",
                    traceback_str=traceback.format_exc()
                )

    log_dev_event(
        level="INFO",
        message=f"Sample documents ingested: {len(files_processed)} files, {total_chunks} chunks.",
        endpoint="/api/ingest-sample"
    )

    return {
        "message": f"Successfully ingested {len(files_processed)} sample document(s)",
        "total_chunks_added": total_chunks,
        "files": files_processed
    }


@app.post("/api/clear")
def clear_knowledge_base():
    """Clear all indexed documents from ChromaDB."""
    engine.clear_index()
    log_dev_event(level="WARN", message="ChromaDB knowledge base was cleared by user.", endpoint="/api/clear")
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
    Streams sources first, followed by incremental tokens, developer telemetry, then completion flag.
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


# ==============================================================================
# Developer Mode Dedicated Endpoints
# ==============================================================================

@app.post("/api/dev/auth")
def dev_authenticate(req: DevAuthRequest):
    """Verify developer passcode to unlock Developer Mode."""
    if req.passcode.strip() == DEV_PASSCODE:
        token = f"dev_auth_{int(time.time())}"
        log_dev_event(level="INFO", message="Developer Mode access unlocked successfully.", endpoint="/api/dev/auth")
        return {
            "authenticated": True,
            "token": token,
            "message": "Developer Mode authorized."
        }
    else:
        log_dev_event(level="WARN", message=f"Failed developer access attempt with passcode '{req.passcode}'.", endpoint="/api/dev/auth")
        raise HTTPException(status_code=401, detail="Invalid developer passcode. Default is 'dev123'.")


@app.get("/api/dev/logs")
def get_dev_logs(level: Optional[str] = None):
    """Return recent logs and bugs with full stack traces and timings."""
    all_logs = list(DEV_LOGS)
    if level and level.upper() != "ALL":
        filtered = [l for l in all_logs if l["level"] == level.upper()]
        return {"logs": filtered, "total": len(filtered)}
    return {"logs": all_logs, "total": len(all_logs)}


@app.post("/api/dev/logs/clear")
def clear_dev_logs():
    """Clear all recorded developer logs."""
    DEV_LOGS.clear()
    log_dev_event(level="INFO", message="Developer logs buffer cleared.", endpoint="/api/dev/logs/clear")
    return {"message": "Logs cleared successfully."}


@app.post("/api/dev/logs/simulate-error")
def simulate_error():
    """Simulate an error to test the bug diagnostics pipeline."""
    try:
        # Intentionally produce a KeyError / ZeroDivisionError for trace demonstration
        x = 100 / 0
    except Exception as e:
        tb_str = traceback.format_exc()
        log_dev_event(
            level="ERROR",
            message="[Simulated Bug] ZeroDivisionError: float division by zero in RAG vector pipeline calculation.",
            endpoint="/api/dev/logs/simulate-error",
            traceback_str=tb_str,
            details={"debug_instruction": "Test bug generated to verify stack trace inspection in Developer Console."}
        )
        return {
            "simulated": True,
            "message": "Error simulated and logged to Developer Console.",
            "error_type": "ZeroDivisionError",
            "traceback": tb_str
        }


@app.post("/api/dev/embed-test")
def test_text_embedding(req: EmbedTestRequest):
    """Compute and inspect real-time vector embedding for any input string."""
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    details = engine.test_embedding(text)
    log_dev_event(
        level="DEBUG",
        message=f"Embedding computed for test text ('{text[:30]}...'): {details.get('dimensions')} dimensions in {details.get('latency_ms')}ms",
        endpoint="/api/dev/embed-test",
        duration_ms=details.get("latency_ms"),
        details={"dimensions": details.get("dimensions"), "norm": details.get("norm")}
    )
    return details


@app.get("/api/dev/collection-chunks")
def get_collection_chunks(limit: int = 50):
    """Inspect raw chunks, IDs, and metadata from ChromaDB."""
    chunks = engine.get_all_chunks(limit=limit)
    return {
        "total_returned": len(chunks),
        "total_in_db": engine.collection.count(),
        "chunks": chunks
    }


@app.get("/api/dev/diagnostics")
def get_system_diagnostics():
    """Return complete environment, ChromaDB, and Ollama diagnostics."""
    # Check Ollama available models
    ollama_models = []
    try:
        import ollama
        models_data = ollama.list()
        if isinstance(models_data, dict) and "models" in models_data:
            ollama_models = [m.get("model") or m.get("name", "") for m in models_data["models"]]
        elif hasattr(models_data, "models"):
            ollama_models = [m.model for m in models_data.models]
    except Exception:
        pass

    import chromadb
    return {
        "status": "healthy",
        "python_version": sys.version.split(" ")[0],
        "chromadb_version": chromadb.__version__,
        "persist_dir": str(Path(engine.persist_dir).resolve()),
        "total_chunks": engine.collection.count(),
        "collection_name": engine.collection_name,
        "ollama_online": engine._check_ollama_available(),
        "available_ollama_models": ollama_models,
        "default_llm_model": engine.llm_model,
        "embedding_model": engine.embedding_model,
        "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "error_count": sum(1 for l in DEV_LOGS if l["level"] == "ERROR")
    }


# Initial server startup log
log_dev_event(
    level="INFO",
    message="Assist Me RAG backend server initialized. Developer Mode ready.",
    endpoint="startup",
    details={"default_model": engine.llm_model, "embedding_model": engine.embedding_model}
)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
