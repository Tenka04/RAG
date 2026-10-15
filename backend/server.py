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
import socket
from pathlib import Path
from datetime import datetime
from collections import deque
from typing import Optional, List, Dict, Any, Union

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from rag_engine import RAGEngine

try:
    from agent_graph import stream_agent_execution
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

app = FastAPI(
    title="Assist Me RAG Backend API",
    description="High-performance backend for document ingestion, semantic search, streaming RAG, and Developer Telemetry.",
    version="1.1.0"
)

# Enable CORS for Next.js frontend and external clients (LAN / friend access)
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


def get_local_ip() -> str:
    """Detect LAN IP address so friends on the same Wi-Fi can connect."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# ==============================================================================
# Pydantic Schemas
# ==============================================================================

class ChatRequest(BaseModel):
    query: str
    model: Optional[str] = None
    top_k: Optional[int] = 3
    use_agent: Optional[bool] = True


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]


class DevAuthRequest(BaseModel):
    passcode: str


class EmbedTestRequest(BaseModel):
    text: str


# ── Schemas for Local Model Sharing & OpenAI-Compatible Endpoints ─────────────

class LocalEmbedRequest(BaseModel):
    text: Optional[str] = None
    texts: Optional[List[str]] = None
    model: Optional[str] = None


class LocalGenerateRequest(BaseModel):
    prompt: str
    system: Optional[str] = None
    model: Optional[str] = None
    stream: Optional[bool] = False


class LocalRAGRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3


class OpenAIMessage(BaseModel):
    role: str
    content: str


class OpenAIChatRequest(BaseModel):
    model: Optional[str] = "qwen3:4b"
    messages: List[OpenAIMessage]
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.7


class OpenAIEmbedRequest(BaseModel):
    model: Optional[str] = "qwen3-embedding:0.6b"
    input: Any


# ==============================================================================
# Standard Endpoints
# ==============================================================================

@app.get("/api/status")
def get_status():
    """Return database statistics, Ollama availability, OpenRouter state, and system state."""
    stats = engine.get_stats()
    return {
        "status": "healthy",
        "total_chunks": stats["total_chunks"],
        "unique_documents": stats["unique_documents"],
        "collection_name": stats["collection_name"],
        "ollama_online": stats["ollama_online"],
        "ollama_host": stats.get("ollama_host", "https://ollama.com"),
        "is_cloud": stats.get("is_cloud", True),
        "openrouter_configured": stats.get("openrouter_configured", False),
        "default_model": engine.llm_model,
        "embedding_model": engine.embedding_model,
        "persist_dir": stats["persist_dir"],
        "langgraph_enabled": LANGGRAPH_AVAILABLE,
        "available_tools": ["chroma_rag", "web_search_ddg_wiki", "safe_python_calculator"] if LANGGRAPH_AVAILABLE else []
    }


@app.get("/api/models")
def get_models():
    """Return available models supported by the system (OpenRouter DeepSeek, Ollama Cloud, and Local)."""
    models = [
        {"id": "qwen3:4b", "name": "Qwen3 4B (Local GPU ⚡ Sub-Second)", "desc": "On-device RTX 3050 Ti • Instant line-by-line streaming", "is_cloud": False},
        {"id": "direct_synthesis", "name": "Direct Grounded Mode (⚡ Instant 10ms)", "desc": "Zero LLM wait • Instant local document synthesis", "is_cloud": False},
        {"id": "deepseek/deepseek-chat", "name": "DeepSeek V3 (deepseek-chat)", "desc": "OpenRouter • SOTA Speed & Flagship Intelligence", "is_cloud": True},
        {"id": "deepseek/deepseek-r1", "name": "DeepSeek R1 (deepseek-r1)", "desc": "OpenRouter • Deep Chain-of-Thought Reasoning", "is_cloud": True},
        {"id": "gpt-oss:20b", "name": "Assist Me Cloud 20B (gpt-oss:20b)", "desc": "Free Ollama Cloud • Blazing fast", "is_cloud": True},
        {"id": "gemma4:31b", "name": "Gemma 4 31B (gemma4:31b)", "desc": "Free Ollama Cloud • Deep reasoning", "is_cloud": True},
        {"id": "gpt-oss:120b", "name": "GPT-OSS 120B (gpt-oss:120b)", "desc": "Free Ollama Cloud • Flagship intelligence", "is_cloud": True},
        {"id": "nemotron-3-nano:30b", "name": "Nemotron 30B (nemotron-3-nano:30b)", "desc": "Free Ollama Cloud • Fast datacenter", "is_cloud": True},
    ]
    return {"models": models, "active_model": engine.local_model if engine._check_local_ollama_available() else engine.llm_model}


@app.post("/api/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload one or more documents and index them into ChromaDB."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    t_up_start = time.perf_counter()
    print(f"\n📥 [API-UPLOAD] Received {len(files)} file(s) for upload and indexing...")
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

    up_duration_ms = round((time.perf_counter() - t_up_start) * 1000, 2)
    print(f"📥 [API-UPLOAD] Indexed {total_chunks} chunks across {len(processed_files)} file(s) in {up_duration_ms:,.2f} ms\n")

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

    t_sample_start = time.perf_counter()
    print(f"\n📚 [API-SAMPLE] Ingesting sample documents from: {sample_dir}...")
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

    sample_ms = round((time.perf_counter() - t_sample_start) * 1000, 2)
    print(f"📚 [API-SAMPLE] Completed sample documents: {len(files_processed)} files, {total_chunks} chunks in {sample_ms:,.2f} ms\n")

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
    print("\n🗑️ [API-CLEAR] Clearing all indexed documents from ChromaDB...")
    t_clear = time.perf_counter()
    engine.clear_index()
    clear_ms = round((time.perf_counter() - t_clear) * 1000, 2)
    print(f"🗑️ [API-CLEAR] ChromaDB knowledge base reset in {clear_ms:,.2f} ms\n")
    log_dev_event(level="WARN", message="ChromaDB knowledge base was cleared by user.", endpoint="/api/clear")
    return {"message": "Knowledge base cleared successfully."}


@app.post("/api/query", response_model=QueryResponse)
def query_documents(req: ChatRequest):
    """Synchronous non-streaming query endpoint with console timing."""
    print(f"\n📡 [API-QUERY] Synchronous request received: \"{req.query[:60]}\" (model: {req.model or engine.llm_model}, agent: {req.use_agent})")
    if req.use_agent and LANGGRAPH_AVAILABLE:
        full_tokens = []
        sources = []
        for event in stream_agent_execution(engine=engine, query=req.query, model_override=req.model, top_k=req.top_k or 3):
            if event["type"] == "sources":
                sources = event.get("sources", [])
            elif event["type"] == "token":
                full_tokens.append(event.get("token", ""))
        return QueryResponse(answer="".join(full_tokens), sources=sources)
    res = engine.query(query=req.query, top_k=req.top_k or 3, model_override=req.model)
    return QueryResponse(answer=res["answer"], sources=res["sources"])


@app.post("/api/chat")
async def chat_stream(req: ChatRequest):
    """
    Streaming Server-Sent Events (SSE) chat endpoint.
    Streams sources first, followed by incremental tokens, developer telemetry, then completion flag.
    """
    t_chat_start = time.perf_counter()
    print(f"\n📡 [API-CHAT] Client connected for SSE stream: \"{req.query[:60]}\" (model: {req.model or engine.llm_model})")

    def event_generator():
        event_count = 0
        if req.use_agent and LANGGRAPH_AVAILABLE:
            stream = stream_agent_execution(
                engine=engine,
                query=req.query,
                model_override=req.model,
                top_k=req.top_k or 3
            )
        else:
            stream = engine.generate_response_stream(
                query=req.query,
                top_k=req.top_k or 3,
                model_override=req.model
            )
        for chunk in stream:
            event_count += 1
            payload = json.dumps(chunk)
            yield f"data: {payload}\n\n"
        stream_ms = round((time.perf_counter() - t_chat_start) * 1000, 2)
        print(f"📡 [API-CHAT] Finished SSE stream: {event_count} events dispatched to client in {stream_ms:,.2f} ms\n")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Content-Type": "text/event-stream; charset=utf-8",
            "Cache-Control": "no-cache, no-transform, must-revalidate",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


# ==============================================================================
# Dedicated Local Model Sharing & OpenAI-Compatible Endpoints
# (Share with friends over Wi-Fi, LAN, or Tunnels like ngrok/cloudflared)
# ==============================================================================

@app.get("/api/v1/share-info")
def get_share_info():
    """Information and quick-start instructions for friends connecting to this server."""
    local_ip = get_local_ip()
    port = 8000
    base_url = f"http://{local_ip}:{port}"
    return {
        "status": "online",
        "service": "Assist Me Local Model Sharing API",
        "local_ip": local_ip,
        "port": port,
        "base_url": base_url,
        "interactive_docs": f"{base_url}/docs",
        "running_models": {
            "chat_model": engine.local_model,
            "embedding_model": engine.embedding_model,
            "local_ollama_host": engine.local_ollama_host,
            "local_ollama_online": engine._check_local_ollama_available(),
        },
        "endpoints": {
            "generate_chat": f"POST {base_url}/api/v1/generate",
            "embeddings": f"POST {base_url}/api/v1/embed",
            "rag_query": f"POST {base_url}/api/v1/rag",
            "document_upload": f"POST {base_url}/api/upload",
            "openai_chat": f"POST {base_url}/v1/chat/completions",
            "openai_embeddings": f"POST {base_url}/v1/embeddings",
            "openai_models": f"GET {base_url}/v1/models",
        },
        "friend_quickstart_curl": {
            "chat": f'curl -X POST "{base_url}/api/v1/generate" -H "Content-Type: application/json" -d \'{{"prompt": "Hello from my device!"}}\'',
            "embed": f'curl -X POST "{base_url}/api/v1/embed" -H "Content-Type: application/json" -d \'{{"text": "Sample text to embed"}}\'',
            "rag": f'curl -X POST "{base_url}/api/v1/rag" -H "Content-Type: application/json" -d \'{{"query": "What is in the document?"}}\'',
            "openai_sdk": f'import openai; client = openai.OpenAI(base_url="{base_url}/v1", api_key="not-needed")'
        }
    }


@app.post("/api/v1/embed")
def local_embedding(req: LocalEmbedRequest):
    """
    Generate embeddings using the running local embedding model (e.g. qwen3-embedding:0.6b).
    Accepts a single string `text` or a list of strings `texts`.
    """
    target_text = req.texts if req.texts is not None else (req.text or "")
    if not target_text:
        raise HTTPException(status_code=400, detail="Must provide either 'text' or 'texts' to embed.")

    res = engine.embed_local(target_text, model=req.model)
    return res


@app.post("/api/v1/generate")
def local_generate(req: LocalGenerateRequest):
    """
    Generate text using the running local LLM (e.g. qwen3:4b).
    Supports both non-streaming JSON response and SSE streaming (`stream: true`).
    """
    if req.stream:
        def stream_generator():
            try:
                for token in engine.generate_local_stream(prompt=req.prompt, model=req.model, system=req.system):
                    payload = json.dumps({"token": token})
                    yield f"data: {payload}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
    else:
        try:
            answer = engine.generate_local_sync(prompt=req.prompt, model=req.model, system=req.system)
            return {
                "model": req.model or engine.local_model,
                "response": answer
            }
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Local LLM error: {str(e)}")


@app.post("/api/v1/rag")
def local_rag_query(req: LocalRAGRequest):
    """
    100% Local RAG query: searches ChromaDB with local embeddings,
    then answers the question using the local LLM.
    """
    result = engine.query_local(query=req.query, top_k=req.top_k or 3)
    return result


# ── OpenAI-Compatible Endpoints ───────────────────────────────────────────────

@app.get("/v1/models")
def openai_list_models():
    """List available local models in OpenAI-compatible format."""
    now_ts = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": engine.local_model, "object": "model", "created": now_ts, "owned_by": "local-ollama"},
            {"id": engine.embedding_model, "object": "model", "created": now_ts, "owned_by": "local-ollama"},
            {"id": "local-rag", "object": "model", "created": now_ts, "owned_by": "assist-me-rag"},
        ]
    }


@app.post("/v1/chat/completions")
def openai_chat_completions(req: OpenAIChatRequest):
    """OpenAI-compatible chat completions endpoint (works with Python openai SDK, LangChain, etc.)."""
    # Build prompt from messages
    prompt_parts = []
    system_prompt = None
    for msg in req.messages:
        if msg.role == "system":
            system_prompt = msg.content
        else:
            role_label = "User" if msg.role == "user" else "Assistant"
            prompt_parts.append(f"{role_label}: {msg.content}")
    prompt_parts.append("Assistant: ")
    full_prompt = "\n".join(prompt_parts)

    call_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_ts = int(time.time())
    model_name = req.model or engine.local_model

    if req.stream:
        def openai_stream():
            try:
                for token in engine.generate_local_stream(prompt=full_prompt, model=model_name, system=system_prompt):
                    chunk = {
                        "id": call_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": model_name,
                        "choices": [
                            {"index": 0, "delta": {"content": token}, "finish_reason": None}
                        ]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                # Final finish chunk
                final_chunk = {
                    "id": call_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                err_chunk = {"error": {"message": str(e), "type": "server_error"}}
                yield f"data: {json.dumps(err_chunk)}\n\n"

        return StreamingResponse(
            openai_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
    else:
        try:
            content = engine.generate_local_sync(prompt=full_prompt, model=model_name, system=system_prompt)
            return {
                "id": call_id,
                "object": "chat.completion",
                "created": created_ts,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": len(full_prompt) // 4,
                    "completion_tokens": len(content) // 4,
                    "total_tokens": (len(full_prompt) + len(content)) // 4
                }
            }
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Ollama generation failed: {str(e)}")


@app.post("/v1/embeddings")
def openai_embeddings(req: OpenAIEmbedRequest):
    """OpenAI-compatible embeddings endpoint."""
    inputs = [req.input] if isinstance(req.input, str) else req.input
    embed_res = engine.embed_local(inputs, model=req.model)
    data = []
    for idx, vec in enumerate(embed_res.get("embeddings", [])):
        data.append({
            "object": "embedding",
            "index": idx,
            "embedding": vec
        })
    return {
        "object": "list",
        "data": data,
        "model": req.model or engine.embedding_model,
        "usage": {
            "prompt_tokens": sum(len(x) // 4 for x in inputs if isinstance(x, str)),
            "total_tokens": sum(len(x) // 4 for x in inputs if isinstance(x, str))
        }
    }


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
    ollama_models = []
    try:
        models_data = engine.client.list()
        if isinstance(models_data, dict) and "models" in models_data:
            ollama_models = [m.get("model") or m.get("name", "") for m in models_data["models"]]
        elif hasattr(models_data, "models"):
            ollama_models = [m.model for m in models_data.models]
    except Exception:
        pass

    masked_key = f"{engine.ollama_api_key[:6]}...{engine.ollama_api_key[-6:]}" if engine.ollama_api_key else "None"
    masked_openrouter = f"{engine.openrouter_api_key[:8]}...{engine.openrouter_api_key[-4:]}" if engine.openrouter_api_key else "None"

    import chromadb
    return {
        "status": "healthy",
        "python_version": sys.version.split(" ")[0],
        "chromadb_version": chromadb.__version__,
        "persist_dir": str(Path(engine.persist_dir).resolve()),
        "total_chunks": engine.collection.count(),
        "collection_name": engine.collection_name,
        "ollama_online": engine._check_ollama_available(),
        "ollama_host": engine.ollama_host,
        "is_cloud": "ollama.com" in engine.ollama_host,
        "api_key_configured": bool(engine.ollama_api_key),
        "api_key_masked": masked_key,
        "openrouter_configured": bool(engine.openrouter_api_key),
        "openrouter_masked": masked_openrouter,
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
