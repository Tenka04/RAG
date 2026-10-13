"""
RAG Engine: Vector Store (ChromaDB), Embedding, Retrieval, and Generation (Ollama / Ollama Cloud)
Equipped with Developer Telemetry, Embedding Inspection, and Diagnostics.
"""

import os
import sys
import time
import uuid
import math
from pathlib import Path
from typing import List, Dict, Any, Generator, Optional
import chromadb
from chromadb.config import Settings
import ollama

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

try:
    import dotenv
    dotenv.load_dotenv(dotenv_path=backend_dir / ".env")
except ImportError:
    pass

from document_loader import load_and_chunk_file


class RAGEngine:
    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: str = "rag_knowledge_base",
        embedding_model: Optional[str] = None,
        llm_model: Optional[str] = None,
        ollama_host: Optional[str] = None,
        ollama_api_key: Optional[str] = None,
    ):
        if persist_dir is None:
            backend_chroma = Path(__file__).resolve().parent / "chroma_db"
            root_chroma = Path(__file__).resolve().parent.parent / "chroma_db"
            if backend_chroma.exists():
                persist_dir = str(backend_chroma)
            elif root_chroma.exists():
                persist_dir = str(root_chroma)
            else:
                persist_dir = str(backend_chroma)

        self.persist_dir = persist_dir
        self.collection_name = collection_name

        # Host & API Key configuration (Ollama Cloud or Local)
        self.ollama_host = ollama_host or os.environ.get("OLLAMA_HOST", "https://ollama.com")
        self.ollama_api_key = ollama_api_key or os.environ.get("OLLAMA_API_KEY", "0f64427080054799997426594bb41a76.Ga6nHBfHUGxPBYFMGcyygSEO")

        # Models configuration
        self.llm_model = llm_model or os.environ.get("OLLAMA_DEFAULT_MODEL", "gpt-oss:20b")
        self.embedding_model = embedding_model or os.environ.get("OLLAMA_EMBED_MODEL", "all-minilm")
        self.init_time = time.time()

        # Initialize Ollama Client with Auth Bearer token if provided
        headers = {}
        if self.ollama_api_key:
            headers["Authorization"] = f"Bearer {self.ollama_api_key}"
        self.client = ollama.Client(host=self.ollama_host, headers=headers if headers else None)

        # Initialize persistent ChromaDB client
        os.makedirs(self.persist_dir, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=self.persist_dir)

        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG document embeddings collection"}
        )

        # Default embedding fallback helper
        self._fallback_embedder = None

    def _check_ollama_available(self) -> bool:
        """Check if Ollama service (cloud or local) is accessible."""
        try:
            self.client.list()
            return True
        except Exception:
            return False

    def _get_fallback_embedder(self):
        """Lazy load chromadb default embedding function (ONNX all-MiniLM-L6-v2) when needed."""
        if self._fallback_embedder is None:
            try:
                from chromadb.utils import embedding_functions
                self._fallback_embedder = embedding_functions.DefaultEmbeddingFunction()
            except Exception:
                pass
        return self._fallback_embedder

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding vector for a given text.
        Attempts Ollama embed first; if unauthorized or not supported by endpoint,
        seamlessly uses local Chroma ONNX DefaultEmbeddingFunction (all-MiniLM-L6-v2).
        """
        if self._check_ollama_available():
            try:
                # Try newer embed API
                res = self.client.embed(model=self.embedding_model, input=text)
                if isinstance(res, dict) and "embeddings" in res and res["embeddings"]:
                    return [float(x) for x in res["embeddings"][0]]
                elif hasattr(res, "embeddings") and res.embeddings:
                    return [float(x) for x in res.embeddings[0]]
            except Exception:
                pass

            try:
                # Try legacy embeddings API
                response = self.client.embeddings(
                    model=self.embedding_model,
                    prompt=text
                )
                emb = response.get("embedding")
                if emb:
                    return [float(x) for x in emb]
            except Exception:
                pass

        # Robust Fallback to Chroma's built-in ONNX embedding function (local all-MiniLM-L6-v2)
        try:
            embedder = self._get_fallback_embedder()
            if embedder:
                results = embedder([text])
                if results and len(results) > 0:
                    return [float(x) for x in results[0]]
        except Exception:
            pass

        return None

    def test_embedding(self, text: str) -> Dict[str, Any]:
        """
        Developer Mode inspection: computes embedding for arbitrary text,
        measuring latency, dimensions, norm, and sample float values.
        """
        t0 = time.time()
        embedding = self.get_embedding(text)
        latency_ms = round((time.time() - t0) * 1000, 2)

        if embedding:
            norm = math.sqrt(sum(x * x for x in embedding))
            # Determine provider tag
            provider = "chroma-onnx-all-MiniLM-L6-v2 (offline local)"
            if len(embedding) != 384:
                provider = f"ollama/{self.embedding_model}"
            return {
                "success": True,
                "model": provider,
                "dimensions": len(embedding),
                "norm": round(norm, 4),
                "latency_ms": latency_ms,
                "sample_vector": [round(x, 5) for x in embedding[:24]],
                "text_length": len(text)
            }
        else:
            return {
                "success": False,
                "error": "Could not generate vector embedding. Both Ollama and local ONNX failed.",
                "model": self.embedding_model,
                "dimensions": 0,
                "norm": 0.0,
                "latency_ms": latency_ms,
                "sample_vector": [],
                "text_length": len(text)
            }

    def ingest_file(self, file_path: str, chunk_size: int = 600, chunk_overlap: int = 100) -> int:
        """
        Load, chunk, and index a file into ChromaDB.
        Returns the number of chunks indexed.
        """
        chunks = load_and_chunk_file(file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunks:
            return 0

        documents = []
        metadatas = []
        ids = []
        embeddings = []

        filename = Path(file_path).name
        can_use_ollama_embed = self._check_ollama_available()

        for chunk in chunks:
            chunk_content = chunk["content"]
            chunk_meta = chunk["metadata"]
            
            clean_meta = {k: str(v) if not isinstance(v, (int, float, bool, str)) else v for k, v in chunk_meta.items()}
            doc_id = f"{filename}#chunk_{clean_meta.get('chunk_id', uuid.uuid4().hex[:8])}"
            
            documents.append(chunk_content)
            metadatas.append(clean_meta)
            ids.append(doc_id)

            if can_use_ollama_embed:
                emb = self.get_embedding(chunk_content)
                if emb:
                    embeddings.append(emb)

        # Upsert into ChromaDB
        if embeddings and len(embeddings) == len(documents):
            self.collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )
        else:
            self.collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

        return len(documents)

    def retrieve(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Retrieve relevant context chunks with performance timing and raw distances.
        Returns dictionary containing:
          - chunks: list of retrieved documents
          - retrieval_ms: latency of vector query
          - embedding_ms: latency of query embedding
          - query_embedding: raw query vector preview
        """
        t_start = time.time()
        retrieval_res = {
            "chunks": [],
            "retrieval_ms": 0.0,
            "embedding_ms": 0.0,
            "query_embedding": None
        }

        if self.collection.count() == 0:
            retrieval_res["retrieval_ms"] = round((time.time() - t_start) * 1000, 2)
            return retrieval_res

        t_embed = time.time()
        query_embedding = self.get_embedding(query)
        retrieval_res["embedding_ms"] = round((time.time() - t_embed) * 1000, 2)
        retrieval_res["query_embedding"] = query_embedding

        t_search = time.time()
        if query_embedding:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self.collection.count())
            )
        else:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(top_k, self.collection.count())
            )
        retrieval_res["retrieval_ms"] = round((time.time() - t_search) * 1000, 2)

        retrieved_docs = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0] if "distances" in results else [0.0] * len(docs)
            ids = results.get("ids", [[]])[0] if "ids" in results else []

            for i in range(len(docs)):
                raw_dist = distances[i] if i < len(distances) else 0.0
                score = round(1.0 - raw_dist, 3) if raw_dist <= 1.0 else round(1.0 / (1.0 + raw_dist), 3)
                retrieved_docs.append({
                    "content": docs[i],
                    "metadata": metas[i] if i < len(metas) else {},
                    "score": score,
                    "raw_distance": round(raw_dist, 4),
                    "chunk_id": ids[i] if i < len(ids) else f"chunk_{i}"
                })

        retrieval_res["chunks"] = retrieved_docs
        return retrieval_res

    def build_prompt(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Construct a structured grounded prompt with source context."""
        context_blocks = []
        for i, chunk in enumerate(context_chunks):
            source = chunk.get("metadata", {}).get("source", f"Document {i+1}")
            page_info = chunk.get("metadata", {}).get("chunk_id", i)
            context_blocks.append(f"--- SOURCE: {source} (Chunk {page_info}) ---\n{chunk['content']}")

        context_text = "\n\n".join(context_blocks)

        system_instruction = (
            "You are an expert AI assistant specializing in Retrieval-Augmented Generation (RAG).\n"
            "Answer the user's question using ONLY the provided context sources below.\n"
            "Guidelines:\n"
            "1. Be concise, precise, and directly answer what was asked.\n"
            "2. Cite the source document name(s) when referencing information.\n"
            "3. If the provided context does not contain enough information to answer the question, state:\n"
            "   'I cannot answer this question based on the provided documents.' Do NOT invent facts.\n"
        )

        full_prompt = (
            f"{system_instruction}\n\n"
            f"=== CONTEXT SOURCES ===\n{context_text}\n========================\n\n"
            f"User Question: {query}\n\n"
            f"Answer:"
        )
        return full_prompt

    def _generate_fallback_synthesis(self, query: str, sources: List[Dict[str, Any]]) -> str:
        """Fallback synthesis when Ollama is offline or not installed."""
        lines = [
            "### 📄 Grounded Document Findings\n",
            f"Here are the most relevant excerpts retrieved for: **\"{query}\"**\n\n"
        ]
        for i, src in enumerate(sources):
            src_name = src.get("metadata", {}).get("source", f"Source {i+1}")
            score = src.get("score", "N/A")
            lines.append(f"**From `{src_name}` (Relevance: {score}):**\n")
            lines.append(f"> {src['content'].strip()}\n\n")
        
        lines.append("\n---\n*💡 **Tip**: Ollama is offline. Run `ollama serve` and `ollama pull llama3.2:3b` to enable full conversational LLM generation.*")
        return "".join(lines)

    def generate_response_stream(
        self,
        query: str,
        top_k: int = 3,
        model_override: Optional[str] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream the generated answer while yielding retrieved sources and developer telemetry.
        Yields:
          - {"type": "sources", "sources": [...]}
          - {"type": "token", "token": "..."}
          - {"type": "dev_telemetry", "telemetry": {...}}
          - {"type": "done"}
        """
        t_overall_start = time.time()
        retrieval_data = self.retrieve(query, top_k=top_k)
        sources = retrieval_data["chunks"]
        retrieval_ms = retrieval_data["retrieval_ms"]
        embedding_ms = retrieval_data["embedding_ms"]
        query_embedding = retrieval_data["query_embedding"]

        # Yield sources right away for UI display
        yield {"type": "sources", "sources": sources}

        if not sources:
            yield {
                "type": "token",
                "token": "No relevant documents found in the database. Please upload or ingest some documents first."
            }
            total_time_ms = round((time.time() - t_overall_start) * 1000, 2)
            yield {
                "type": "dev_telemetry",
                "telemetry": {
                    "query": query,
                    "total_time_ms": total_time_ms,
                    "retrieval_time_ms": retrieval_ms,
                    "embedding_time_ms": embedding_ms,
                    "llm_time_ms": 0.0,
                    "model_used": model_override or self.llm_model,
                    "used_fallback": False,
                    "fallback_reason": None,
                    "prompt_preview": "No documents available for prompt context.",
                    "query_embedding": {
                        "model": self.embedding_model if self._check_ollama_available() else "chroma-default",
                        "dimensions": len(query_embedding) if query_embedding else 0,
                        "norm": round(math.sqrt(sum(x * x for x in query_embedding)), 4) if query_embedding else 0.0,
                        "sample_values": [round(x, 4) for x in query_embedding[:16]] if query_embedding else [],
                    },
                    "chunks_retrieved_count": 0,
                    "raw_chunks": []
                }
            }
            yield {"type": "done"}
            return

        active_model = model_override or self.llm_model
        prompt = self.build_prompt(query, sources)
        t_llm_start = time.time()
        used_fallback = False
        fallback_reason = None

        if active_model == "direct_synthesis":
            used_fallback = True
            fallback_reason = "User requested Direct Grounded Mode."
            fallback_text = self._generate_fallback_synthesis(query, sources)
            for w in fallback_text.split(" "):
                yield {"type": "token", "token": w + " "}
        else:
            try:
                stream = self.client.generate(
                    model=active_model,
                    prompt=prompt,
                    stream=True
                )
                for chunk in stream:
                    token = chunk.get("response", "")
                    if token:
                        yield {"type": "token", "token": token}

            except Exception as e:
                used_fallback = True
                fallback_reason = str(e)
                fallback_text = self._generate_fallback_synthesis(query, sources)
                words = fallback_text.split(" ")
                for w in words:
                    yield {"type": "token", "token": w + " "}

        llm_time_ms = round((time.time() - t_llm_start) * 1000, 2)
        total_time_ms = round((time.time() - t_overall_start) * 1000, 2)

        # Build comprehensive developer telemetry
        dev_telemetry = {
            "query": query,
            "total_time_ms": total_time_ms,
            "retrieval_time_ms": retrieval_ms,
            "embedding_time_ms": embedding_ms,
            "llm_time_ms": llm_time_ms,
            "model_used": active_model,
            "used_fallback": used_fallback,
            "fallback_reason": fallback_reason,
            "prompt_preview": prompt,
            "query_embedding": {
                "model": self.embedding_model if self._check_ollama_available() else "chroma-default",
                "dimensions": len(query_embedding) if query_embedding else 0,
                "norm": round(math.sqrt(sum(x * x for x in query_embedding)), 4) if query_embedding else 0.0,
                "sample_values": [round(x, 4) for x in query_embedding[:16]] if query_embedding else [],
            },
            "chunks_retrieved_count": len(sources),
            "raw_chunks": [
                {
                    "id": s.get("chunk_id", i),
                    "source": s.get("metadata", {}).get("source", "unknown"),
                    "score": s.get("score"),
                    "raw_distance": s.get("raw_distance"),
                    "content_preview": s["content"][:240] + ("..." if len(s["content"]) > 240 else ""),
                    "char_count": len(s["content"]),
                    "metadata": s.get("metadata", {})
                }
                for i, s in enumerate(sources)
            ]
        }

        yield {"type": "dev_telemetry", "telemetry": dev_telemetry}
        yield {"type": "done"}

    def query(self, query: str, top_k: int = 3, model_override: Optional[str] = None) -> Dict[str, Any]:
        """Synchronous query execution."""
        retrieval_data = self.retrieve(query, top_k=top_k)
        sources = retrieval_data["chunks"]
        if not sources:
            return {
                "answer": "No relevant documents found. Please ingest documents first.",
                "sources": [],
                "telemetry": None
            }

        active_model = model_override or self.llm_model
        prompt = self.build_prompt(query, sources)

        try:
            res = self.client.generate(model=active_model, prompt=prompt, stream=False)
            return {
                "answer": res.get("response", ""),
                "sources": sources
            }
        except Exception:
            return {
                "answer": self._generate_fallback_synthesis(query, sources),
                "sources": sources
            }

    def get_all_chunks(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch indexed chunks from ChromaDB for developer inspection."""
        count = self.collection.count()
        if count == 0:
            return []

        data = self.collection.get(
            limit=limit,
            include=["documents", "metadatas"]
        )

        chunks_list = []
        docs = data.get("documents", [])
        metas = data.get("metadatas", [])
        ids = data.get("ids", [])

        for i in range(len(docs)):
            chunks_list.append({
                "id": ids[i] if i < len(ids) else f"chunk_{i}",
                "content": docs[i],
                "char_count": len(docs[i]),
                "metadata": metas[i] if i < len(metas) else {}
            })

        return chunks_list

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics on currently indexed documents."""
        count = self.collection.count()
        sources = set()
        if count > 0:
            metas = self.collection.get(include=["metadatas"])["metadatas"]
            for m in metas:
                if m and "source" in m:
                    sources.add(m["source"])

        return {
            "total_chunks": count,
            "unique_documents": list(sources),
            "collection_name": self.collection_name,
            "ollama_online": self._check_ollama_available(),
            "ollama_host": self.ollama_host,
            "is_cloud": "ollama.com" in self.ollama_host,
            "default_model": self.llm_model,
            "embedding_model": self.embedding_model,
            "persist_dir": self.persist_dir
        }

    def clear_index(self):
        """Wipe the current collection and reset."""
        self.chroma_client.delete_collection(self.collection_name)
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG document embeddings collection"}
        )
