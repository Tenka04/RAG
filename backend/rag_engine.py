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
import json
import requests


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

        # OpenRouter Configuration (Supports OPENROUTER_API_KEY, OPEN_ROUTER_KEY, OPENROUTER_KEY)
        self.openrouter_api_key = (
            os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("OPEN_ROUTER_KEY")
            or os.environ.get("OPENROUTER_KEY")
        )

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

    def _is_openrouter_model(self, model: str) -> bool:
        """Check if model targets OpenRouter (e.g. DeepSeek models)."""
        if not model:
            return False
        m = model.strip().lower()
        if m in ("deepseek", "deepseek-chat", "deepseek-r1", "deepseek/deepseek-chat", "deepseek/deepseek-r1"):
            return True
        if m.startswith("deepseek/") or m.startswith("openrouter/"):
            return True
        if "/" in m and self.openrouter_api_key:
            return True
        return False

    def _resolve_openrouter_model(self, model: str) -> str:
        """Resolve model name alias to canonical OpenRouter model slug."""
        m = model.strip()
        if m.startswith("openrouter/"):
            m = m[len("openrouter/"):]
        if m in ("deepseek", "deepseek-chat"):
            return "deepseek/deepseek-chat"
        if m == "deepseek-r1":
            return "deepseek/deepseek-r1"
        return m

    def _generate_openrouter_stream(self, model: str, prompt: str) -> Generator[str, None, None]:
        """Stream response tokens from OpenRouter API with provider fallback and reasoning support."""
        if not self.openrouter_api_key:
            raise ValueError("OpenRouter API key is not configured. Add OPEN_ROUTER_KEY in backend/.env")

        target_model = self._resolve_openrouter_model(model)
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "Assist Me RAG"
        }
        fallback_models = [target_model]
        if target_model != "deepseek/deepseek-chat":
            fallback_models.append("deepseek/deepseek-chat")
        if target_model != "deepseek/deepseek-r1":
            fallback_models.append("deepseek/deepseek-r1")

        payload = {
            "model": target_model,
            "models": fallback_models,
            "provider": {"allow_fallbacks": True},
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2048,
            "stream": True
        }

        response = None
        for attempt in range(3):
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                stream=True,
                timeout=45
            )
            if response.status_code == 429 and attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue
            break

        if response is None or response.status_code != 200:
            err_msg = response.text if response is not None else "No response"
            raise RuntimeError(f"OpenRouter returned HTTP {response.status_code if response else 'ERR'}: {err_msg}")

        in_reasoning = False
        for line in response.iter_lines():
            if not line:
                continue
            decoded = line.decode("utf-8")
            if not decoded.startswith("data: "):
                continue
            data_str = decoded[6:].strip()
            if data_str == "[DONE]":
                break
            try:
                data = json.loads(data_str)
                choices = data.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})

                # Handle DeepSeek R1 reasoning chain-of-thought
                reasoning = delta.get("reasoning")
                if reasoning:
                    if not in_reasoning:
                        in_reasoning = True
                        yield "> **Thinking Process:**\n> "
                    yield reasoning.replace("\n", "\n> ")

                content = delta.get("content")
                if content:
                    if in_reasoning:
                        in_reasoning = False
                        yield "\n\n"
                    yield content
            except Exception:
                continue

    def _generate_openrouter_sync(self, model: str, prompt: str) -> str:
        """Synchronously generate response from OpenRouter API."""
        if not self.openrouter_api_key:
            raise ValueError("OpenRouter API key is not configured. Add OPEN_ROUTER_KEY in backend/.env")

        target_model = self._resolve_openrouter_model(model)
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "Assist Me RAG"
        }
        fallback_models = [target_model]
        if target_model != "deepseek/deepseek-chat":
            fallback_models.append("deepseek/deepseek-chat")
        if target_model != "deepseek/deepseek-r1":
            fallback_models.append("deepseek/deepseek-r1")

        payload = {
            "model": target_model,
            "models": fallback_models,
            "provider": {"allow_fallbacks": True},
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2048,
            "stream": False
        }

        response = None
        for attempt in range(3):
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=45
            )
            if response.status_code == 429 and attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue
            break

        if response is None or response.status_code != 200:
            err_msg = response.text if response is not None else "No response"
            raise RuntimeError(f"OpenRouter returned HTTP {response.status_code if response else 'ERR'}: {err_msg}")

        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            return ""

        msg = choices[0].get("message", {})
        content = msg.get("content", "")
        reasoning = msg.get("reasoning")

        if reasoning and not content:
            return f"> **Thinking Process:**\n{reasoning}"
        elif reasoning and content:
            return f"> **Thinking Process:**\n{reasoning}\n\n{content}"
        return content


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

    @staticmethod
    def _is_casual_message(query: str) -> bool:
        """Check if user query is a simple greeting or casual conversational message."""
        clean = query.strip().lower().rstrip("!.,?").strip()
        casual_phrases = [
            "hi", "hello", "hey", "hola", "greetings", "good morning",
            "good afternoon", "good evening", "howdy", "sup", "what's up",
            "yo", "how are you", "who are you", "what can you do", "thanks",
            "thank you", "bye", "goodbye"
        ]
        if clean in casual_phrases:
            return True
        for p in casual_phrases:
            if clean.startswith(f"{p} ") or clean.endswith(f" {p}"):
                return True
        return False

    def build_prompt(self, query: str, context_chunks: Optional[List[Dict[str, Any]]] = None) -> str:
        """Construct a friendly, flexible grounded prompt or general chat prompt."""
        if context_chunks and len(context_chunks) > 0 and not self._is_casual_message(query):
            context_blocks = []
            for i, chunk in enumerate(context_chunks):
                source = chunk.get("metadata", {}).get("source", f"Document {i+1}")
                page_info = chunk.get("metadata", {}).get("chunk_id", i)
                context_blocks.append(f"--- SOURCE: {source} (Chunk {page_info}) ---\n{chunk['content']}")

            context_text = "\n\n".join(context_blocks)

            system_instruction = (
                "You are Assist Me, a friendly, intelligent, and versatile AI assistant equipped with document knowledge.\n"
                "Instructions:\n"
                "1. GREETINGS & CASUAL CHAT: If the user says 'hi', 'hello', asks how you are, or engages in casual conversation, reply naturally, warmly, and politely without citing documents.\n"
                "2. DOCUMENT QUESTIONS: When the user asks about topics described in the provided CONTEXT SOURCES, provide an accurate, grounded answer and cite the source name(s) (e.g., [document.txt]).\n"
                "3. GENERAL QUESTIONS: When the user asks general questions not covered in the documents (e.g. coding, math, general knowledge, explanations, writing), answer them directly, helpfully, and clearly using your general intelligence.\n"
                "4. Tone: Helpful, clear, concise, and conversational. Do NOT say 'I cannot answer based on the provided documents' for casual conversation or general knowledge."
            )

            full_prompt = (
                f"{system_instruction}\n\n"
                f"=== CONTEXT SOURCES ===\n{context_text}\n========================\n\n"
                f"User Question: {query}\n\n"
                f"Answer:"
            )
            return full_prompt
        else:
            system_instruction = (
                "You are Assist Me, a friendly, intelligent, and highly capable AI assistant.\n"
                "Answer the user's questions clearly, accurately, and politely. Be conversational, helpful, and concise."
            )
            return f"{system_instruction}\n\nUser Question: {query}\n\nAnswer:"

    def _generate_fallback_synthesis(self, query: str, sources: List[Dict[str, Any]]) -> str:
        """Fallback synthesis when Ollama is offline or in Direct Grounded Mode."""
        if sources:
            lines = [
                "### 📄 Grounded Document Findings\n",
                f"Here are the most relevant excerpts retrieved for: **\"{query}\"**\n\n"
            ]
            for i, src in enumerate(sources):
                src_name = src.get("metadata", {}).get("source", f"Source {i+1}")
                score = src.get("score", "N/A")
                lines.append(f"**From `{src_name}` (Relevance: {score}):**\n")
                lines.append(f"> {src['content'].strip()}\n\n")
            
            lines.append("\n---\n*💡 **Tip**: Direct Grounded Mode active. Select `gpt-oss:20b` for full conversational reasoning.*")
            return "".join(lines)
        else:
            return (
                f"Hello! I am **Assist Me**.\n\n"
                f"You asked: *\"{query}\"*\n\n"
                f"No relevant documents were found in the database. In Direct Grounded Mode, responses are synthesized directly from indexed documents. "
                f"Upload documents in the sidebar or switch to **`gpt-oss:20b`** for conversational AI."
            )

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
        is_casual = self._is_casual_message(query)

        # Retrieve relevant context if not purely a casual greeting and collection has documents
        if not is_casual and self.collection.count() > 0:
            retrieval_data = self.retrieve(query, top_k=top_k)
            raw_sources = retrieval_data["chunks"]
            # Filter sources to only include chunks within meaningful semantic distance
            sources = [s for s in raw_sources if s.get("raw_distance", 2.0) <= 1.45]
            retrieval_ms = retrieval_data["retrieval_ms"]
            embedding_ms = retrieval_data["embedding_ms"]
            query_embedding = retrieval_data["query_embedding"]
        else:
            sources = []
            retrieval_ms = 0.0
            embedding_ms = 0.0
            query_embedding = None

        # Yield sources right away for UI display
        yield {"type": "sources", "sources": sources}

        active_model = model_override or self.llm_model
        prompt = self.build_prompt(query, sources if sources else None)
        t_llm_start = time.time()
        used_fallback = False
        fallback_reason = None

        if active_model == "direct_synthesis":
            used_fallback = True
            fallback_reason = "User requested Direct Grounded Mode."
            fallback_text = self._generate_fallback_synthesis(query, sources)
            for w in fallback_text.split(" "):
                yield {"type": "token", "token": w + " "}
        elif self._is_openrouter_model(active_model):
            try:
                for token in self._generate_openrouter_stream(active_model, prompt):
                    yield {"type": "token", "token": token}
            except Exception as e:
                used_fallback = True
                fallback_reason = str(e)
                fallback_text = self._generate_fallback_synthesis(query, sources)
                words = fallback_text.split(" ")
                for w in words:
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
        is_casual = self._is_casual_message(query)
        retrieval_data = self.retrieve(query, top_k=top_k) if not is_casual else {"chunks": []}
        sources = retrieval_data.get("chunks", [])

        active_model = model_override or self.llm_model
        prompt = self.build_prompt(query, sources if sources else None)

        if self._is_openrouter_model(active_model):
            try:
                ans = self._generate_openrouter_sync(active_model, prompt)
                return {
                    "answer": ans,
                    "sources": sources
                }
            except Exception:
                return {
                    "answer": self._generate_fallback_synthesis(query, sources),
                    "sources": sources
                }

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
            "openrouter_configured": bool(self.openrouter_api_key),
            "persist_dir": self.persist_dir
        }

    def clear_index(self):
        """Wipe the current collection and reset."""
        self.chroma_client.delete_collection(self.collection_name)
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG document embeddings collection"}
        )
