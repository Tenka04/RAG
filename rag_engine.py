"""
RAG Engine: Vector Store (ChromaDB), Embedding, Retrieval, and Generation (Ollama)
"""

import os
import uuid
from pathlib import Path
from typing import List, Dict, Any, Generator, Optional
import chromadb
from chromadb.config import Settings
import ollama

from document_loader import load_and_chunk_file


class RAGEngine:
    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "rag_knowledge_base",
        embedding_model: str = "nomic-embed-text",
        llm_model: str = "llama3.2:3b"
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.llm_model = llm_model

        # Initialize persistent ChromaDB client
        os.makedirs(self.persist_dir, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=self.persist_dir)
        
        # Test if Ollama is accessible
        self.ollama_available = self._check_ollama_available()

        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG document embeddings collection"}
        )

    def _check_ollama_available(self) -> bool:
        """Check if Ollama service is running and accessible."""
        try:
            ollama.list()
            return True
        except Exception:
            return False

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding vector for a given text using Ollama if available."""
        if self._check_ollama_available():
            try:
                response = ollama.embeddings(
                    model=self.embedding_model,
                    prompt=text
                )
                return response.get("embedding")
            except Exception:
                pass
        return None

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
            
            # Ensure metadata values are strings, ints, floats, or bools for ChromaDB
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
            # Fallback to Chroma's built-in ONNX embedding function
            self.collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

        return len(documents)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant context chunks for a query.
        """
        if self.collection.count() == 0:
            return []

        query_embedding = None
        if self._check_ollama_available():
            query_embedding = self.get_embedding(query)

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

        retrieved_docs = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0] if "distances" in results else [0.0] * len(docs)

            for i in range(len(docs)):
                retrieved_docs.append({
                    "content": docs[i],
                    "metadata": metas[i] if i < len(metas) else {},
                    "score": round(1.0 - distances[i], 3) if i < len(distances) else 1.0
                })

        return retrieved_docs

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

    def generate_response_stream(
        self,
        query: str,
        top_k: int = 3,
        model_override: Optional[str] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream the generated answer while yielding retrieved sources.
        Yields:
          - {"type": "sources", "sources": [...]}
          - {"type": "token", "token": "..."}
          - {"type": "done"}
        """
        sources = self.retrieve(query, top_k=top_k)
        yield {"type": "sources", "sources": sources}

        if not sources:
            yield {
                "type": "token",
                "token": "No relevant documents found in the database. Please upload or ingest some documents first."
            }
            yield {"type": "done"}
            return

        active_model = model_override or self.llm_model
        prompt = self.build_prompt(query, sources)

        try:
            stream = ollama.generate(
                model=active_model,
                prompt=prompt,
                stream=True
            )
            for chunk in stream:
                token = chunk.get("response", "")
                if token:
                    yield {"type": "token", "token": token}
            yield {"type": "done"}

        except Exception as e:
            err_msg = (
                f"\n\n**Ollama Connection Notice:** Could not contact Ollama model `{active_model}`.\n"
                f"Details: {str(e)}\n\n"
                f"*Troubleshooting Tip*: Make sure the Ollama service is running (`ollama serve`) and the model is pulled (`ollama pull {active_model}`)."
            )
            yield {"type": "token", "token": err_msg}
            yield {"type": "done"}

    def query(self, query: str, top_k: int = 3, model_override: Optional[str] = None) -> Dict[str, Any]:
        """Synchronous full query generation."""
        sources = self.retrieve(query, top_k=top_k)
        if not sources:
            return {
                "answer": "No relevant documents found. Please ingest documents first.",
                "sources": []
            }

        active_model = model_override or self.llm_model
        prompt = self.build_prompt(query, sources)

        try:
            res = ollama.generate(model=active_model, prompt=prompt, stream=False)
            return {
                "answer": res.get("response", ""),
                "sources": sources
            }
        except Exception as e:
            return {
                "answer": f"Error communicating with Ollama ({active_model}): {e}",
                "sources": sources
            }

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
            "ollama_online": self._check_ollama_available()
        }

    def clear_index(self):
        """Wipe the current collection and reset."""
        self.chroma_client.delete_collection(self.collection_name)
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG document embeddings collection"}
        )
