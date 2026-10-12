# 🧠 Local & Cloud RAG (Retrieval-Augmented Generation)

A modern, production-grade Retrieval-Augmented Generation (RAG) system supporting **all document types** (`.pdf`, `.docx`, `.txt`, `.md`, `.json`, `.csv`, code files), powered by **ChromaDB** and **Ollama** (Local models & Cloud models).

---

## 🌟 Key Features

- **Multi-Format Document Ingestion**: Ingests PDFs, Word docs, Markdown, text files, and code.
- **Smart Chunking**: Recursive character splitting with configurable chunk size and overlap.
- **Persistent Vector Store**: ChromaDB local database with cosine similarity search.
- **Hybrid Embedding & Fallback**: Native Ollama embeddings (`nomic-embed-text`) with automatic built-in fallback to fast ONNX embeddings (`all-MiniLM-L6-v2`).
- **Local & Cloud Generation**:
  - **Local**: `llama3.2:3b` and `llama3.2:1b` (100% offline, fits inside 4 GB VRAM).
  - **Cloud**: `gemma4:cloud` (Ollama's free cloud tier, fast datacenter execution).
- **Dual Interfaces**:
  - **Streamlit Web UI** (`app.py`): Real-time chat, drag-and-drop document upload, source context viewer, and model switcher.
  - **Terminal CLI** (`cli.py`): Ingest, query, and chat directly from PowerShell or terminal.

---

## 🚀 Quick Start

### 1. Activate Environment
```powershell
.\venv\Scripts\Activate.ps1
```

### 2. Launch the Web UI
```powershell
.\venv\Scripts\streamlit.exe run app.py
```
Open your browser at `http://localhost:8501`.

### 3. Using the CLI
- **View Knowledge Base Statistics**:
  ```powershell
  .\venv\Scripts\python.exe cli.py stats
  ```
- **Ingest a Document or Folder**:
  ```powershell
  .\venv\Scripts\python.exe cli.py ingest sample_docs/
  ```
- **Ask a One-Off Question**:
  ```powershell
  .\venv\Scripts\python.exe cli.py query "What is the PTO policy?"
  ```
- **Interactive Chat**:
  ```powershell
  .\venv\Scripts\python.exe cli.py chat --model llama3.2:3b
  ```

---

## 🦙 Ollama Model Setup

### Recommended Local Models (for 4 GB VRAM GPUs like RTX 3050):
```powershell
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

### Using Ollama Cloud Models:
```powershell
ollama signin
ollama run gemma4:cloud
```

---

## 🏋️ Fine-Tuning on Kaggle (Dual T4 GPUs)
You can fine-tune custom models on Kaggle's free 2x T4 GPUs (32 GB total VRAM) using QLoRA:
1. Fine-tune using `unsloth` or `peft` / `trl` in a Kaggle notebook.
2. Export weights to GGUF format (`model.save_pretrained_gguf("my-model", tokenizer, "q4_k_m")`).
3. Download the `.gguf` file to your PC.
4. Import into Ollama:
   ```powershell
   ollama create my-rag-model -f Modelfile
   ```
5. Select `my-rag-model` in the Streamlit UI or CLI!
