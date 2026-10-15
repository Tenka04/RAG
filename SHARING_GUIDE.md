# 🌐 Local Model & RAG Sharing Guide

This guide explains how to share your running local Ollama models (`qwen3:4b` for chat, `qwen3-embedding:0.6b` for embeddings, and Document RAG) with a friend.

---

## 1. Network Connection Setup

### Option A: If you and your friend are on the **Same Wi-Fi / LAN**
Your local network IP address is:
```text
192.168.0.103
```
Your server is already running on port `8000` and bound to `0.0.0.0`, so your friend can immediately connect to:
- **Base URL:** `http://192.168.0.103:8000`
- **Interactive Swagger Documentation:** `http://192.168.0.103:8000/docs`

> **Note:** If Windows Firewall shows an alert, ensure Python / Port 8000 is allowed on Private Networks.

---

### Option B: If your friend is on a **Different Network** (Across the Internet)
You can share your local models securely using a free tunnel without configuring router port-forwarding:

#### 1. Free Cloudflare Tunnel (Active & Ready):
A portable `cloudflared.exe` is already installed in your project root! Your live public tunnel is currently running:
```text
https://dish-betting-floral-found.trycloudflare.com
```
Your friend can open this URL directly from anywhere in the world!

If you ever need to restart the tunnel in the future, run:
```powershell
.\cloudflared.exe tunnel --url http://127.0.0.1:8000
```

---

## 2. Available Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/v1/share-info` | `GET` | Returns connection details & ready-to-copy curl snippets |
| `/api/v1/generate` | `POST` | Chat / text generation with local LLM (`qwen3:4b`) |
| `/api/v1/embed` | `POST` | Vector embeddings with local embedder (`qwen3-embedding:0.6b`) |
| `/api/upload` | `POST` | Upload files (PDF, DOCX, TXT, MD) to the knowledge base |
| `/api/v1/rag` | `POST` | Ask questions answered from indexed documents |
| `/v1/chat/completions` | `POST` | OpenAI-compatible chat (compatible with OpenAI SDK, LangChain, Cursor) |
| `/v1/embeddings` | `POST` | OpenAI-compatible embeddings |
| `/v1/models` | `GET` | OpenAI-compatible list of local models |
| `/docs` | `GET` | Interactive Swagger UI in browser |

---

## 3. Ready-to-Use Code Examples for Your Friend

*(Replace `192.168.0.103` with your tunnel URL if sharing across the internet)*

### A. Chat with the Local LLM (`qwen3:4b`)

#### cURL:
```bash
curl -X POST "http://192.168.0.103:8000/api/v1/generate" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Explain quantum computing in two sentences"}'
```

#### Python (`requests`):
```python
import requests

url = "http://192.168.0.103:8000/api/v1/generate"
payload = {
    "prompt": "Hello! How can you help me today?"
}
response = requests.post(url, json=payload).json()
print("Answer:", response["response"])
```

#### Streaming Tokens:
```python
import requests

url = "http://192.168.0.103:8000/api/v1/generate"
with requests.post(url, json={"prompt": "Write a short poem", "stream": True}, stream=True) as r:
    for line in r.iter_lines():
        if line:
            print(line.decode("utf-8"))
```

---

### B. Generate Embeddings with Local Model (`qwen3-embedding:0.6b`)

#### cURL:
```bash
curl -X POST "http://192.168.0.103:8000/api/v1/embed" \
     -H "Content-Type: application/json" \
     -d '{"text": "Artificial intelligence and retrieval augmented generation"}'
```

#### Python (`requests`):
```python
import requests

url = "http://192.168.0.103:8000/api/v1/embed"
payload = {
    "texts": ["First document chunk", "Second document chunk"]
}
res = requests.post(url, json=payload).json()
print(f"Model: {res['model']}")
print(f"Dimensions: {res['dimensions']}")
print(f"Vectors returned: {len(res['embeddings'])}")
```

---

### C. Upload a Document & Query it with Local RAG

#### 1. Upload a Document:
```bash
curl -X POST "http://192.168.0.103:8000/api/upload" \
     -F "files=@my_notes.pdf"
```

#### 2. Ask a Grounded Question:
```bash
curl -X POST "http://192.168.0.103:8000/api/v1/rag" \
     -H "Content-Type: application/json" \
     -d '{"query": "What are the key takeaways from the document?", "top_k": 3}'
```

---

### D. OpenAI Python SDK (Drop-in Replacement)
Your friend can use the official `openai` Python library by simply pointing `base_url` to your server:

```python
from openai import OpenAI

# Connect to your friend's local server
client = OpenAI(
    base_url="http://192.168.0.103:8000/v1",
    api_key="local-shared-key"  # Any string works
)

# 1. Chat Completion
response = client.chat.completions.create(
    model="qwen3:4b",
    messages=[
        {"role": "user", "content": "What are 3 tips for learning Python?"}
    ]
)
print(response.choices[0].message.content)

# 2. Embeddings
embed_res = client.embeddings.create(
    model="qwen3-embedding:0.6b",
    input="Text to convert into vector representation"
)
print("Vector length:", len(embed_res.data[0].embedding))
```
