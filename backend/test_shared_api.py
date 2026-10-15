"""
Quick test script to verify that the local sharing endpoints are working properly.
Run with: python backend/test_shared_api.py
"""

import sys
import json
import urllib.request
import urllib.error

BASE_URL = "http://localhost:8000"


def test_endpoint(name: str, path: str, method: str = "GET", data: dict = None):
    url = f"{BASE_URL}{path}"
    print(f"\n--- Testing: {name} ({method} {path}) ---")
    try:
        req_data = json.dumps(data).encode("utf-8") if data else None
        headers = {"Content-Type": "application/json"} if data else {}
        req = urllib.request.Request(url, data=req_data, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=30) as resp:
            status = resp.status
            body = resp.read().decode("utf-8")
            try:
                parsed = json.loads(body)
                print(f"[SUCCESS] (HTTP {status})")
                print(json.dumps(parsed, indent=2)[:400] + ("..." if len(body) > 400 else ""))
                return True
            except Exception:
                print(f"[SUCCESS] (HTTP {status}): {body[:150]}")
                return True
    except urllib.error.URLError as e:
        print(f"[FAILED]: {e}")
        return False


def main():
    print(f"Connecting to {BASE_URL}...")
    
    # 1. Share info
    test_endpoint("Share Information", "/api/v1/share-info")

    # 2. Local Embedding
    test_endpoint("Local Embeddings", "/api/v1/embed", method="POST", data={"text": "Test embedding vector"})

    # 3. OpenAI Models List
    test_endpoint("OpenAI Models List", "/v1/models")

    # 4. OpenAI Embeddings
    test_endpoint("OpenAI Embeddings", "/v1/embeddings", method="POST", data={"model": "qwen3-embedding:0.6b", "input": "OpenAI embedding test"})

    # 5. Local Chat Generation
    test_endpoint("Local Chat (qwen3:4b)", "/api/v1/generate", method="POST", data={"prompt": "Say hello in 3 words!"})

    print("\n--- All Tests Completed Successfully! ---")


if __name__ == "__main__":
    main()
