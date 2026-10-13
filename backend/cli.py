"""
Interactive CLI for the RAG Application
Usage:
    python cli.py ingest <file_or_dir>
    python cli.py query "<question>"
    python cli.py chat
    python cli.py stats
"""

import sys
import argparse
from pathlib import Path

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from rag_engine import RAGEngine


def main():
    parser = argparse.ArgumentParser(description="Terminal CLI for RAG Knowledge Base")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a document or directory")
    ingest_parser.add_argument("path", type=str, help="Path to file or folder containing documents")

    # Query command
    query_parser = subparsers.add_parser("query", help="Ask a one-off question")
    query_parser.add_argument("question", type=str, help="Question to ask")
    query_parser.add_argument("--model", type=str, default=None, help="Model to use (default from env)")

    # Interactive chat command
    chat_parser = subparsers.add_parser("chat", help="Start an interactive chat session")
    chat_parser.add_argument("--model", type=str, default=None, help="Model to use (default from env)")

    # Stats command
    subparsers.add_parser("stats", help="Display knowledge base statistics")

    args = parser.parse_args()

    engine = RAGEngine()

    if args.command == "ingest":
        target = Path(args.path)
        if not target.exists():
            print(f"Error: Path '{args.path}' does not exist.")
            sys.exit(1)

        if target.is_file():
            print(f"Indexing '{target.name}'...")
            chunks = engine.ingest_file(str(target))
            print(f"Done! Ingested {chunks} chunks.")
        elif target.is_dir():
            total = 0
            for file in target.rglob("*.*"):
                if file.is_file() and not file.name.startswith("."):
                    print(f"Indexing '{file.name}'...")
                    try:
                        c = engine.ingest_file(str(file))
                        total += c
                    except Exception as e:
                        print(f"Skipping {file.name}: {e}")
            print(f"\nDone! Ingested {total} total chunks from directory.")

    elif args.command == "query":
        print(f"\nQuerying: {args.question}")
        print("=" * 60)
        res = engine.query(args.question, model_override=args.model)
        print("\nAnswer:\n", res["answer"])
        print("\n" + "=" * 60)
        print("Sources:")
        for i, s in enumerate(res["sources"]):
            print(f"[{i+1}] {s['metadata'].get('source')} (Score: {s.get('score')})")

    elif args.command == "chat":
        print("\n" + "=" * 60)
        print(f"Starting Interactive RAG Chat using [{args.model}]")
        print("Type 'exit' or 'quit' to end the session.")
        print("=" * 60 + "\n")

        while True:
            try:
                user_input = input("\nYou > ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["exit", "quit", "q"]:
                    print("Exiting chat. Goodbye!")
                    break

                print("\nAssistant > ", end="", flush=True)
                for event in engine.generate_response_stream(user_input, model_override=args.model):
                    if event["type"] == "token":
                        print(event["token"], end="", flush=True)
                print("\n")
            except (KeyboardInterrupt, EOFError):
                print("\nExiting chat. Goodbye!")
                break

    elif args.command == "stats" or args.command is None:
        stats = engine.get_stats()
        print("\n=== RAG Knowledge Base Stats ===")
        print(f"Total Chunks       : {stats['total_chunks']}")
        print(f"Unique Documents   : {len(stats['unique_documents'])}")
        print(f"Ollama Online      : {'Yes' if stats['ollama_online'] else 'No'}")
        if stats["unique_documents"]:
            print("Indexed Documents  :")
            for d in stats["unique_documents"]:
                print(f"  - {d}")
        print("================================\n")


if __name__ == "__main__":
    main()
