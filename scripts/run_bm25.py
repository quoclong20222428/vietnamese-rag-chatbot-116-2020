"""Standalone CLI for BM25 retrieval."""

import argparse
import sys
import logging
from pathlib import Path

# Configure stdout for Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add scripts directory to path to allow importing from scripts.
SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT = SCRIPTS_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.retrievers.bm25 import BM25Retriever
from scripts.evaluation.metrics import _get_chunk_id
from scripts.evaluation.matching import _is_legal_source

def main():
    parser = argparse.ArgumentParser(description="Standalone CLI for BM25 retrieval.")
    parser.add_argument("--query", type=str, required=True, help="The query text.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to retrieve (default: 5).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    print("Initializing BM25 Retriever...")
    retriever = BM25Retriever()
    
    print()
    print("=" * 60)
    print("AD-HOC QUERY (BM25)")
    print("=" * 60)
    print(f"Query:  {args.query}")
    print(f"Method: BM25")
    print(f"Top-K:  {args.top_k}")
    print()

    results = retriever.retrieve(args.query, top_k=args.top_k)

    if not results:
        print("(no results returned)")
        return 0

    for rank, result in enumerate(results, start=1):
        chunk_id = _get_chunk_id(result) or "N/A"
        score = getattr(result, "score", None)
        score_str = f"{score:.6f}" if isinstance(score, (int, float)) else "N/A"
        meta = getattr(result, "metadata", None) or {}
        text = getattr(result, "text", "") or ""

        print("-" * 60)
        print(f"Rank {rank}")
        print(f"Score:    {score_str}")
        print(f"Chunk ID: {chunk_id}")

        if _is_legal_source(meta):
            if meta.get("document_title"):
                print(f"Document: {meta['document_title']}")
            if meta.get("article"):
                print(f"Article:  {meta['article']}")
            if meta.get("clause"):
                print(f"Clause:   {meta['clause']}")
            if meta.get("point"):
                print(f"Point:    {meta['point']}")
        else:
            print("Source:   QA")

        print()
        print("Text:")
        print(text[:300] + "..." if len(text) > 300 else text)
        print()

    print("=" * 60)
    print(f"Done.  {len(results)} result(s) returned.")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
