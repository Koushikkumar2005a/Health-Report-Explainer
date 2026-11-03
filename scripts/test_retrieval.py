"""Query the medical knowledge base directly, without calling Gemini."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import settings
from backend.rag.retriever import retrieve_query
from backend.rag.vector_store import collection_count

EXAMPLE_QUERIES = (
    "What is haemoglobin?",
    "What does low haemoglobin indicate?",
    "What is HbA1c?",
    "What does creatinine measure?",
    "What is TSH?",
    "What is LDL cholesterol?",
)


def main() -> int:
    try:
        count = collection_count()
    except Exception as exc:
        print(f"Retrieval is unavailable: {exc}")
        return 1

    if count == 0:
        print(
            "The medical knowledge base is empty. Add PDF or TXT references under "
            "data/medical_knowledge/raw/ and run scripts/build_knowledge_base.py first."
        )
        return 0

    for query in EXAMPLE_QUERIES:
        print(f"\nQuery: {query}")
        try:
            results = retrieve_query(
                query,
                top_k=max(1, settings.RAG_TOP_K),
                relevance_threshold=max(0.0, min(1.0, settings.RAG_RELEVANCE_THRESHOLD)),
            )
        except Exception as exc:
            print(f"  Retrieval failed: {exc}")
            return 1
        if not results:
            print("  No sufficiently relevant passages were found.")
            continue
        for result in results:
            metadata = result["metadata"]
            print(
                "  Source: {source}; category: {category}; page: {page}; "
                "relevance: {score:.4f}".format(
                    source=metadata.get("source_filename", "Unknown source"),
                    category=metadata.get("category", "Uncategorized"),
                    page=metadata.get("page_number", "unknown"),
                    score=result["relevance_score"],
                )
            )
            print(f"  {result['text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
