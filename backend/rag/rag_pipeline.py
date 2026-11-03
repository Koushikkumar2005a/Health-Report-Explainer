"""Orchestrate safe retrieval and format source-grounded context for Gemini."""

from __future__ import annotations

import logging

from backend.config import settings
from backend.rag.retriever import (
    empty_test_results,
    extract_medical_terms,
    retrieve_for_report,
)
from backend.rag.vector_store import collection_count

logger = logging.getLogger(__name__)


def retrieve_report_context(report_text: str) -> dict:
    """Retrieve test-scoped evidence and distinguish an empty KB from failed retrieval."""
    terms = []
    try:
        terms = extract_medical_terms(report_text)
        if collection_count() == 0:
            result = retrieve_for_report(
                report_text,
                top_k=max(1, settings.RAG_TOP_K),
                relevance_threshold=max(0.0, min(1.0, settings.RAG_RELEVANCE_THRESHOLD)),
                recognized_terms=terms,
            )
            result["retrieval_status"] = "empty_knowledge_base"
            result["rag_used"] = False
            return result
        return retrieve_for_report(
            report_text,
            top_k=max(1, settings.RAG_TOP_K),
            relevance_threshold=max(0.0, min(1.0, settings.RAG_RELEVANCE_THRESHOLD)),
            recognized_terms=terms,
        )
    except Exception as exc:
        logger.error(
            "Medical reference retrieval failed (%s); test-specific interpretations will be withheld.",
            type(exc).__name__,
        )
        return {
            "terms": terms,
            "tests": empty_test_results(report_text, recognized_terms=terms),
            "passages": [],
            "sources": [],
            "retrieval_status": "unavailable",
            "rag_used": False,
        }


def format_retrieved_context(result: dict) -> list[dict]:
    """Return each test's passages with stable IDs and source provenance."""
    test_contexts = []
    for test in result.get("tests") or []:
        passages = test.get("passages") or []
        if not passages:
            continue
        formatted = []
        for passage in passages:
            metadata = passage.get("metadata") or {}
            formatted.append(
                {
                    "source_id": passage["source_id"],
                    "source_filename": metadata.get("source_filename", "Unknown source"),
                    "category": metadata.get("category", "Uncategorized"),
                    "page_number": metadata.get("page_number", 1),
                    "text": passage.get("text", ""),
                }
            )
        test_contexts.append(
            {
                "test_name": test["test_name"],
                "observations": test.get("observations") or [],
                "evidence": formatted,
            }
        )
    return test_contexts
