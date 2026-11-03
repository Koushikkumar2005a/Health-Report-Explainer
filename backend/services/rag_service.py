"""Compatibility wrapper for callers of the previous lightweight RAG service."""

from backend.rag.rag_pipeline import format_retrieved_context, retrieve_report_context


def retrieve_relevant_context(report_text: str) -> str:
    """Return retrieved corpus context, or an explicit unavailable status."""
    result = retrieve_report_context(report_text)
    test_contexts = format_retrieved_context(result)
    if not test_contexts:
        return f"No supporting medical context was retrieved ({result['retrieval_status']})."

    formatted = []
    for test in test_contexts:
        for evidence in test["evidence"]:
            formatted.append(
                f"Test: {test['test_name']}\n"
                f"Source ID: {evidence['source_id']}\n"
                f"Source: {evidence['source_filename']} "
                f"(category: {evidence['category']}, page: {evidence['page_number']})\n"
                f"{evidence['text']}"
            )
    return "\n\n".join(formatted)
