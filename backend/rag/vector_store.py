"""Persistent ChromaDB storage for medical reference passages."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from backend.rag.embeddings import E5EmbeddingFunction, embed_documents
from backend.config import settings

COLLECTION_NAME = "medical_knowledge"
REQUIRED_CATEGORY_QUERIES = {
    "CBC": "haemoglobin blood test",
    "Diabetes": "blood glucose test",
    "KFT": "creatinine kidney function",
    "LFT": "ALT liver function",
    "Lipid_Profile": "cholesterol lipid profile",
    "Thyroid": "TSH thyroid hormone",
}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_configured_path = Path(settings.RAG_CHROMA_PATH)
CHROMA_PATH = (
    _configured_path if _configured_path.is_absolute() else PROJECT_ROOT / _configured_path
).resolve()

_client = None
_collection = None
_store_lock = threading.Lock()
logger = logging.getLogger(__name__)


class VectorStoreError(RuntimeError):
    """Raised when persistent vector storage cannot be accessed."""


def get_collection():
    """Return the process-wide persistent collection with the custom E5 function."""
    global _client, _collection
    if _collection is None:
        with _store_lock:
            if _collection is None:
                try:
                    import chromadb
                except ImportError as exc:
                    raise VectorStoreError(
                        "chromadb is required for medical reference retrieval."
                    ) from exc

                CHROMA_PATH.mkdir(parents=True, exist_ok=True)
                _client = chromadb.PersistentClient(path=str(CHROMA_PATH))
                _collection = _client.get_or_create_collection(
                    name=COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"},
                    embedding_function=E5EmbeddingFunction(),
                )
    return _collection


def collection_count() -> int:
    return int(get_collection().count())


def get_indexed_source_hashes() -> dict[str, str]:
    collection = get_collection()
    records = collection.get(include=["metadatas"])
    hashes: dict[str, str] = {}
    for metadata in records.get("metadatas") or []:
        if metadata:
            source_path = metadata.get("source_path")
            source_hash = metadata.get("source_hash")
            if source_path and source_hash:
                hashes[source_path] = source_hash
    return hashes


def upsert_source_chunks(
    source_path: str,
    chunks: list[dict[str, Any]],
) -> int:
    """Index a source's chunks and remove obsolete chunk IDs for that source."""
    collection = get_collection()
    previous = collection.get(
        where={"source_path": source_path},
        include=["metadatas"],
    )
    previous_ids = set(previous.get("ids") or [])

    if chunks:
        embeddings = embed_documents([chunk["text"] for chunk in chunks])
        ids = [chunk["id"] for chunk in chunks]
        collection.upsert(
            ids=ids,
            documents=[chunk["text"] for chunk in chunks],
            metadatas=[chunk["metadata"] for chunk in chunks],
            embeddings=embeddings,
        )
        stale_ids = previous_ids.difference(ids)
        if stale_ids:
            collection.delete(ids=sorted(stale_ids))
        return len(ids)

    if previous_ids:
        collection.delete(ids=sorted(previous_ids))
    return 0


def remove_missing_sources(current_source_paths: set[str]) -> int:
    """Remove indexed sources that no longer exist in the raw knowledge folders."""
    collection = get_collection()
    records = collection.get(include=["metadatas"])
    stale_ids = [
        record_id
        for record_id, metadata in zip(records.get("ids") or [], records.get("metadatas") or [])
        if metadata and metadata.get("source_path") not in current_source_paths
    ]
    if stale_ids:
        collection.delete(ids=stale_ids)
    return len(stale_ids)


def query_vectors(
    queries: list[str],
    top_k: int,
    category: str | None = None,
) -> list[dict[str, Any]]:
    """Search selected queries, using explicit E5 vectors and optional category filters."""
    if not queries or top_k < 1:
        return []

    collection = get_collection()
    if collection.count() == 0:
        return []

    from backend.rag.embeddings import embed_queries

    query_embeddings = embed_queries(queries)
    where = {"category": category} if category else None
    results = []
    for query, embedding in zip(queries, query_embeddings):
        arguments = {
            "query_embeddings": [embedding],
            "n_results": min(top_k, collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            arguments["where"] = where
        response = collection.query(**arguments)

        ids = (response.get("ids") or [[]])[0]
        documents = (response.get("documents") or [[]])[0]
        metadatas = (response.get("metadatas") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]
        for record_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            results.append(
                {
                    "id": record_id,
                    "text": document or "",
                    "metadata": metadata or {},
                    "relevance_score": max(0.0, min(1.0, 1.0 - float(distance))),
                    "query": query,
                }
            )
    return results


def validate_knowledge_base() -> int:
    """Fail startup unless the persisted collection is populated and retrievable."""
    collection = get_collection()
    count = int(collection.count())
    if count == 0:
        raise VectorStoreError(
            f"Chroma collection {COLLECTION_NAME!r} is empty. "
            "Deploy the prebuilt medical knowledge database."
        )

    records = collection.get(include=["metadatas"])
    categories = {
        metadata.get("category")
        for metadata in records.get("metadatas") or []
        if metadata and metadata.get("category")
    }
    missing_categories = sorted(set(REQUIRED_CATEGORY_QUERIES) - categories)
    if missing_categories:
        raise VectorStoreError(
            "The medical knowledge collection is missing categories: "
            + ", ".join(missing_categories)
        )

    for category, query in REQUIRED_CATEGORY_QUERIES.items():
        results = query_vectors([query], top_k=1, category=category)
        if (
            not results
            or not results[0]["text"].strip()
            or results[0]["metadata"].get("category") != category
        ):
            raise VectorStoreError(
                f"Medical knowledge retrieval returned no passage for {category}."
            )

    logger.info(
        "Medical knowledge base is ready (chunks=%d, categories=%s).",
        count,
        ",".join(sorted(categories)),
    )
    return count


def validate_knowledge_base_metadata() -> int:
    """Validate the persisted collection without loading the embedding model."""
    collection = get_collection()
    count = int(collection.count())

    if count == 0:
        raise VectorStoreError(
            f"Chroma collection {COLLECTION_NAME!r} is empty."
        )

    records = collection.get(include=["metadatas"])
    categories = {
        metadata.get("category")
        for metadata in records.get("metadatas") or []
        if metadata and metadata.get("category")
    }

    missing_categories = sorted(set(REQUIRED_CATEGORY_QUERIES) - categories)

    if missing_categories:
        raise VectorStoreError(
            "The medical knowledge collection is missing categories: "
            + ", ".join(missing_categories)
        )

    logger.info(
        "Medical knowledge base metadata is ready (chunks=%d, categories=%s).",
        count,
        ",".join(sorted(categories)),
    )

    return count


def _reset_store_for_tests() -> None:
    """Reset only the in-process handles; intended for isolated tests."""
    global _client, _collection
    with _store_lock:
        client = _client
        _client = None
        _collection = None
        if client is not None:
            close = getattr(client, "close", None)
            if callable(close):
                close()
            else:
                system = getattr(client, "_system", None)
                stop = getattr(system, "stop", None)
                if callable(stop):
                    stop()
