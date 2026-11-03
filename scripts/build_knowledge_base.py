"""Build or incrementally update the persistent medical reference collection."""

from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import settings
from backend.rag.document_loader import discover_documents, load_document
from backend.rag.text_chunker import chunk_text
from backend.rag.vector_store import (
    get_indexed_source_hashes,
    remove_missing_sources,
    upsert_source_chunks,
)

RAW_ROOT = PROJECT_ROOT / "data" / "medical_knowledge" / "raw"
CHUNK_SIZE = settings.RAG_CHUNK_SIZE
CHUNK_OVERLAP = settings.RAG_CHUNK_OVERLAP

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _chunk_id(source_path: str, page_number: int, chunk_number: int) -> str:
    value = f"{source_path}\0{page_number}\0{chunk_number}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def build_chunks(document) -> list[dict]:
    chunks = []
    for page in document.pages:
        page_chunks = chunk_text(page.text, CHUNK_SIZE, CHUNK_OVERLAP)
        for chunk_number, text in enumerate(page_chunks, start=1):
            chunks.append(
                {
                    "id": _chunk_id(document.source_path, page.page_number, chunk_number),
                    "text": text,
                    "metadata": {
                        "source_path": document.source_path,
                        "source_filename": document.source_filename,
                        "category": document.category,
                        "page_number": page.page_number,
                        "chunk_id": f"{page.page_number}-{chunk_number}",
                        "source_hash": document.source_hash,
                    },
                }
            )
    return chunks


def main() -> int:
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    files = discover_documents(RAW_ROOT)
    if not files:
        print(
            "No PDF or TXT medical reference documents were found. "
            f"Add files under {RAW_ROOT} and run this command again."
        )
        return 0

    try:
        indexed_hashes = get_indexed_source_hashes()
    except Exception:
        logger.exception("Could not open the persistent medical knowledge base.")
        return 1

    current_source_paths = {
        path.relative_to(RAW_ROOT).as_posix()
        for path in files
    }
    indexed = 0
    skipped = 0
    failed = 0
    for path in files:
        source_path = path.relative_to(RAW_ROOT).as_posix()
        try:
            document = load_document(path, RAW_ROOT)
            if indexed_hashes.get(source_path) == document.source_hash:
                skipped += 1
                continue
            chunks = build_chunks(document)
            count = upsert_source_chunks(source_path, chunks)
            if count:
                indexed += count
                logger.info("Indexed %s (%s chunks).", source_path, count)
            else:
                logger.warning("No extractable text found in %s; old chunks removed.", source_path)
        except Exception:
            failed += 1
            logger.exception("Unable to process medical reference file: %s", source_path)

    try:
        removed = remove_missing_sources(current_source_paths)
    except Exception:
        logger.exception("Could not remove references that no longer exist.")
        return 1

    print(
        f"Knowledge-base build complete: {indexed} chunks indexed, "
        f"{skipped} unchanged files skipped, {removed} stale chunks removed, "
        f"{failed} files failed."
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
