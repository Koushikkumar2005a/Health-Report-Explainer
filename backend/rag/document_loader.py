"""Load and clean PDF and TXT medical reference documents."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)
SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


@dataclass(frozen=True)
class DocumentPage:
    page_number: int
    text: str


@dataclass(frozen=True)
class MedicalDocument:
    path: Path
    source_path: str
    source_filename: str
    category: str
    source_hash: str
    pages: tuple[DocumentPage, ...]


def clean_text(text: str) -> str:
    """Normalize whitespace while retaining line and table boundaries."""
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    cleaned = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def extract_pages(path: Path) -> list[DocumentPage]:
    """Extract page-numbered text from a supported reference document."""
    extension = path.suffix.lower()
    if extension == ".pdf":
        try:
            import pymupdf
        except ImportError as exc:
            raise RuntimeError("PyMuPDF is required to read PDF references.") from exc

        pages = []
        with pymupdf.open(path) as pdf:
            if pdf.is_encrypted:
                raise ValueError("The PDF is encrypted.")
            for index, page in enumerate(pdf, start=1):
                text = clean_text(page.get_text("text"))
                if text:
                    pages.append(DocumentPage(page_number=index, text=text))
        return pages

    if extension == ".txt":
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            text = path.read_text(encoding="cp1252")
        cleaned = clean_text(text)
        return [DocumentPage(page_number=1, text=cleaned)] if cleaned else []

    raise ValueError(f"Unsupported reference file type: {extension or '(none)'}")


def discover_documents(raw_root: Path) -> list[Path]:
    """Find supported files recursively and report unsupported files without failing."""
    if not raw_root.exists():
        return []

    supported = []
    for path in sorted(item for item in raw_root.rglob("*") if item.is_file()):
        if path.name == ".gitkeep":
            continue
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            supported.append(path)
        else:
            logger.warning(
                "Skipping unsupported medical reference file: %s",
                path.relative_to(raw_root).as_posix(),
            )
    return supported


def load_document(path: Path, raw_root: Path) -> MedicalDocument:
    """Extract a supported document and assign its top-level category."""
    source_path = path.relative_to(raw_root).as_posix()
    category = Path(source_path).parts[0] if len(Path(source_path).parts) > 1 else "Uncategorized"
    hasher = hashlib.sha256()
    with path.open("rb") as source_file:
        for block in iter(lambda: source_file.read(1024 * 1024), b""):
            hasher.update(block)
    pages = tuple(extract_pages(path))
    return MedicalDocument(
        path=path,
        source_path=source_path,
        source_filename=path.name,
        category=category,
        source_hash=hasher.hexdigest(),
        pages=pages,
    )
