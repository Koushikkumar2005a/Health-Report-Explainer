"""Sentence- and paragraph-aware chunking for reference text."""

from __future__ import annotations


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    """Split text into readable chunks with a bounded character overlap."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be greater than zero.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")

    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            minimum_boundary = start + int(chunk_size * 0.6)
            boundaries = [
                text.rfind(separator, minimum_boundary, end)
                for separator in ("\n\n", "\n", ". ", "? ", "! ", " ")
            ]
            boundary = max(boundaries)
            if boundary > start:
                end = boundary + (2 if text[boundary : boundary + 2] in {". ", "? ", "! ", "\n\n"} else 1)

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break

        next_start = max(start + 1, end - overlap)
        while next_start < end and text[next_start].isspace():
            next_start += 1
        start = next_start

    return chunks
