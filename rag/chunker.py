"""Chunking strategies for Nykaa Knowledge Base documents.

Strategy A: Fixed-size chunks with character overlap.
Strategy B: Sentence-based chunks using punctuation delimiters.
"""

import re
from typing import List, Dict, Any


def chunk_fixed_size_overlap(
    text: str,
    chunk_size: int = 150,
    overlap: int = 30
) -> List[str]:
    """Splits text into fixed character-length chunks with specified overlap.

    Args:
        text: Input document string.
        chunk_size: Maximum character length per chunk.
        overlap: Character overlap between consecutive chunks.

    Returns:
        List of text chunks.
    """
    clean_text = " ".join(text.strip().split())
    if len(clean_text) <= chunk_size:
        return [clean_text]

    chunks = []
    step = chunk_size - overlap
    start = 0

    while start < len(clean_text):
        end = min(start + chunk_size, len(clean_text))
        chunk = clean_text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(clean_text):
            break
        start += step

    return chunks


def chunk_by_sentences(text: str) -> List[str]:
    """Splits text into sentence-based chunks using sentence-ending punctuation.

    Args:
        text: Input document string.

    Returns:
        List of individual sentence chunks.
    """
    clean_text = " ".join(text.strip().split())
    # Match sentences ending with period, exclamation, or question mark
    sentences = re.split(r'(?<=[.!?])\s+', clean_text)
    return [s.strip() for s in sentences if s.strip()]


def prepare_chunks(
    documents: List[Dict[str, Any]],
    strategy: str
) -> List[Dict[str, Any]]:
    """Chunks documents according to the chosen strategy and attaches metadata.

    Args:
        documents: List of document dicts with keys 'doc_id', 'topic', 'content'.
        strategy: 'fixed_size' or 'sentence'.

    Returns:
        List of chunk dicts ready for embedding and indexing.
    """
    chunk_records = []

    for doc in documents:
        doc_id = doc["doc_id"]
        topic = doc["topic"]
        content = doc["content"]

        if strategy == "fixed_size":
            raw_chunks = chunk_fixed_size_overlap(content, chunk_size=150, overlap=30)
        elif strategy == "sentence":
            raw_chunks = chunk_by_sentences(content)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

        for idx, chunk_text in enumerate(raw_chunks):
            chunk_records.append({
                "chunk_id": f"{doc_id}_{strategy}_{idx}",
                "doc_id": doc_id,
                "topic": topic,
                "strategy": strategy,
                "chunk_index": idx,
                "text": chunk_text
            })

    return chunk_records

