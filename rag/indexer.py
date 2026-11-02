"""ChromaDB Dual Collection Indexer for Nykaa Knowledge Base.

Indexes:
1. 'nykaa_kb_fixed' (Strategy A: Fixed-size with overlap)
2. 'nykaa_kb_sentence' (Strategy B: Sentence-based chunks)

Uses collection.upsert() for deterministic and idempotent population.
"""

import os
import sys
import json
from typing import Dict, List, Any

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import chromadb
from chromadb.config import Settings

from rag.chunker import prepare_chunks
from rag.embedder import embed_texts

CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chroma_db")
KB_JSON_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "knowledge_base.json")

COLLECTION_FIXED = "nykaa_kb_fixed"
COLLECTION_SENTENCE = "nykaa_kb_sentence"


def get_chroma_client() -> chromadb.ClientAPI:
    """Returns a persistent ChromaDB client instance."""
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


def load_knowledge_base(path: str = KB_JSON_PATH) -> List[Dict[str, Any]]:
    """Loads knowledge base JSON documents."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def index_strategy(
    client: chromadb.ClientAPI,
    collection_name: str,
    chunks: List[Dict[str, Any]]
) -> int:
    """Indexes chunks into a specified ChromaDB collection using upsert."""
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    ids = [c["chunk_id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    metadatas = [{
        "doc_id": c["doc_id"],
        "topic": c["topic"],
        "strategy": c["strategy"],
        "chunk_index": c["chunk_index"]
    } for c in chunks]

    embeddings = embed_texts(texts)

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return len(ids)


def build_and_index_both_collections() -> Dict[str, int]:
    """Builds and indexes both fixed-size and sentence-based collections."""
    documents = load_knowledge_base()
    client = get_chroma_client()

    # 1. Strategy A: Fixed-size with overlap
    fixed_chunks = prepare_chunks(documents, strategy="fixed_size")
    fixed_count = index_strategy(client, COLLECTION_FIXED, fixed_chunks)

    # 2. Strategy B: Sentence-based chunks
    sentence_chunks = prepare_chunks(documents, strategy="sentence")
    sentence_count = index_strategy(client, COLLECTION_SENTENCE, sentence_chunks)

    print(f"[Indexer] Successfully indexed '{COLLECTION_FIXED}' with {fixed_count} chunks.")
    print(f"[Indexer] Successfully indexed '{COLLECTION_SENTENCE}' with {sentence_count} chunks.")

    return {
        COLLECTION_FIXED: fixed_count,
        COLLECTION_SENTENCE: sentence_count
    }


if __name__ == "__main__":
    build_and_index_both_collections()
