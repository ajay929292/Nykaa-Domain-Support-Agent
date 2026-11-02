"""RAG Component Unit Tests.

Tests:
- Knowledge base documents and topic coverage (12 topics, 2-5 sentences).
- Chunking strategies (Fixed-size with overlap vs. Sentence-based).
- ChromaDB collections population and vector similarity search.
- Empirically calibrated similarity threshold (tau = 0.3160).
- Grounded answer generation and fallback refusal on out-of-scope query.
- Precision and recall calculation on benchmark queries.
"""

import re
import pytest
from rag.chunker import chunk_fixed_size_overlap, chunk_by_sentences, prepare_chunks
from rag.indexer import (
    load_knowledge_base,
    get_chroma_client,
    COLLECTION_FIXED,
    COLLECTION_SENTENCE
)
from rag.retriever import (
    query_collection,
    generate_grounded_answer,
    calibrate_threshold,
    FALLBACK_ANSWER
)
from rag.eval_chunking import evaluate_collection


def test_knowledge_base_topics_and_sentence_length():
    """Verify KB has at least 12 documents, all required topics, and 2-5 sentences each."""
    docs = load_knowledge_base()
    assert len(docs) >= 12, f"Expected >= 12 documents, found {len(docs)}"

    required_topics = [
        "return window by product category",
        "COD refund timelines",
        "delivery SLAs",
        "reverse-pickup eligibility",
        "warranty terms by category",
        "order-cancellation policy",
        "loyalty-points redemption policy",
        "payment-failure/retry policy",
        "size-exchange policy",
        "damaged-item claim process",
        "international shipping restrictions",
        "customer-support escalation matrix"
    ]

    kb_topics = [d["topic"] for d in docs]
    for topic in required_topics:
        assert topic in kb_topics, f"Missing required KB topic: {topic}"

    for doc in docs:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', doc["content"]) if s.strip()]
        assert 2 <= len(sentences) <= 5, (
            f"Doc '{doc['doc_id']}' has {len(sentences)} sentences (required: 2 to 5)"
        )


def test_chunking_strategies():
    """Verify both chunking algorithms function correctly."""
    sample_text = (
        "Nykaa provides a 15-day return window for unopened beauty products. "
        "Apparel and footwear items are eligible for returns within 7 days. "
        "Personal hygiene products are strictly non-returnable."
    )

    # Strategy A: Fixed-size with overlap
    fixed_chunks = chunk_fixed_size_overlap(sample_text, chunk_size=80, overlap=20)
    assert len(fixed_chunks) >= 2
    for c in fixed_chunks:
        assert len(c) <= 80

    # Strategy B: Sentence-based
    sentence_chunks = chunk_by_sentences(sample_text)
    assert len(sentence_chunks) == 3


def test_chromadb_collections_exist_and_searchable():
    """Verify both collections exist in ChromaDB and return similarity results."""
    client = get_chroma_client()
    col_fixed = client.get_collection(COLLECTION_FIXED)
    col_sentence = client.get_collection(COLLECTION_SENTENCE)

    assert col_fixed.count() > 0
    assert col_sentence.count() > 0

    # Test vector search
    results = query_collection(COLLECTION_SENTENCE, "return window for beauty products", top_k=2)
    assert len(results) == 2
    assert results[0]["similarity"] > 0.40
    assert results[0]["doc_id"] == "KB-01"


def test_calibrated_threshold_discrimination():
    """Verify threshold cleanly separates in-scope from out-of-scope queries."""
    calib = calibrate_threshold(COLLECTION_SENTENCE)
    tau = calib["calibrated_tau"]

    # In-scope must exceed threshold
    in_scope = generate_grounded_answer("How long does COD refund take?", threshold=tau)
    assert in_scope["grounded"] is True
    assert in_scope["fallback_triggered"] is False
    assert in_scope["top_similarity"] >= tau

    # Out-of-scope must trigger fallback
    out_scope = generate_grounded_answer(
        "Explain how to fix a flat tire on a motorcycle",
        threshold=tau
    )
    assert out_scope["grounded"] is False
    assert out_scope["fallback_triggered"] is True
    assert out_scope["answer"] == FALLBACK_ANSWER
    assert out_scope["top_similarity"] < tau


def test_precision_and_recall_metrics():
    """Verify precision and recall benchmark computes valid numbers."""
    eval_sentence = evaluate_collection(COLLECTION_SENTENCE, top_k=3)
    assert eval_sentence["avg_precision"] >= 0.50
    assert eval_sentence["avg_recall"] == 1.00
    assert len(eval_sentence["query_results"]) == 5

