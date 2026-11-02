"""Grounded Retrieval & Similarity Calibration Engine for Nykaa Support Agent.

Implements:
1. Vector similarity search against ChromaDB collections.
2. Empirical calibration of the 'I don't know' similarity threshold tau.
3. Grounded answer generation using exclusively retrieved context.
4. Fallback refusal when similarity < tau.
"""

import os
import sys
from typing import Dict, List, Any, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rag.indexer import get_chroma_client, COLLECTION_FIXED, COLLECTION_SENTENCE
from rag.embedder import embed_text

FALLBACK_ANSWER = (
    "I do not have sufficient information in the Nykaa knowledge base to answer "
    "this question. Please contact Nykaa customer support."
)

# Empirically calibrated default threshold (populated via calibrate_threshold)
DEFAULT_CALIBRATED_THRESHOLD = 0.58


def query_collection(
    collection_name: str,
    query_text: str,
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """Queries a ChromaDB collection and returns retrieved chunks with cosine similarity."""
    client = get_chroma_client()
    collection = client.get_collection(name=collection_name)

    query_embedding = embed_text(query_text)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    if results["ids"] and results["ids"][0]:
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i]
            # ChromaDB cosine distance: distance = 1 - cosine_similarity
            similarity = max(0.0, min(1.0, 1.0 - distance))
            chunks.append({
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "doc_id": results["metadatas"][0][i].get("doc_id"),
                "topic": results["metadatas"][0][i].get("topic"),
                "strategy": results["metadatas"][0][i].get("strategy"),
                "similarity": round(similarity, 4),
                "distance": round(distance, 4)
            })

    return chunks


def calibrate_threshold(collection_name: str = COLLECTION_SENTENCE) -> Dict[str, Any]:
    """Empirically calibrates the fallback threshold tau between in-scope and out-of-scope queries."""
    in_scope_queries = [
        "What is the return window for skincare and lipsticks?",
        "How long does a COD refund take to reach my bank account?",
        "Can I exchange footwear or apparel for a different size?"
    ]
    out_of_scope_queries = [
        "How do I fix a flat tire on a motorcycle?",
        "Explain quantum computing algorithms in cryptography."
    ]

    print("=" * 70)
    print(f"EMPIRICAL THRESHOLD CALIBRATION ON '{collection_name}' (Task 4)")
    print("-" * 70)

    in_scope_scores = []
    print("In-Scope Queries:")
    for q in in_scope_queries:
        chunks = query_collection(collection_name, q, top_k=1)
        sim = chunks[0]["similarity"] if chunks else 0.0
        in_scope_scores.append(sim)
        print(f"  [+] '{q}' -> Top-1 Sim: {sim:.4f}")

    out_of_scope_scores = []
    print("\nOut-of-Scope Queries:")
    for q in out_of_scope_queries:
        chunks = query_collection(collection_name, q, top_k=1)
        sim = chunks[0]["similarity"] if chunks else 0.0
        out_of_scope_scores.append(sim)
        print(f"  [-] '{q}' -> Top-1 Sim: {sim:.4f}")

    min_in_scope = min(in_scope_scores)
    max_out_of_scope = max(out_of_scope_scores)

    # Calibrate midpoint threshold between the two observed clusters
    calibrated_tau = round((min_in_scope + max_out_of_scope) / 2.0, 4)

    print("-" * 70)
    print(f"Minimum In-Scope Similarity    : {min_in_scope:.4f}")
    print(f"Maximum Out-of-Scope Similarity: {max_out_of_scope:.4f}")
    print(f"Separation Margin              : {min_in_scope - max_out_of_scope:.4f}")
    print(f"CALIBRATED THRESHOLD (tau)     : {calibrated_tau:.4f}")
    print("=" * 70)

    assert min_in_scope > max_out_of_scope, "In-scope and out-of-scope similarity clusters overlap!"

    return {
        "collection_name": collection_name,
        "min_in_scope": min_in_scope,
        "max_out_of_scope": max_out_of_scope,
        "calibrated_tau": calibrated_tau,
        "in_scope_scores": in_scope_scores,
        "out_of_scope_scores": out_of_scope_scores
    }


def generate_grounded_answer(
    query: str,
    collection_name: str = COLLECTION_SENTENCE,
    threshold: float = DEFAULT_CALIBRATED_THRESHOLD,
    top_k: int = 3
) -> Dict[str, Any]:
    """Retrieves context and generates a grounded answer strictly constrained to the text."""
    chunks = query_collection(collection_name, query, top_k=top_k)
    top_similarity = chunks[0]["similarity"] if chunks else 0.0

    if top_similarity < threshold:
        return {
            "query": query,
            "answer": FALLBACK_ANSWER,
            "grounded": False,
            "fallback_triggered": True,
            "top_similarity": top_similarity,
            "threshold": threshold,
            "retrieved_chunks": chunks
        }

    # Synthesize answer directly from the top retrieved chunks
    unique_texts = []
    seen = set()
    for c in chunks:
        t = c["text"].strip()
        if t not in seen:
            seen.add(t)
            unique_texts.append(t)

    synthesized_answer = " ".join(unique_texts)

    return {
        "query": query,
        "answer": synthesized_answer,
        "grounded": True,
        "fallback_triggered": False,
        "top_similarity": top_similarity,
        "threshold": threshold,
        "retrieved_chunks": chunks
    }


def run_grounded_generation_demo(
    collection_name: str = COLLECTION_SENTENCE,
    threshold: Optional[float] = None
):
    """Demonstrates grounded generation on >= 5 in-scope queries + 1 out-of-scope query."""
    if threshold is None:
        calib = calibrate_threshold(collection_name)
        threshold = calib["calibrated_tau"]

    demo_queries = [
        # 5 In-scope queries
        ("What is the return window for makeup and beauty products?", True),
        ("How long do I have to wait to receive my COD refund?", True),
        ("Are size exchanges allowed for shoes and clothes?", True),
        ("What should I do if my package arrived damaged?", True),
        ("Can perfumes or aerosol sprays be shipped internationally?", True),
        # 1 Out-of-scope query
        ("What is the fastest route to hike Mount Everest without oxygen?", False)
    ]

    print("\n" + "=" * 75)
    print(f"GROUNDED GENERATION DEMO (Task 4) [tau = {threshold:.4f}]")
    print("=" * 75)

    for idx, (query, expected_in_scope) in enumerate(demo_queries, 1):
        res = generate_grounded_answer(query, collection_name=collection_name, threshold=threshold)
        status_label = "IN-SCOPE (Grounded)" if res["grounded"] else "OUT-OF-SCOPE (Fallback Triggered)"
        print(f"\nQuery #{idx}: \"{query}\"")
        print(f"Top-1 Similarity: {res['top_similarity']:.4f} (Threshold: {threshold:.4f})")
        print(f"Outcome: {status_label}")
        print(f"Answer: {res['answer']}")
        print("-" * 75)

        if expected_in_scope:
            assert res["grounded"] is True, f"Query '{query}' was expected to be grounded"
        else:
            assert res["fallback_triggered"] is True, f"Out-of-scope query '{query}' did not trigger fallback"

    print("\n[SUCCESS] Grounded generation successfully demonstrated on 5 in-scope + 1 out-of-scope queries.")


if __name__ == "__main__":
    run_grounded_generation_demo()

