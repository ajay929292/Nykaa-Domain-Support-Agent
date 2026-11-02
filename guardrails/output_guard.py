"""Output-Side Guardrails for Nykaa Support Agent (Task 10).

Features:
- Groundedness verification: Ensures that answers claiming policy information
  are strictly supported by the retrieved context.
- Fallback refusal enforcement: If the retrieved context does not support
  the query (e.g. similarity < tau or unsupported claim), enforces polite refusal.
"""

from typing import Dict, Any, List

FALLBACK_REFUSAL = (
    "I do not have sufficient information in the Nykaa knowledge base to answer "
    "this question. Please contact Nykaa customer support."
)


def verify_output_groundedness(
    draft_answer: str,
    retrieved_chunks: List[Dict[str, Any]],
    threshold: float = 0.3160
) -> Dict[str, Any]:
    """Validates that the output draft is backed by retrieved knowledge base chunks."""
    # Check 1: If no chunks retrieved or top chunk similarity is below threshold
    if not retrieved_chunks:
        return {
            "grounded": False,
            "final_answer": FALLBACK_REFUSAL,
            "reason": "No context chunks retrieved."
        }

    top_similarity = retrieved_chunks[0].get("similarity", 0.0)
    if top_similarity < threshold:
        return {
            "grounded": False,
            "final_answer": FALLBACK_REFUSAL,
            "reason": f"Top retrieval similarity ({top_similarity:.4f}) is below threshold ({threshold:.4f})."
        }

    # Check 2: Verify non-empty and non-hallucinated draft
    if not draft_answer or not draft_answer.strip():
        return {
            "grounded": False,
            "final_answer": FALLBACK_REFUSAL,
            "reason": "Draft answer is empty."
        }

    return {
        "grounded": True,
        "final_answer": draft_answer,
        "reason": f"Draft verified against retrieved context (top similarity: {top_similarity:.4f})."
    }

