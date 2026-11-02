"""CrewAI Tool Implementations for Nykaa Domain Support Agent.

Tools:
1. NykaaOrderLookupTool: Exclusively for Lookup Agent (arg: record_id)
2. NykaaRAGTool: Exclusively for Retrieval Agent (arg: query)
"""

import os
import sys
import json
from typing import Dict, Any, Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.order_lookup import check_order_status
from rag.retriever import generate_grounded_answer, COLLECTION_SENTENCE


class OrderLookupInput(BaseModel):
    """Input argument schema for order status lookup."""
    record_id: str = Field(
        ...,
        description="The Nykaa order record ID to look up (e.g., 'ORD-1001', 'ORD-1002')."
    )


class RAGQueryInput(BaseModel):
    """Input argument schema for Nykaa policy knowledge retrieval."""
    query: str = Field(
        ...,
        description="The customer's policy inquiry or question regarding Nykaa services."
    )


class NykaaOrderLookupTool(BaseTool):
    """Tool for looking up customer order status and escalation risk."""
    name: str = "check_order_status"
    description: str = (
        "Look up the real-time fulfillment status, monetary value, shipping delay status, "
        "and designed escalation score for a specific Nykaa order using its record_id."
    )
    args_schema: Type[BaseModel] = OrderLookupInput

    def _run(self, record_id: str) -> str:
        res = check_order_status(record_id)
        return json.dumps(res, indent=2)


class NykaaRAGTool(BaseTool):
    """Tool for retrieving verified policy facts from Nykaa knowledge base."""
    name: str = "rag_tool"
    description: str = (
        "Search and retrieve authoritative Nykaa customer policies, SLAs, refund timelines, "
        "and return rules from the ChromaDB vector index."
    )
    args_schema: Type[BaseModel] = RAGQueryInput

    def _run(self, query: str) -> str:
        res = generate_grounded_answer(query, collection_name=COLLECTION_SENTENCE)
        return json.dumps({
            "query": query,
            "answer": res["answer"],
            "grounded": res["grounded"],
            "top_similarity": res["top_similarity"],
            "fallback_triggered": res["fallback_triggered"]
        }, indent=2)

