"""CrewAI Multi-Agent Orchestrator for Nykaa Domain Support (Task 7, 8, 9, 10).

Integrates:
- 3 CrewAI Agents: Retrieval, Lookup, Response Composer.
- Memory integration with LangChain InMemoryChatMessageHistory.
- Structured Pydantic response validation (AgentResponse).
- Input Guardrail (PII masking, Injection prevention) and Output Guardrail.
- Telemetry suppression guaranteed.
"""

import os
import sys
import re
from typing import Dict, Any, Optional, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Enforce telemetry deactivation
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

from crewai import Crew, Process
from crew.agents import (
    create_retrieval_agent,
    create_lookup_agent,
    create_response_composer_agent,
    _SHARED_MOCK_LLM
)
from crew.tasks import create_retrieval_task, create_lookup_task, create_composer_task
from crew.memory import (
    add_user_message,
    add_ai_message,
    format_session_history,
    extract_contextual_order_id
)
from models.response_models import AgentResponse
from guardrails.input_guard import process_input_guardrail
from guardrails.output_guard import verify_output_groundedness
from tools.order_lookup import check_order_status
from rag.retriever import query_collection, COLLECTION_SENTENCE, FALLBACK_ANSWER
from governance.governance_policy import (
    LeastAutonomyGuard,
    RuntimeBudgetGuard,
    GovernanceViolationError
)
from cache.query_cache import get_cached_response, store_cached_response


def run_nykaa_agent(
    raw_query: str,
    session_id: str = "default_session"
) -> AgentResponse:
    """Executes the complete multi-agent pipeline for a customer query.

    Steps:
    1. Input Guardrail: PII masking and prompt injection detection.
    2. Context & Memory: Resolve references (e.g. order IDs) from session history.
    3. Multi-Agent Crew: Assemble specialized agents and tasks, execute via .kickoff().
    4. Structured Validation: Validate response against AgentResponse Pydantic schema.
    5. Output Guardrail & Memory Persistence.
    """
    # 0. Runtime Governance Budget Check (Task 15)
    try:
        RuntimeBudgetGuard.check_request_budget(raw_query)
    except GovernanceViolationError as gov_err:
        return AgentResponse(
            query=raw_query,
            status="refused",
            answer=str(gov_err),
            confidence=0.0
        )

    # 1. Input Guardrail
    guard_result = process_input_guardrail(raw_query)
    if not guard_result["allowed"]:
        refusal_resp = AgentResponse(
            query=raw_query,
            status="refused",
            answer=f"Request rejected by safety guardrail: {guard_result['rejection_reason']}",
            confidence=0.0
        )
        return refusal_resp

    sanitized_query = guard_result["sanitized_query"]

    # 2. Context & Session Memory
    resolved_order_id = extract_contextual_order_id(session_id, sanitized_query)

    # 2.1 In-Memory Response Caching Check for Grounded Policy Inquiries (Task 16)
    # Order lookups depend on session/order context and bypass general policy cache
    if not resolved_order_id:
        cached = get_cached_response(sanitized_query)
        if cached is not None and isinstance(cached, AgentResponse):
            return cached

    # 3. Assemble Agents with Least Autonomy Verification (Task 15)
    retrieval_agent = create_retrieval_agent()
    lookup_agent = create_lookup_agent()
    composer_agent = create_response_composer_agent()

    # Enforce Application Layer: check_order_status authorized ONLY for lookup agent
    LeastAutonomyGuard.validate_tool_wiring(lookup_agent.role, "check_order_status")

    agents = [retrieval_agent, lookup_agent, composer_agent]
    tasks = []
    task_context = []
    sources = []
    escalation_score = None
    escalation_required = False

    # Determine query intent: Order Lookup vs Policy Question vs Mixed
    is_order_query = bool(resolved_order_id)
    is_policy_query = not is_order_query or any(
        kw in sanitized_query.lower() for kw in ["return", "policy", "refund", "sla", "warranty", "exchange"]
    )

    if is_order_query:
        lookup_task = create_lookup_task(lookup_agent, resolved_order_id)
        tasks.append(lookup_task)
        task_context.append(lookup_task)
        sources.append(resolved_order_id)

        # Lookup details for metadata
        order_info = check_order_status(resolved_order_id)
        escalation_score = order_info.get("escalation_score")
        escalation_required = order_info.get("escalation_recommended", False)

    if is_policy_query or not is_order_query:
        retrieval_task = create_retrieval_task(retrieval_agent, sanitized_query)
        tasks.append(retrieval_task)
        task_context.append(retrieval_task)

        # Retrieve top chunks for source tracking and groundedness verification
        top_chunks = query_collection(COLLECTION_SENTENCE, sanitized_query, top_k=2)
        for c in top_chunks:
            if c.get("doc_id") and c["doc_id"] not in sources:
                sources.append(c["doc_id"])
    else:
        top_chunks = []

    composer_task = create_composer_task(composer_agent, sanitized_query, task_context)
    tasks.append(composer_task)

    # 4. Multi-Agent Orchestration via .kickoff()
    crew = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=False
    )

    crew_output = crew.kickoff()
    raw_answer = str(crew_output).strip()

    # Clean any "Final Answer:" prefix from mock LLM output
    if "Final Answer:" in raw_answer:
        raw_answer = raw_answer.split("Final Answer:")[-1].strip()

    # 5. Output Guardrail (for policy queries)
    if is_policy_query and top_chunks:
        output_guard = verify_output_groundedness(raw_answer, top_chunks)
        final_answer = output_guard["final_answer"]
        if not output_guard["grounded"]:
            status = "refused"
        elif escalation_required:
            status = "escalated"
        else:
            status = "answered"
    else:
        final_answer = raw_answer
        status = "escalated" if escalation_required else "answered"

    # 6. Structured Output Validation (Task 9)
    structured_response = AgentResponse(
        query=raw_query,
        status=status,
        answer=final_answer,
        escalation_score=escalation_score,
        escalation_required=escalation_required,
        retrieved_sources=sources,
        confidence=1.0 if status != "refused" else 0.0
    )

    # 7. Update Session Memory & In-Memory Response Cache for policy queries
    add_user_message(session_id, raw_query)
    add_ai_message(session_id, structured_response.answer)
    if not is_order_query:
        store_cached_response(sanitized_query, structured_response)

    return structured_response


if __name__ == "__main__":
    print("--- Demo 1: Policy Question (Invoking RAG Tool) ---")
    r1 = run_nykaa_agent("What is the return window for lipsticks and skincare?")
    print("Status:", r1.status)
    print("Answer:", r1.answer)
    print("Sources:", r1.retrieved_sources)

    print("\n--- Demo 2: Order Lookup (Invoking Order Lookup Tool) ---")
    r2 = run_nykaa_agent("Can you check the delivery status for order ORD-1002?")
    print("Status:", r2.status)
    print("Answer:", r2.answer)
    print("Escalation Score:", r2.escalation_score)
    print("Escalation Required:", r2.escalation_required)
