"""Task Definitions for Nykaa Domain Support Crew (Task 7)."""

from typing import List, Optional
from crewai import Task, Agent


def create_retrieval_task(agent: Agent, query: str) -> Task:
    """Task for policy knowledge retrieval."""
    return Task(
        description=f"Query the Nykaa policy database for relevant guidelines regarding: '{query}'",
        expected_output="Grounded policy details retrieved from the knowledge base.",
        agent=agent
    )


def create_lookup_task(agent: Agent, record_id: str) -> Task:
    """Task for order record lookup and escalation scoring."""
    return Task(
        description=f"Look up order status, shipping delay, and escalation score for order record '{record_id}'",
        expected_output="JSON details of order status and escalation risk score.",
        agent=agent
    )


def create_composer_task(agent: Agent, query: str, context_tasks: Optional[List[Task]] = None) -> Task:
    """Task for response synthesis."""
    return Task(
        description=(
            f"Draft a polite, empathetic, and fully grounded response for customer query: '{query}'. "
            f"Incorporate findings from specialists without making unverified claims."
        ),
        expected_output="Final customer-ready draft answer adhering to verified policies and order facts.",
        agent=agent,
        context=context_tasks or []
    )

