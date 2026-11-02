"""Agent Definitions for Nykaa Domain Support Crew (Task 7).

Defines 3 specialized CrewAI agents:
1. Retrieval Agent (equipped with NykaaRAGTool)
2. Lookup Agent (equipped exclusively with NykaaOrderLookupTool)
3. Response Composer Agent (pure synthesis, no tools)
"""

import os
import sys
from crewai import Agent

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mock_llm.crew_mock_llm import CrewMockLLM
from tools.crew_tools import NykaaOrderLookupTool, NykaaRAGTool

# Shared deterministic local Mock LLM instance
_SHARED_MOCK_LLM = CrewMockLLM(model="mock-nykaa-domain-llm")


def create_retrieval_agent(llm: CrewMockLLM = _SHARED_MOCK_LLM) -> Agent:
    """Creates the Policy Retrieval Agent equipped with the RAG tool."""
    return Agent(
        role="Nykaa Policy Specialist",
        goal="Retrieve verified policy facts, return windows, SLAs, and guidelines from the Nykaa knowledge base.",
        backstory=(
            "You are a policy expert for Nykaa E-commerce. You specialize in accurate policy retrieval "
            "and refuse to hallucinate unsupported terms."
        ),
        tools=[NykaaRAGTool()],
        llm=llm,
        verbose=False,
        allow_delegation=False
    )


def create_lookup_agent(llm: CrewMockLLM = _SHARED_MOCK_LLM) -> Agent:
    """Creates the Order Lookup Agent equipped exclusively with the lookup tool (Least Autonomy)."""
    return Agent(
        role="Nykaa Order Fulfillment Specialist",
        goal="Look up order statuses, values, delivery delays, and compute escalation scores.",
        backstory=(
            "You are an order investigator with exclusive clearance to query the internal order records. "
            "You calculate escalation risks to identify orders requiring human supervisor escalation."
        ),
        tools=[NykaaOrderLookupTool()],
        llm=llm,
        verbose=False,
        allow_delegation=False
    )


def create_response_composer_agent(llm: CrewMockLLM = _SHARED_MOCK_LLM) -> Agent:
    """Creates the Response Composer Agent that synthesizes findings into a customer-facing response."""
    return Agent(
        role="Nykaa Customer Support Lead",
        goal="Synthesize policy facts and order status into an empathetic, professional, grounded final answer.",
        backstory=(
            "You are a seasoned customer support lead at Nykaa. You review findings from your specialists "
            "and draft clear, actionable, policy-compliant responses for customers."
        ),
        tools=[],  # Pure composer - no direct tool execution
        llm=llm,
        verbose=False,
        allow_delegation=False
    )

