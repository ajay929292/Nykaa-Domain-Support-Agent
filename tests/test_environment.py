"""Environment & Telemetry Configuration Tests.

Tests:
- Telemetry suppression flags (CREWAI_DISABLE_TELEMETRY, OTEL_SDK_DISABLED).
- Deterministic MOCK_LLM mode flag.
- Core library imports and availability.
"""

import os
import pytest
from dotenv import load_dotenv

load_dotenv()


def test_telemetry_suppressed():
    """Verify telemetry outbound network calls are disabled."""
    assert os.getenv("CREWAI_DISABLE_TELEMETRY") == "true"
    assert os.getenv("OTEL_SDK_DISABLED") == "true"
    assert os.getenv("MOCK_LLM_MODE") == "true"


def test_core_dependencies_importable():
    """Verify core AI and framework modules are installed and functional."""
    import crewai
    import autogen_agentchat
    import langchain_core
    import chromadb
    import sentence_transformers
    import fastapi
    import pydantic

    assert hasattr(crewai, "Agent")
    assert hasattr(fastapi, "FastAPI")
    assert hasattr(pydantic, "BaseModel")

