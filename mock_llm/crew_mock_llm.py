"""Deterministic Mock LLM Engine for CrewAI (Task 7).

Extends crewai.llms.base_llm.BaseLLM.
Runs 100% locally with zero external API keys and zero outbound network calls.

Addressed Pitfalls (from Capstone Guidelines):
1. Avoids matching the system-prompt template string 'Observation: the result of the action'.
   Only parses genuine tool observations generated during the execution loop.
2. Dispatches tool arguments strictly by declared argument schema (e.g. checking
   for 'record_id' vs 'query' fields in args_schema), NEVER by substring-matching
   the tool's name (which would misclassify tools like 'rag_lookup').
"""

import os
import re
import json
from typing import Any, List, Dict, Optional, Type
from pydantic import Field, BaseModel

# Enforce telemetry suppression before importing or running CrewAI
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

from crewai.llms.base_llm import BaseLLM


class CrewMockLLM(BaseLLM):
    """Deterministic Mock LLM adhering to CrewAI's BaseLLM interface."""
    model: str = Field(default="mock-nykaa-domain-llm")

    def supports_stop_words(self) -> bool:
        return True

    def get_context_window_size(self) -> int:
        return 8192

    def _extract_query_and_order_id(self, text: str) -> Dict[str, Optional[str]]:
        """Extracts customer query and order identifier from task description."""
        order_match = re.search(r"\b(ORD-\d{4})\b", text, re.IGNORECASE)
        order_id = order_match.group(1).upper() if order_match else None

        # Clean query extraction
        clean_text = text
        for prefix in ["Customer Inquiry:", "Query:", "Task:", "Task Description:"]:
            if prefix in clean_text:
                clean_text = clean_text.split(prefix)[-1]

        return {
            "order_id": order_id,
            "query": clean_text.strip()
        }

    def _get_tool_dispatch(self, tools: List[Any], task_text: str) -> Optional[Dict[str, Any]]:
        """Determines which tool to call based STRICTLY on declared args_schema, NOT name substrings."""
        extracted = self._extract_query_and_order_id(task_text)

        for tool in tools:
            schema: Optional[Type[BaseModel]] = getattr(tool, "args_schema", None)
            if not schema:
                continue

            fields = getattr(schema, "model_fields", getattr(schema, "__fields__", {}))

            # Dispatch Schema 1: Order lookup tools requiring 'record_id'
            if "record_id" in fields:
                record_id = extracted["order_id"] or "ORD-1002"
                return {
                    "tool_name": tool.name,
                    "action_input": {"record_id": record_id}
                }

            # Dispatch Schema 2: RAG retrieval tools requiring 'query'
            if "query" in fields or "query_text" in fields:
                query_field = "query" if "query" in fields else "query_text"
                query_val = extracted["query"] or "What is the return window for beauty products?"
                # Extract first sentence or question if multiple lines
                first_line = query_val.split("\n")[0].strip()
                return {
                    "tool_name": tool.name,
                    "action_input": {query_field: first_line}
                }

        return None

    def _detect_genuine_observation(self, prompt_text: str) -> Optional[str]:
        """Detects genuine tool observation while strictly ignoring the system prompt template.

        Pitfall 1 safeguard: CrewAI's prompt template contains:
        'Observation: the result of the action'
        We ignore this exact template text and only accept genuine observations.
        """
        lines = prompt_text.splitlines()
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("Observation:") and "the result of the action" not in stripped:
                obs_body = stripped.replace("Observation:", "").strip()
                # If multi-line observation, gather subsequent non-empty lines
                remaining = []
                for sub_line in lines[idx + 1:]:
                    if sub_line.strip().startswith("Thought:") or sub_line.strip().startswith("Action:"):
                        break
                    remaining.append(sub_line)
                full_obs = obs_body + "\n" + "\n".join(remaining)
                return full_obs.strip()
        return None

    def call(
        self,
        messages: Any,
        tools: Optional[List[Any]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
        from_task: Optional[Any] = None,
        from_agent: Optional[Any] = None,
        response_model: Optional[Type[BaseModel]] = None
    ) -> str:
        """Processes agent prompt and outputs valid ReAct tool actions or final answers."""
        # Normalize prompt into single text buffer
        if isinstance(messages, list):
            prompt_text = "\n".join(
                getattr(m, "content", str(m)) for m in messages
            )
        else:
            prompt_text = str(messages)

        # Check for genuine observation (Pitfall 1 safeguard)
        observation = self._detect_genuine_observation(prompt_text)

        # Case 1: An actual tool observation was returned. Formulate the Final Answer.
        if observation:
            try:
                obs_data = json.loads(observation)
                if "status" in obs_data and "order_value_inr" in obs_data:
                    # Order lookup result
                    escalate_note = (
                        "Escalation has been flagged for priority support."
                        if obs_data.get("escalation_recommended")
                        else "No immediate escalation required."
                    )
                    ans = (
                        f"Order {obs_data.get('record_id')} is currently '{obs_data.get('status')}'. "
                        f"Total order value is INR {obs_data.get('order_value_inr')}. "
                        f"Escalation Score: {obs_data.get('escalation_score', 0.0):.4f}. {escalate_note}"
                    )
                elif "answer" in obs_data:
                    # RAG policy result
                    ans = obs_data["answer"]
                else:
                    ans = str(obs_data)
            except Exception:
                ans = observation

            return (
                f"Thought: I now have the genuine findings from the tool execution.\n"
                f"Final Answer: {ans}"
            )

        # Case 2: Agent has tools available and needs to execute a tool action.
        if tools and len(tools) > 0:
            dispatch = self._get_tool_dispatch(tools, prompt_text)
            if dispatch:
                action_name = dispatch["tool_name"]
                action_input_str = json.dumps(dispatch["action_input"])
                return (
                    f"Thought: I need to use the authorized tool to gather verified domain facts.\n"
                    f"Action: {action_name}\n"
                    f"Action Input: {action_input_str}"
                )

        # Case 3: Pure Composer Agent or fallback without tools.
        # Synthesize from context in prompt
        extracted = self._extract_query_and_order_id(prompt_text)
        if extracted["order_id"]:
            final_resp = (
                f"Based on the verified order records, order {extracted['order_id']} is being processed. "
                f"Our Nykaa customer support team has reviewed the status and will ensure prompt delivery."
            )
        else:
            final_resp = (
                f"Nykaa provides dedicated customer support under verified retail SLAs. "
                f"Please refer to the specific category return window and COD guidelines."
            )

        return (
            f"Thought: I have synthesized the relevant details into a complete, professional response.\n"
            f"Final Answer: {final_resp}"
        )

