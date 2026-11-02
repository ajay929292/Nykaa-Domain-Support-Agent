# Phase-Wise Implementation Plan
## Nykaa Domain Support Agent (CrewAI + AutoGen + FastAPI)

Based on the [Architecture Plan](file:///c:/Users/AJAY%20SHARMA/OneDrive/Desktop/AI_PROJECTS/Nykaa_Final_Capstone/doc/architectureplan.md) and [Problem Statement](file:///c:/Users/AJAY%20SHARMA/OneDrive/Desktop/AI_PROJECTS/Nykaa_Final_Capstone/doc/problemstatement.md).

---

## Roadmap Overview

```
Phase 0: Environment & Core Infrastructure Setup
   │
   ▼
Phase 1: Dataset Design & RAG Core Engine (30 Marks)
   │  ├── Seeded Order Generator (dataset.py)
   │  ├── 12-Topic Knowledge Base Authoring
   │  ├── Dual Chunking & ChromaDB Dual Indexing
   │  ├── Empirical Similarity Calibration & Grounded Fallback
   │  └── Precision & Recall Evaluation Benchmark
   │
   ▼
Phase 2: CrewAI Multi-Agent Orchestration & Guardrails (30 Marks)
   │  ├── check_order_status Tool with Escalation Score Formula
   │  ├── Deterministic MOCK_LLM Engine (BaseLLM Subclass)
   │  ├── CrewAI 3-Agent Crew (Retrieval, Lookup, Composer)
   │  ├── Multi-Turn Session Memory (InMemoryChatMessageHistory)
   │  ├── Pydantic Structured Output Schema
   │  └── Input & Output Guardrails (PII Masking, Prompt Injection, Groundedness)
   │
   ▼
Phase 3: FastAPI Deployment, Observability & Evaluation (20 Marks)
   │  ├── FastAPI App (POST /ask, POST /add-document, WebSocket /ws/chat)
   │  ├── Structured JSON-Lines Logger with Trace IDs & PII Redaction
   │  └── 15-Query Evaluation Suite (Accuracy, Grounding, Completeness, Safety)
   │
   ▼
Phase 4: Resilience, AutoGen Review & Governance (20 Marks)
   │  ├── AutoGen RoundRobin Review Stage (StructuredMessage[VerdictModel])
   │  ├── Four-Layer AI Governance (Least Autonomy, Medium Risk, Cost Budget Cap)
   │  ├── In-Memory Normalized Query Caching
   │  └── Comprehensive Test Suite & Grader README Verification
```

---

## Phase 0: Environment Setup & Foundation Setup

### Objectives
Establish a clean, reproducible virtual environment with zero external API dependencies, install pinned dependencies, and ensure telemetry is disabled.

### Implementation Steps
1. **Generate `requirements.txt`**:
   - `crewai` (compatible with BaseLLM subclassing)
   - `autogen-agentchat` / `pyautogen`
   - `fastapi`, `uvicorn`, `websockets`
   - `chromadb`
   - `sentence-transformers`
   - `pydantic` (v2)
   - `langchain`, `langchain-core`, `langchain-community`
   - `pytest`
2. **Environment Configuration**:
   - Create `.env.example` documenting environment variables:
     ```bash
     CREWAI_DISABLE_TELEMETRY=true
     OTEL_SDK_DISABLED=true
     MOCK_LLM_MODE=true
     ```
3. **Verification**:
   - Verify local execution in `.venv` without missing imports.

---

## Phase 1: Dataset Design & RAG Core Engine (30 Marks)

### Objectives
Implement deterministic order dataset generation, write the 12 required Nykaa policy documents, build dual-chunking strategies with local ChromaDB indexing, empirically calibrate retrieval similarity thresholds, and run precision/recall benchmarks.

### Detailed Tasks

#### Task 1.1: Seeded Order Dataset Generator (`data/dataset.py`)
- **File**: `data/dataset.py`
- **Logic**:
  - Set fixed random seed (e.g., `seed = 42`).
  - Vocabulary constraints:
    - `categories`: `['Apparel', 'Electronics', 'Home', 'Footwear', 'Beauty']` (all $\ge 3$ records).
    - `statuses`: `['Placed', 'Shipped', 'Delivered', 'Returned', 'Refunded']` (all $\ge 1$ record).
    - `order_value_inr`: ₹250 to ₹15,000 (realistic retail range).
    - `days_since_created`: 0 to 30.
    - `delayed_shipment`: Boolean.
  - Tune weights so `delayed_shipment` percentage lands strictly between **10% and 30%** (target: ~20%).
  - Total records: $\ge 40$ (e.g., 45 records).
  - Add reporting function:
    - Print category distribution.
    - Print status distribution.
    - Print exact percentage of delayed shipments.
- **Verification**: Run `python data/dataset.py` and assert constraints programmatically.

#### Task 1.2: Nykaa Policy Knowledge Base (`data/knowledge_base.json`)
- **File**: `data/knowledge_base.json`
- **Content**: Author 12 distinct policy documents (2–5 sentences each) covering:
  1. Return window by product category
  2. COD refund timelines
  3. Delivery SLAs
  4. Reverse-pickup eligibility
  5. Warranty terms by category
  6. Order-cancellation policy
  7. Loyalty-points (Nykaa Prive) redemption policy
  8. Payment-failure/retry policy
  9. Size-exchange policy
  10. Damaged-item claim process
  11. International shipping restrictions
  12. Customer-support escalation matrix
- **Verification**: Ensure all 12 documents have unique `doc_id`, `topic`, and `content`.

#### Task 1.3: Dual Chunking & ChromaDB Dual Collections (`rag/chunker.py`, `rag/indexer.py`)
- **Files**:
  - `rag/chunker.py`:
    - `chunk_fixed_size_overlap(text, chunk_size=120, overlap=30) -> List[str]`
    - `chunk_by_sentences(text) -> List[str]`
  - `rag/embedder.py`:
    - SentenceTransformers wrapper (`all-MiniLM-L6-v2`) running locally on CPU.
  - `rag/indexer.py`:
    - Initialize local ChromaDB client (`./data/chroma_db`).
    - Collection 1: `nykaa_kb_fixed` (Strategy A).
    - Collection 2: `nykaa_kb_sentence` (Strategy B).
    - Index all 12 documents into both collections with metadata (`doc_id`, `topic`, `strategy`).
- **Verification**: Verify both collections hold non-zero vectors and query both with a test query.

#### Task 1.4: Grounded Generation & Empirical Threshold Calibration (`rag/retriever.py`)
- **File**: `rag/retriever.py`
- **Empirical Calibration Procedure**:
  - Test 3 In-Scope Queries:
    1. "What is the return window for skincare and lipsticks?"
    2. "How long does a COD refund take to reach my bank account?"
    3. "Can I exchange footwear for another size?"
  - Test 2 Out-of-Scope Queries:
    1. "How do I fix a punctured motorcycle tire?"
    2. "Explain quantum computing algorithms in cryptography."
  - Record Top-1 Cosine Similarities ($S = 1 - \text{distance}$).
  - Choose threshold $\tau$ strictly between $\max(S_{\text{out\_of\_scope}})$ and $\min(S_{\text{in\_scope}})$.
- **Grounded Generator**:
  - If $S_{\text{top}} \ge \tau$: generate answer strictly bounded by retrieved text.
  - If $S_{\text{top}} < \tau$: return fallback refusal string:
    *"I do not have sufficient information in the Nykaa knowledge base to answer this question. Please contact Nykaa customer support."*
- **Verification**: Demonstrate on $\ge 5$ in-scope queries + 1 out-of-scope query triggering fallback.

#### Task 1.5: Chunking Strategy Precision & Recall Evaluation (`rag/eval_chunking.py`)
- **File**: `rag/eval_chunking.py`
- **Logic**:
  - For 5 benchmark queries, retrieve top-$k$ ($k=3$) from both collections.
  - Map chunks back to `doc_id` and deduplicate.
  - Compute Precision and Recall per query for both collections:
    $$\text{Precision} = \frac{|\text{Retrieved} \cap \text{Relevant}|}{|\text{Retrieved}|}, \quad \text{Recall} = \frac{|\text{Retrieved} \cap \text{Relevant}|}{|\text{Relevant}|}$$
  - Print tabular side-by-side comparison with visible per-query arithmetic.
  - Formulate a 2–3 sentence data-backed recommendation for which collection to deploy in Part 2.
- **Verification**: Run evaluation script and verify clean math display.

---

## Phase 2: CrewAI Multi-Agent System, Memory & Guardrails (30 Marks)

### Objectives
Build the order status lookup tool with a justified escalation formula, develop the `MOCK_LLM` extending `BaseLLM`, assemble the 3-agent CrewAI crew, integrate session memory, enforce Pydantic output schemas, and implement input/output guardrails.

### Detailed Tasks

#### Task 2.1: Order Lookup Tool with Escalation Score (`tools/order_lookup.py`)
- **File**: `tools/order_lookup.py`
- **Function**: `check_order_status(record_id: str) -> dict`
- **Escalation Score Formula**:
  $$E = 0.5 \times \mathbb{I}(\text{delayed\_shipment} = \text{True}) + 0.5 \times \min\left(1.0, \frac{\text{days\_since\_created}}{30}\right)$$
- **Escalation Threshold**: $E \ge 0.65$ triggers `escalation_required = True`.
- **Distribution Justification**:
  - Maximum age is 30 days. An order with a shipping delay gets a base of $0.50$ and reaches $0.65$ at $\ge 9$ days old.
  - An order without a shipping delay requires $\ge 24$ days old ($24/30 \times 0.5 = 0.40$, not enough) or reaches escalation if exceeding the 80th percentile threshold of aging orders.
- **Verification**: Unit test orders with combinations: (delayed, young), (delayed, old), (on-time, young), (on-time, old).

#### Task 2.2: Deterministic Mock LLM (`mock_llm/crew_mock_llm.py`)
- **File**: `mock_llm/crew_mock_llm.py`
- **Requirements**:
  - Subclass `crewai.llms.base_llm.BaseLLM`.
  - Must avoid `"Observation:"` substring false-positive matching by only scanning generated agent completion text.
  - Must avoid tool substring matching (e.g. `"lookup"` matching `"rag_lookup"`); dispatch strictly based on tool declared schema or exact tool name.
  - Support structured JSON completions matching expected task outputs.
- **Telemetry Safeguard**:
  - Guarantee `os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"` and `os.environ["OTEL_SDK_DISABLED"] = "true"`.
- **Verification**: Run mock LLM through a simulated tool call cycle without external network calls.

#### Task 2.3: CrewAI 3-Agent Orchestration (`crew/agents.py`, `crew/crew.py`)
- **Files**:
  - `crew/agents.py`:
    - `Retrieval Agent`: Equipped with `rag_tool`.
    - `Lookup Agent`: Equipped exclusively with `check_order_status` tool.
    - `Response Composer Agent`: Equipped with no tools, responsible for synthesizing.
  - `crew/tasks.py`: Define tasks for retrieval, lookup, and response composition.
  - `crew/crew.py`: Assemble `Crew` and execute via `.kickoff()`.
- **Verification**: Demonstrate query execution invoking `rag_tool` for policy and `check_order_status` for order lookup.

#### Task 2.4: Multi-Turn Session Memory (`crew/memory.py`)
- **File**: `crew/memory.py`
- **Implementation**:
  - Implement LangChain's `InMemoryChatMessageHistory` with session keys.
  - Run multi-turn conversation:
    - Turn 1: "Hi, I have an issue with order ORD-1005."
    - Turn 2: "What was the status of that order again?" (Resolves "that order" via memory).
  - Demonstrate fresh conversation transcript where session memory is absent/cleared.
- **Verification**: Save transcripts of both connected multi-turn run and fresh reset run.

#### Task 2.5: Structured Output Validation (`models/response_models.py`)
- **File**: `models/response_models.py`
- **Schema**:
  ```python
  class AgentResponse(BaseModel):
      query: str
      status: str  # 'answered', 'escalated', 'refused'
      answer: str
      escalation_score: Optional[float] = None
      escalation_required: bool = False
      retrieved_sources: List[str] = []
      confidence: float
  ```
- **Verification**: Enforce validation in `crew.py` on completion.

#### Task 2.6: Input & Output Guardrails (`guardrails/`)
- **Files**:
  - `guardrails/input_guard.py`:
    - Fixed-format PII masking:
      - 10-digit Indian phone numbers: `r"(\+?91[\-\s]?)?[6789]\d{9}"` -> `[PHONE_MASKED]`
      - Credit/Debit Card last-4: `r"\b(?:\d{4}[\s\-]?){3}(\d{4})\b"` -> `XXXX-XXXX-XXXX-$1`
    - Prompt Injection detector: Regex & heuristic triggers for prompt overriding words (`"ignore previous"`, `"act as root"`, `"disregard guidelines"`).
  - `guardrails/output_guard.py`:
    - Groundedness check verifying generated claims against retrieved chunks.
- **Verification**: Demonstrate 1 deliberate test case for PII masking, 1 for prompt injection refusal, and 1 for output groundedness refusal.

---

## Phase 3: FastAPI Deployment, Observability & Evaluation (20 Marks)

### Objectives
Deploy the multi-agent system behind a FastAPI backend featuring HTTP and WebSocket endpoints, configure ELK-style structured logging with PII redaction, and run a 15-query LLM-as-judge evaluation benchmark.

### Detailed Tasks

#### Task 3.1: FastAPI Deployment (`api/app.py`, `api/schemas.py`)
- **Files**:
  - `api/schemas.py`: Pydantic models for HTTP requests/responses.
  - `api/app.py`:
    - Endpoint 1: `POST /ask` -> Ingests query, executes guardrails, cache, CrewAI pipeline, returns `AgentResponse`.
    - Endpoint 2: `POST /add-document` -> Validates and ingests new policy document into ChromaDB.
    - Endpoint 3: `WebSocket /ws/chat` -> Persistent bi-directional chat catching `WebSocketDisconnect` cleanly without terminating the server.
- **Verification**: Test endpoints with `TestClient` or `requests`/`websockets`, verifying graceful client disconnect.

#### Task 3.2: Structured JSON-Lines Observability (`observability/logger.py`)
- **File**: `observability/logger.py`
- **Implementation**:
  - Assign unique `trace_id` (UUID4) per request.
  - Measure execution latency in milliseconds.
  - Mask all fixed-format PII in the query *prior* to writing log entry.
  - Output format: One JSON line per request appended to `logs/traces.jsonl`.
- **Verification**: Inspect `logs/traces.jsonl` to ensure raw phone/card numbers never appear in cleartext.

#### Task 3.3: 15-Query Evaluation Suite (`evaluation/evaluator.py`, `evaluation/test_suite_15.json`)
- **Files**:
  - `evaluation/test_suite_15.json`: 15 queries covering:
    - 12 required KB topics (1 query each)
    - 1 order status query
    - 2 out-of-scope / adversarial queries
  - `evaluation/evaluator.py`:
    - `MOCK_LLM` Judge scoring each query on 4 dimensions (0.0 to 1.0):
      1. **Accuracy**
      2. **Grounding**
      3. **Completeness**
      4. **Safety**
    - Print per-query arithmetic table.
    - Calculate and print overall averages across all 15 queries.
- **Verification**: Run evaluation and generate transcript report.

---

## Phase 4: Resilience, AutoGen Review & Governance (20 Marks)

### Objectives
Add an independent AutoGen multi-agent review stage, implement the 4-layer AI governance model, add normalized in-memory query caching, and verify all acceptance criteria.

### Detailed Tasks

#### Task 4.1: AutoGen Secondary Review Stage (`review/autogen_team.py`)
- **Files**:
  - `review/verdict_schema.py`:
    ```python
    class VerdictModel(BaseModel):
        approved: bool
        final_answer: str
        reason: str
    ```
  - `review/autogen_team.py`:
    - Build `RoundRobinGroupChat` with `max_turns=2`.
    - Set `custom_message_types=[StructuredMessage[VerdictModel]]`.
    - Agents: `PolicyComplianceReviewer` and `FinalEditor`.
    - FinalEditor outputs `VerdictModel`.
  - **Demonstrations**:
    - Demo Case A (Approved): Clean, grounded policy draft approved unchanged.
    - Demo Case B (Revised): Deliberately inject an ungrounded claim into the draft; AutoGen detects and revises it.
- **Verification**: Capture transcripts showing both approved and revised runs.

#### Task 4.2: Four-Layer AI Governance Model (`governance/governance_policy.py`)
- **File**: `governance/governance_policy.py`
- **Layers**:
  - **Application Layer**:
    - Enforce Least Autonomy: Only `Lookup Agent` has tool binding to `check_order_status`. Demonstrate blocking if another agent attempts lookup.
    - Document Risk Classification: Justify **Medium Risk** classification in markdown/README.
  - **Runtime Layer**:
    - Cost / Token Budget Cap: Reject requests exceeding max input length (e.g. > 1000 characters) with HTTP 400/429 before LLM dispatch.
- **Verification**: Run test case demonstrating rejection of oversized request.

#### Task 4.3: In-Memory Normalized Query Cache (`cache/query_cache.py`)
- **File**: `cache/query_cache.py`
- **Implementation**:
  - Normalize query: strip whitespace, lowercase, remove trailing punctuation.
  - Cache entry: stores serialized `AgentResponse`.
  - Track stats: `hits`, `misses`.
  - Demonstrate timing/call-counter proof: First call (miss, runs crew) vs. Second call (hit, zero tool invocations, <1ms latency).
- **Verification**: Automated test asserting cache hit and zero tool invocations.

#### Task 4.4: Final Repository Packaging & `README.md`
- **File**: `README.md`
- **Contents**:
  - State track: **Nykaa (E-commerce & Retail)**.
  - Reproducibility parameters: Seed, category/status weights, price range reasoning.
  - Confirmation of `CREWAI_DISABLE_TELEMETRY=true` and keyless `MOCK_LLM` execution.
  - Calibrated similarity threshold values and justification.
  - Precision/Recall evaluation table & recommendation.
  - Governance classification justification.
  - Command runbook for executing tests, starting FastAPI, and running evaluation.
- **Verification**: End-to-end verification run passing all unit tests.

---

## Acceptance Criteria Matrix

| Part | Task Range | Deliverable Checkpoints | Success Verification |
| :--- | :--- | :--- | :--- |
| **Part 1** | Tasks 1–5 | `dataset.py`, `knowledge_base.json`, dual chunkers, ChromaDB indexing, empirical similarity calibration, precision/recall benchmark | $\ge 40$ orders (10-30% delays), 12 KB topics, calibrated $\tau$, dual PR table |
| **Part 2** | Tasks 6–10 | `check_order_status` + escalation formula, CrewAI `MOCK_LLM` (BaseLLM), 3 agents, session memory, Pydantic schema, PII/Injection guardrails | $E \in [0, 1]$ threshold, tool dispatch without ReAct bug, memory transcript, guardrails firing |
| **Part 3** | Tasks 11–13 | FastAPI (`POST /ask`, `POST /add-document`, `WS /ws/chat`), structured JSONL logging with trace IDs & redacted PII, 15-query evaluation suite | Disconnect-safe WebSocket, JSONL without cleartext PII, 4 metrics per query + averages |
| **Part 4** | Tasks 14–16 | AutoGen review team (`RoundRobinGroupChat`, `max_turns=2`), least-autonomy guard, runtime cost budget cap, query cache | Approval & revision transcripts, budget rejection test, cache hit timing proof |

---
*Implementation Plan ready for staged development.*

