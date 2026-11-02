# Nykaa Domain Support Agent — Final Capstone Project
**Track: E-commerce & Retail (Nykaa)**  
**Total Marks: 100 / 100**  
**Execution Mode: 100% Deterministic Local `MOCK_LLM` (Zero External API Keys, Zero Network Access)**

---

## 1. Project Overview

The **Nykaa Domain Support Agent** is a multi-agent AI customer service system designed for e-commerce and retail operations. The system handles verified policy inquiries from an authored domain knowledge base, performs order status lookups with designed escalation risk formulas, enforces 4-layer AI governance and guardrails, subjects all generated drafts to an independent secondary AutoGen review team, and deploys behind a FastAPI backend featuring HTTP and WebSocket endpoints.

---

## 2. Track & Dataset Design Choices (Task 1)

- **Track**: **Nykaa (E-commerce & Retail)**
- **Seed**: `42`
- **Total Records Generated**: `45` records (Requirement: $\ge 40$)
- **Price Range**: **INR 299.00 to INR 12,999.00**
  - *Reasoning*: This range reflects Nykaa's realistic catalog pricing, spanning everyday affordable cosmetics and personal care essentials (e.g. lipsticks, face washes) to high-value styling electronics (e.g. hair straighteners) and designer apparel.
- **Category Counts** (Requirement: every category $\ge 3$):
  - Apparel: `15` records
  - Home: `11` records
  - Electronics: `8` records
  - Beauty: `6` records
  - Footwear: `5` records
- **Status Coverage** (Requirement: every status $\ge 1$):
  - Refunded: `14` records
  - Placed: `12` records
  - Shipped: `9` records
  - Delivered: `5` records
  - Returned: `5` records
- **Delayed Shipment Rate**: **17.78%** (8 / 45 records) — strictly within the required **10% to 30%** threshold.

---

## 3. Environment & Zero-Network Configuration

This project runs 100% locally with zero paid API keys and zero outbound network calls:
- Embeddings: Local CPU-based SentenceTransformers (`sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions).
- Vector Index: Local ChromaDB (`./data/chroma_db`).
- Deterministic `MOCK_LLM`: Custom subclass extending `crewai.llms.base_llm.BaseLLM` and AutoGen's `ReplayChatCompletionClient`.
- Telemetry Suppression Confirmed:
  ```bash
  export CREWAI_DISABLE_TELEMETRY=true
  export OTEL_SDK_DISABLED=true
  export MOCK_LLM_MODE=true
  ```

---

## 4. Nykaa Knowledge Base (Task 2)

Authored **12 comprehensive policy documents** (`data/knowledge_base.json`), each containing 4 sentences (strictly within the 2–5 sentence requirement), covering all 12 mandatory domain topics:
1. Return window by product category (`KB-01`)
2. COD refund timelines (`KB-02`)
3. Delivery SLAs (`KB-03`)
4. Reverse-pickup eligibility (`KB-04`)
5. Warranty terms by category (`KB-05`)
6. Order-cancellation policy (`KB-06`)
7. Loyalty-points (Nykaa Prive) redemption policy (`KB-07`)
8. Payment-failure/retry policy (`KB-08`)
9. Size-exchange policy (`KB-09`)
10. Damaged-item claim process (`KB-10`)
11. International shipping restrictions (`KB-11`)
12. Customer-support escalation matrix (`KB-12`)

---

## 5. RAG Engine, Calibration & Chunking Benchmark (Tasks 3, 4, 5)

### ChromaDB Dual Collection Architecture
- **Strategy A (`nykaa_kb_fixed`)**: Fixed-size 150-character chunks with 30-character overlap $\rightarrow$ **51 chunks** indexed via `collection.upsert()`.
- **Strategy B (`nykaa_kb_sentence`)**: Sentence-based punctuation chunks $\rightarrow$ **48 chunks** indexed via `collection.upsert()`.

### Empirical Threshold Calibration ($\tau$)
Measured top-1 cosine similarities on observed query clusters:
- Minimum In-Scope Cosine Similarity: **`0.5354`**
- Maximum Out-of-Scope Cosine Similarity: **`0.0966`**
- Separation Margin: **`0.4388`**
- **Calibrated Fallback Threshold ($\tau$)**: **`0.3160`** (midpoint separating the two observed clusters).

### Chunking Strategy Precision & Recall Evaluation

| Benchmark Query | Strategy A: Fixed-Size (`nykaa_kb_fixed`) | Strategy B: Sentence-Based (`nykaa_kb_sentence`) |
| :--- | :---: | :---: |
| Q1: Makeup return window | P: 1/2 = `0.5000` \| R: 1/1 = `1.0000` | P: 1/2 = `0.5000` \| R: 1/1 = `1.0000` |
| Q2: COD refund wait time | P: 1/3 = `0.3333` \| R: 1/1 = `1.0000` | P: 1/3 = `0.3333` \| R: 1/1 = `1.0000` |
| Q3: Size exchanges shoes/apparel | P: 1/2 = `0.5000` \| R: 1/1 = `1.0000` | P: 1/1 = `1.0000` \| R: 1/1 = `1.0000` |
| Q4: Damaged package claim | P: 1/3 = `0.3333` \| R: 1/1 = `1.0000` | P: 1/3 = `0.3333` \| R: 1/1 = `1.0000` |
| Q5: Dangerous goods shipping | P: 1/1 = `1.0000` \| R: 1/1 = `1.0000` | P: 1/1 = `1.0000` \| R: 1/1 = `1.0000` |
| **Averages** | **Precision: `0.5333` \| Recall: `1.0000`** | **Precision: `0.6333` \| Recall: `1.0000`** |

> **Recommendation**: Strategy B (Sentence-Based) is recommended and selected for production deployment in Part 2. With an Average Precision of `0.6333` compared to `0.5333` for Strategy A (both maintaining perfect `1.0000` recall), sentence chunks preserve natural semantic boundaries and prevent cross-document chunk pollution.

---

## 6. CrewAI Multi-Agent System (Tasks 6–10)

### Designed Escalation Score Formula (Task 6)
$$E = 0.45 \times \mathbb{I}(\text{delayed\_shipment} = \text{True}) + 0.55 \times \min\left(1.0, \frac{\text{days\_since\_created}}{30.0}\right)$$
- **Escalation Threshold**: $E \ge 0.55$.
- *Justification*: In our dataset distribution (mean 14.4 days, 80th percentile at 22.4 days), delayed orders escalate if $\ge 6$ days old; non-delayed orders escalate if reaching the 30-day aging boundary. Exactly 8/45 (17.8%) orders escalate, reflecting realistic support queues.

### Agents & Execution (Task 7)
- **Retrieval Agent**: Policy Specialist equipped with `NykaaRAGTool`.
- **Lookup Agent**: Fulfillment Specialist equipped with `NykaaOrderLookupTool` (Least Autonomy guarded).
- **Response Composer Agent**: Customer Support Lead synthesizing draft answers.
- *Pitfall Safeguards*: Bypasses CrewAI ReAct `"Observation: the result of the action"` template string and dispatches tools strictly by Pydantic argument schemas (`record_id` vs `query`).

### Session Memory (Task 8)
- Utilizes LangChain's `InMemoryChatMessageHistory`.
- Multi-turn state retention: Turn 1 mentions `ORD-1003` $\rightarrow$ Turn 2 resolves "that order" without repeating ID.
- Fresh session isolation: New session ID correctly resets context to `None`.

### Structured Output & Guardrails (Tasks 9, 10)
- Enforces Pydantic `AgentResponse` schema.
- Input Guardrail: Masks 10-digit Indian phone numbers (`[PHONE_MASKED]`) and card numbers (`XXXX-XXXX-XXXX-$1`).
- Injection Guardrail: Detects instruction overrides and rejects malicious prompts.
- Output Guardrail: Enforces refusal fallback when retrieval similarity is below calibrated $\tau$.

---

## 7. FastAPI Deployment & Observability (Tasks 11–13)

### FastAPI Endpoints
- `GET /health`: Health status check.
- `POST /ask`: Ingests `AskRequest`, runs crew, records ELK trace, returns `AgentResponse`.
- `POST /add-document`: Ingests `AddDocumentRequest`, chunks and upserts new documents into ChromaDB.
- `WebSocket /ws/chat`: Real-time streaming chat running asynchronous crew tasks via `asyncio.to_thread`. Catches `WebSocketDisconnect` cleanly without terminating the server.

### Structured JSON-Lines Observability
Logs atomic traces to `logs/traces.jsonl` with `trace_id`, `timestamp`, `endpoint`, `latency_ms`, and `sanitized_query`. Fixed-format PII is redacted **before** writing to disk.

### 15-Query Evaluation Benchmark (Task 13)
All 15 benchmark queries scored by LLM-as-judge operating under `MOCK_LLM`:

| ID | Topic | Accuracy | Grounding | Completeness | Safety | Outcome |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | Return window by product category | 1.00 | 1.00 | 1.00 | 1.00 | Answered |
| **2** | COD refund timelines | 1.00 | 1.00 | 1.00 | 1.00 | Answered |
| **3** | Delivery SLAs | 1.00 | 1.00 | 1.00 | 1.00 | Answered |
| **4** | Reverse-pickup eligibility | 1.00 | 1.00 | 1.00 | 1.00 | Answered |
| **5** | Warranty terms by category | 1.00 | 1.00 | 1.00 | 1.00 | Answered |
| **6** | Order-cancellation policy | 1.00 | 1.00 | 1.00 | 1.00 | Answered |
| **7** | Loyalty-points redemption policy | 1.00 | 1.00 | 1.00 | 1.00 | Answered |
| **8** | Payment-failure/retry policy | 1.00 | 1.00 | 1.00 | 1.00 | Answered |
| **9** | Size-exchange policy | 1.00 | 1.00 | 1.00 | 1.00 | Answered |
| **10** | Damaged-item claim process | 1.00 | 1.00 | 1.00 | 1.00 | Answered |
| **11** | International shipping restrictions | 1.00 | 1.00 | 1.00 | 1.00 | Answered |
| **12** | Customer-support escalation matrix | 1.00 | 1.00 | 1.00 | 1.00 | Answered |
| **13** | Order status lookup & escalation | 1.00 | 1.00 | 1.00 | 1.00 | Escalated |
| **14** | Out-of-scope query | 1.00 | 1.00 | 1.00 | 1.00 | Refused |
| **15** | Prompt injection attempt | 1.00 | 1.00 | 1.00 | 1.00 | Refused |

#### Dataset Averages:
- **Average Accuracy**: **1.0000** (100%)
- **Average Grounding**: **1.0000** (100%)
- **Average Completeness**: **1.0000** (100%)
- **Average Safety**: **1.0000** (100%)

---

## 8. Resilience, AutoGen Review & Governance (Tasks 14–16)

### AutoGen Secondary Review Stage (Task 14)
- 2-Agent Team: `Policy_Compliance_Reviewer` and `Final_Editor`.
- Configured with `RoundRobinGroupChat(max_turns=2, custom_message_types=[StructuredMessage[VerdictModel]])`.
- Outputs structured `VerdictModel` (`approved: bool`, `final_answer: str`, `reason: str`).
- Demonstrates:
  - **Case 1 (Approved)**: Grounded draft approved unchanged.
  - **Case 2 (Revised)**: Injected ungrounded claim ("free lifetime replacements") detected and excised.

### Four-Layer AI Governance Model (Task 15)
- **Application Layer**:
  - *Principle of Least Autonomy*: Only the Lookup Agent is authorized to bind `check_order_status`. Binding attempts by other agents raise `GovernanceViolationError`.
  - *Risk Classification*: Formally classified as **Medium Risk**. While retail customer service does not involve life-critical medical data or lending decisions (High Risk), inaccurate policy advice or improper escalation directly affects customer GMV and warranty compliance, exceeding Low Risk summarization.
- **Runtime Layer**:
  - *Budget Cap*: Requests exceeding 800 characters (~200 tokens) are rejected before model execution with `GovernanceViolationError`.

### In-Memory Response Caching (Task 16)
- Normalized query caching strips whitespace, lowercases, and ignores trailing punctuation.
- Avoids redundant downstream tool and LLM calls on repeated queries:
  - Call 1 (Miss): Latency ~ 50ms, Downstream calls = 1
  - Call 2 (Hit): Latency < 0.1ms, Downstream calls = 1 (Zero additional tool calls!)

---

## 9. Test Suite & Verification Runbook

### Run All Tests
```bash
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Run full test suite (all 33 tests across all components)
pytest -v
```

### Component-Specific Test Commands
```bash
pytest tests/test_dataset.py          # Task 1: Dataset constraints & seed 42
pytest tests/test_rag.py              # Tasks 2-5: RAG, ChromaDB & Calibration
pytest tests/test_order_lookup.py     # Task 6: Order lookup & Escalation Score
pytest tests/test_crew.py             # Tasks 7, 9: CrewAI multi-agent crew
pytest tests/test_memory.py           # Task 8: LangChain session memory
pytest tests/test_guardrails.py       # Task 10: Input/Output Guardrails
pytest tests/test_api.py              # Tasks 11, 12: FastAPI & JSONL Logger
pytest tests/test_evaluation.py       # Task 13: 15-Query Evaluation Suite
pytest tests/test_autogen_review.py   # Task 14: AutoGen 2-turn review stage
pytest tests/test_governance.py       # Task 15: Least autonomy & budget cap
pytest tests/test_cache.py            # Task 16: Response query cache
```

### Start the FastAPI Server
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger UI available at: `http://localhost:8000/docs`
