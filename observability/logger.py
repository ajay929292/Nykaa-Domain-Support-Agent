"""Structured JSON-Lines Logger with PII Redaction for Nykaa Support Agent (Task 12).

Logs every request as an atomic JSON line with:
- trace_id (UUID4)
- ISO8601 timestamp
- endpoint
- latency_ms
- sanitized query text (fixed-format PII strictly redacted BEFORE disk write)
- execution status
"""

import os
import sys
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from guardrails.input_guard import mask_pii

LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
TRACES_FILE = os.path.join(LOGS_DIR, "traces.jsonl")


def get_traces_file_path() -> str:
    """Ensures logs directory exists and returns traces.jsonl path."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    return TRACES_FILE


def log_request_trace(
    endpoint: str,
    raw_query: str,
    latency_ms: float,
    status: str,
    status_code: int = 200,
    trace_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Writes a structured, PII-sanitized JSON-Lines trace entry to disk.

    Guarantees:
      Raw phone numbers and payment-card details are intercepted and masked
      PRIOR to disk write, ensuring no cleartext PII ever reaches the log file.
    """
    traces_path = get_traces_file_path()
    actual_trace_id = trace_id or str(uuid.uuid4())

    # Pre-log PII redaction (Task 12 requirement)
    sanitized_text, pii_was_masked = mask_pii(raw_query)

    entry = {
        "trace_id": actual_trace_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "latency_ms": round(latency_ms, 2),
        "sanitized_query": sanitized_text,
        "pii_masked": pii_was_masked,
        "status": status,
        "status_code": status_code
    }

    if extra:
        entry["metadata"] = extra

    # Atomic append of a single JSON line
    with open(traces_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    return entry


def read_recent_traces(limit: int = 20) -> list[Dict[str, Any]]:
    """Reads the most recent log entries from traces.jsonl."""
    traces_path = get_traces_file_path()
    if not os.path.exists(traces_path):
        return []

    entries = []
    with open(traces_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    return entries[-limit:]

