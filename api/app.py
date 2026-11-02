"""FastAPI Backend Application for Nykaa Domain Support Agent (Tasks 11 & 12).

Endpoints:
1. POST /ask: Primary customer query endpoint with multi-agent orchestration.
2. POST /add-document: Dynamic policy ingestion into ChromaDB vector collections.
3. WebSocket /ws/chat: Real-time multi-turn chat handling WebSocketDisconnect cleanly.

Includes:
- ELK-style structured JSON-Lines logging with Trace IDs.
- Fixed-format PII redaction prior to logging and LLM ingestion.
"""

import os
import sys
import time
import uuid
import json
import asyncio
from typing import Dict, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import AskRequest, AddDocumentRequest, AddDocumentResponse
from models.response_models import AgentResponse
from crew.nykaa_crew import run_nykaa_agent
from observability.logger import log_request_trace
from rag.chunker import chunk_by_sentences, chunk_fixed_size_overlap
from rag.embedder import embed_texts
from rag.indexer import get_chroma_client, COLLECTION_FIXED, COLLECTION_SENTENCE

app = FastAPI(
    title="Nykaa Domain Support Agent API",
    description="Production-minded multi-agent customer support backend for Nykaa e-commerce.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/health", tags=["Health"])
def health_check() -> Dict[str, str]:
    """Basic health check endpoint."""
    return {"status": "healthy", "service": "Nykaa Domain Support Agent"}


@app.post("/ask", response_model=AgentResponse, tags=["Agent Support"])
def ask_support_agent(payload: AskRequest) -> AgentResponse:
    """Submits a customer query to the multi-agent support crew.

    Applies input guardrails, queries RAG/order tools, validates against Pydantic schema,
    and records an ELK-style structured trace with redacted PII.
    """
    start_time = time.perf_counter()
    trace_id = str(uuid.uuid4())

    try:
        response = run_nykaa_agent(
            raw_query=payload.query,
            session_id=payload.session_id or "default_session"
        )
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Structured JSON-Lines logging with pre-log PII redaction
        log_request_trace(
            endpoint="/ask",
            raw_query=payload.query,
            latency_ms=latency_ms,
            status=response.status,
            status_code=200,
            trace_id=trace_id,
            extra={
                "session_id": payload.session_id,
                "sources": response.retrieved_sources,
                "escalation_required": response.escalation_required
            }
        )

        return response

    except Exception as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        log_request_trace(
            endpoint="/ask",
            raw_query=payload.query,
            latency_ms=latency_ms,
            status="error",
            status_code=500,
            trace_id=trace_id,
            extra={"error": str(exc)}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent processing error: {str(exc)}"
        )


@app.post("/add-document", response_model=AddDocumentResponse, tags=["Knowledge Base"])
def add_policy_document(payload: AddDocumentRequest) -> AddDocumentResponse:
    """Dynamically chunks, embeds, and indexes a new policy document into ChromaDB."""
    start_time = time.perf_counter()
    trace_id = str(uuid.uuid4())

    try:
        client = get_chroma_client()
        doc_id = f"KB-DYN-{uuid.uuid4().hex[:6].upper()}"

        # 1. Chunk by sentences
        sentence_chunks = chunk_by_sentences(payload.content)
        fixed_chunks = chunk_fixed_size_overlap(payload.content)

        if not sentence_chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Provided content could not be parsed into sentences."
            )

        # 2. Embed
        sentence_embeddings = embed_texts(sentence_chunks)
        fixed_embeddings = embed_texts(fixed_chunks)

        # 3. Upsert into Sentence collection
        col_sentence = client.get_or_create_collection(name=COLLECTION_SENTENCE)
        col_sentence.upsert(
            ids=[f"{doc_id}_sent_{i}" for i in range(len(sentence_chunks))],
            documents=sentence_chunks,
            embeddings=sentence_embeddings,
            metadatas=[{
                "doc_id": doc_id,
                "topic": payload.topic,
                "title": payload.title,
                "strategy": "sentence"
            } for _ in sentence_chunks]
        )

        # 4. Upsert into Fixed collection
        col_fixed = client.get_or_create_collection(name=COLLECTION_FIXED)
        col_fixed.upsert(
            ids=[f"{doc_id}_fixed_{i}" for i in range(len(fixed_chunks))],
            documents=fixed_chunks,
            embeddings=fixed_embeddings,
            metadatas=[{
                "doc_id": doc_id,
                "topic": payload.topic,
                "title": payload.title,
                "strategy": "fixed_size"
            } for _ in fixed_chunks]
        )

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        log_request_trace(
            endpoint="/add-document",
            raw_query=f"Add doc: {payload.title}",
            latency_ms=latency_ms,
            status="indexed",
            status_code=200,
            trace_id=trace_id,
            extra={"doc_id": doc_id, "chunks_count": len(sentence_chunks)}
        )

        return AddDocumentResponse(
            success=True,
            doc_id=doc_id,
            topic=payload.topic,
            chunks_indexed=len(sentence_chunks),
            message=f"Document '{payload.title}' successfully indexed into ChromaDB collections."
        )

    except HTTPException:
        raise
    except Exception as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        log_request_trace(
            endpoint="/add-document",
            raw_query=f"Add doc: {payload.title}",
            latency_ms=latency_ms,
            status="error",
            status_code=500,
            trace_id=trace_id,
            extra={"error": str(exc)}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document indexing failed: {str(exc)}"
        )


@app.websocket("/ws/chat")
async def websocket_chat_endpoint(websocket: WebSocket):
    """Real-time multi-turn chat over WebSocket.

    Gracefully handles client disconnects (WebSocketDisconnect) without crashing the server.
    """
    await websocket.accept()
    session_id = f"ws_session_{uuid.uuid4().hex[:8]}"

    try:
        while True:
            # Receive incoming query string or JSON payload from client
            raw_message = await websocket.receive_text()
            if not raw_message.strip():
                continue

            try:
                data = json.loads(raw_message)
                query_text = data.get("query", raw_message)
                client_session = data.get("session_id", session_id)
            except Exception:
                query_text = raw_message
                client_session = session_id

            # Execute multi-agent support pipeline in background thread
            response = await asyncio.to_thread(
                run_nykaa_agent,
                raw_query=query_text,
                session_id=client_session
            )

            # Send back serialized AgentResponse JSON
            await websocket.send_text(response.model_dump_json())

    except WebSocketDisconnect:
        # Graceful disconnect handling as required by Task 11
        print(f"[WebSocket] Client in session '{session_id}' disconnected cleanly.")
    except Exception as exc:
        print(f"[WebSocket Error] Unexpected error in session '{session_id}': {exc}")
        try:
            await websocket.close()
        except Exception:
            pass

