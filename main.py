"""Nykaa Domain Support Agent — Main Application Entry Point.

Run modes:
  1. python main.py                 -> Interactive terminal chat with Nykaa Support Agent
  2. python main.py --demo          -> Runs 3 representative demo queries
  3. python main.py --server        -> Starts FastAPI web server + Swagger UI
  4. python main.py --test          -> Runs the complete automated test suite
"""

import sys
import os
import argparse

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from crew.nykaa_crew import run_nykaa_agent


def run_interactive_mode() -> None:
    """Runs an interactive multi-turn terminal chat session."""
    session_id = "terminal-session-01"
    print("=" * 70)
    print("   NYKAA DOMAIN SUPPORT AGENT — INTERACTIVE CONSOLE")
    print("   Type your question, or type 'exit' or 'quit' to end.")
    print("=" * 70)
    print("\nSample queries to try:")
    print("  • 'What is the return window for makeup products?'")
    print("  • 'How long does a COD refund take?'")
    print("  • 'What is the status of my order ORD-1003?'")
    print("  • 'My phone number is 9876543210, check order ORD-1002'\n")

    while True:
        try:
            user_input = input("You > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("\nThank you for using Nykaa Domain Support. Goodbye!")
                break

            print("\nProcessing with Nykaa Crew...")
            response = run_nykaa_agent(user_input, session_id=session_id)

            print("-" * 70)
            print(f"Status      : {response.status.upper()}")
            print(f"Answer      : {response.answer}")
            if response.retrieved_sources:
                print(f"Sources     : {', '.join(response.retrieved_sources)}")
            if response.escalation_score is not None:
                print(f"Escalation  : Score={response.escalation_score:.2f} | Required={response.escalation_required}")
            print(f"Confidence  : {response.confidence * 100:.0f}%")
            print("-" * 70 + "\n")

        except (KeyboardInterrupt, EOFError):
            print("\n\nSession terminated. Goodbye!")
            break


def run_demo_mode() -> None:
    """Executes representative demonstration queries."""
    demos = [
        ("1. Policy Inquiry (RAG Retrieval)", "What is the return window for makeup items?"),
        ("2. Order Lookup & Escalation Score", "What is the current delivery status for order ORD-1003?"),
        ("3. Guardrail PII Scrubbing", "My phone is 9876543210 and card is 4532-1234-5678-9012, check ORD-1002")
    ]

    print("=" * 70)
    print("   NYKAA DOMAIN SUPPORT AGENT — DEMO RUNNER")
    print("=" * 70)

    for title, query in demos:
        print(f"\n>>> Running Demo: {title}")
        print(f"Query: \"{query}\"")
        resp = run_nykaa_agent(query)
        print(f"Status     : {resp.status}")
        print(f"Answer     : {resp.answer}")
        print(f"Sources    : {resp.retrieved_sources}")
        if resp.escalation_score is not None:
            print(f"Escalation : Score={resp.escalation_score} | Required={resp.escalation_required}")
        print(f"Confidence : {resp.confidence}")


def start_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Starts the FastAPI server with uvicorn."""
    import uvicorn
    print(f"\nStarting Nykaa Support API at http://{host}:{port}")
    print(f"Swagger Documentation available at: http://{host}:{port}/docs")
    uvicorn.run("api.app:app", host=host, port=port, reload=False)


def run_tests() -> None:
    """Executes the complete pytest test suite."""
    import pytest
    print("\nRunning full test suite via pytest...")
    retcode = pytest.main(["-v"])
    sys.exit(retcode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Nykaa Domain Support Agent CLI")
    parser.add_argument("--demo", action="store_true", help="Run automated demonstration queries")
    parser.add_argument("--server", action="store_true", help="Start FastAPI REST & WebSocket server")
    parser.add_argument("--test", action="store_true", help="Run the full test suite")
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")

    args = parser.parse_args()

    if args.demo:
        run_demo_mode()
    elif args.server:
        start_server(host=args.host, port=args.port)
    elif args.test:
        run_tests()
    else:
        run_interactive_mode()


if __name__ == "__main__":
    main()

