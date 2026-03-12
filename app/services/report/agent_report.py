"""
agent_report.py
STUB — Future LLM agent for dynamic diagnostic report generation.

This module exposes the same interface as static_report.py:
    generate_report(predicted_class, status) -> (summary, treatment)

When you're ready to implement the agent, replace the body of generate_report()
below. The routes and services that call it need zero changes.

Possible implementation approaches:
    - Claude API (claude-sonnet) with a structured prompt
    - RAG over a plant disease knowledge base
    - Tool-calling agent that queries external agronomic databases
"""
from app.schemas.response import PredictionStatus


def generate_report(
    predicted_class: str | None,
    status: PredictionStatus,
) -> tuple[str | None, str | None]:
    """
    Placeholder — falls back to static_report until the agent is implemented.
    Swap this body out when building the agent.
    """
    # ── Future agent implementation goes here ─────────────────────────────────
    # Example sketch:
    #
    # response = anthropic_client.messages.create(
    #     model="claude-sonnet-4-20250514",
    #     system=AGRONOMIST_SYSTEM_PROMPT,
    #     messages=[{
    #         "role": "user",
    #         "content": f"Disease detected: {predicted_class}. Generate a diagnostic report."
    #     }]
    # )
    # return parse_agent_response(response)
    # ──────────────────────────────────────────────────────────────────────────

    raise NotImplementedError(
        "agent_report.generate_report() is not yet implemented. "
        "Use static_report.generate_report() until the agent is built."
    )