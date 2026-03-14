import os
import logging
from openai import OpenAI
from dotenv import load_dotenv

from app.schemas.response import PredictionStatus

load_dotenv()
logger = logging.getLogger(__name__)

_client = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        token = os.getenv("GITHUB_TOKEN")
        if not token:
            raise RuntimeError(
                "Token for the agent report LLM not found. Please setit in .env "
                "with a valid GitHub Personal Access Token."
            )
        _client = OpenAI(
            base_url = "https://models.github.ai/inference",
            api_key  = token,
        )
    return _client


SYSTEM_PROMPT = """You are an expert agricultural plant pathologist specializing in tomato diseases.

When given a detected disease, you generate a clear, practical diagnostic report for a farmer.
Your report must always follow this EXACT format with these two sections:

SUMMARY:
<2-3 sentences describing what the disease is, how it looks, and how it spreads>

TREATMENT:
<numbered list of 3-5 concrete, actionable treatment steps the farmer can take immediately>

Rules:
- Use simple language a farmer can understand
- Be specific about products and methods
- Always start treatment steps with a number (1. 2. 3.)
- Never add extra sections or commentary outside the two sections above
- If the disease is Tomato_Healthy, confirm good health and give basic maintenance tips
"""


def _parse_response(text: str) -> tuple[str, str]:
    """Parses the LLM response into (summary, treatment) tuple."""
    try:
        if "SUMMARY:" in text and "TREATMENT:" in text:
            parts     = text.split("TREATMENT:", 1)
            summary   = parts[0].replace("SUMMARY:", "").strip()
            treatment = parts[1].strip()
        else:
            summary   = text.strip()
            treatment = "Please consult an agricultural specialist for treatment advice."
    except Exception:
        summary   = text.strip()
        treatment = "Please consult an agricultural specialist for treatment advice."

    return summary, treatment


def generate_report(
    predicted_class: str | None,
    status: PredictionStatus,
) -> tuple[str | None, str | None]:
    if status == PredictionStatus.FAILED or predicted_class is None:
        return None, None

    disease_display = predicted_class.replace("_", " ")
    confidence_note = (
        "The model is moderately confident in this diagnosis."
        if status == PredictionStatus.UNCERTAIN
        else "The model is highly confident in this diagnosis."
    )

    user_message = (
        f"Disease detected: {disease_display}\n"
        f"Confidence level: {status.value}\n"
        f"{confidence_note}\n\n"
        f"Generate a diagnostic report for this tomato plant disease."
    )

    try:
        client   = _get_client()
        response = client.chat.completions.create(
            model       = "openai/gpt-4o-mini",
            messages    = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            temperature = 0.3,
            max_tokens  = 400,
        )
        raw_text = response.choices[0].message.content or ""
        logger.info(f"Agent report generated for {predicted_class} ({status})")
        return _parse_response(raw_text)

    except Exception as e:
        logger.error(f"Agent report failed: {e}. Falling back.")
        return (
            f"{disease_display} was detected on the tomato leaf.",
            "Report generation failed. Please consult an agricultural specialist.",
        )