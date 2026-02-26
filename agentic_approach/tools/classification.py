from __future__ import annotations

from typing import Literal, TypedDict

from groq import Groq
import os
from langchain_core.tools import Tool

from agentic_approach.config import CONFIG
from agentic_approach.tools.summarization import _load_room_messages


class ClassificationResult(TypedDict):
    is_task_related: bool
    intent: Literal["create_task", "update_task", "none"]
    confidence: float


_groq_client: Groq | None = None


def _get_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _groq_client


def classify_message(message: str, room_id: str | None = None) -> ClassificationResult:
    """
    Classify a single message as task-related or not.

    This function is deliberately synchronous and stateless so it is easy
    to unit test. It uses a deterministic Groq chat completion and returns
    a simple TypedDict with score and intent.
    """
    if not message.strip():
        return {
            "is_task_related": False,
            "intent": "none",
            "confidence": 0.0,
        }

    system_prompt = """
You are a HIGH-PRECISION classifier that decides whether a chat message
and its recent context are about Jira-style task management.

Answer in STRICT JSON:
{
  "is_task_related": true | false,
  "intent": "create_task" | "update_task" | "none",
  "confidence": 0.0-1.0
}

Only mark a message as task-related when it CLEARLY:
- Asks to create a new work item, or
- Requests a change to an existing Jira issue, or
- Explicitly assigns work with a concrete action and scope.

Do NOT treat vague chatter, greetings, or general discussion as task-related,
even if they mention projects or people.

If you are NOT clearly confident (>= 0.8) that it is task-related,
then set:
  "is_task_related": false,
  "intent": "none",
  "confidence": a value < 0.8.
"""

    context_block = ""
    if room_id:
        transcript = _load_room_messages(room_id, CONFIG.last_n_messages)
        if transcript.strip():
            context_block = f"\n\nRecent messages in this room:\n{transcript}"

    user_content = f"Message:\n{message}{context_block}"

    client = _get_client()
    completion = client.chat.completions.create(
        model=CONFIG.groq_model,
        temperature=0.0,
        max_completion_tokens=256,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )

    raw = completion.choices[0].message.content.strip()

    import json

    try:
        data = json.loads(raw)
    except Exception:
        # Fail closed: treat as non-task-related
        return {
            "is_task_related": False,
            "intent": "none",
            "confidence": 0.0,
        }

    is_task_related = bool(data.get("is_task_related"))
    intent = data.get("intent") or "none"
    confidence = float(data.get("confidence") or 0.0)

    # Enforce threshold semantics: below threshold → none
    if not is_task_related or confidence < CONFIG.classification_threshold:
        return {
            "is_task_related": False,
            "intent": "none",
            "confidence": confidence,
        }

    if intent not in ("create_task", "update_task"):
        intent = "none"

    return {
        "is_task_related": is_task_related,
        "intent": intent,  # type: ignore[return-value]
        "confidence": confidence,
    }


def _classify_message_entry(message: str) -> ClassificationResult:
    """
    Thin wrapper to expose classify_message as a LangChain Tool entrypoint.
    """
    return classify_message(message)


CLASSIFY_MESSAGE_TOOL: Tool = Tool(
    name="classify_message",
    description=(
        "Classify a single chat message as task-related or not, and determine "
        "whether it implies creating a new Jira task or updating an existing one. "
        "Returns JSON with is_task_related, intent, and confidence."
    ),
    func=_classify_message_entry,
)

