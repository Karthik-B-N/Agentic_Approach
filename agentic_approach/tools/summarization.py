from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import json
import os
from groq import Groq
from langchain_core.tools import Tool
from agentic_approach.config import CONFIG

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))


@dataclass
class ConversationEntities:
    possible_task_title: Optional[str]
    assignee: Optional[str]
    deadline: Optional[str]
    priority: Optional[str]
    jira_key: Optional[str]

def _load_room_messages(room_id: str, last_n: int) -> str:
    """
    Read the last N messages from the room log file written by websocket.py.
    """
    safe_room_id = room_id.replace(":", "_")
    room_file = os.path.join(PROJECT_ROOT, "rooms", f"{safe_room_id}.txt")
    if not os.path.exists(room_file):
        return ""

    with open(room_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if last_n <= 0:
        return ""

    return "".join(lines[-last_n:])


_groq_client: Groq | None = None

def _get_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _groq_client

def summarize_conversation(room_id: str, last_n: int | None = None) -> Dict[str, Any]:
    """
    Summarize the last N messages for a given room and extract task entities.

    Returns:
        {
          "summary": str,
          "entities": {
            "possible_task_title": str | None,
            "assignee": str | None,
            "deadline": str | None,
            "priority": str | None,
            "jira_key": str | None
          }
        }
    """
    n = last_n or CONFIG.last_n_messages
    transcript = _load_room_messages(room_id, n)

    if not transcript.strip():
        return {
            "summary": "",
            "entities": {
                "possible_task_title": None,
                "assignee": None,
                "deadline": None,
                "priority": None,
                "jira_key": None,
            },
        }

    system_prompt = """
        You are a summarizer for a Matrix room conversation.

        Given a slice of recent messages from a single room, you must:
        - Produce a concise summary of task-relevant content only.
        - Extract at most one likely task being discussed (if any).
        - Identify any explicit assignee, deadline, priority, or Jira key.

        Ambiguous or missing values MUST be returned as null.

        Reply in STRICT JSON:
        {
        "summary": "string",
        "entities": {
            "possible_task_title": "string or null",
            "assignee": "string or null",
            "deadline": "string or null",
            "priority": "string or null",
            "jira_key": "string or null"
        }
        }
"""

    user_prompt = f"Recent messages for room {room_id}:\n\n{transcript}"

    client = _get_client()
    completion = client.chat.completions.create(
        model=CONFIG.groq_model,
        temperature=0.0,
        max_completion_tokens=512,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    raw = completion.choices[0].message.content.strip()

    try:
        data = json.loads(raw)
    except Exception:
        return {
            "summary": "",
            "entities": {
                "possible_task_title": None,
                "assignee": None,
                "deadline": None,
                "priority": None,
                "jira_key": None,
            },
        }

    entities = data.get("entities") or {}
    return {
        "summary": data.get("summary") or "",
        "entities": {
            "possible_task_title": entities.get("possible_task_title"),
            "assignee": entities.get("assignee"),
            "deadline": entities.get("deadline"),
            "priority": entities.get("priority"),
            "jira_key": entities.get("jira_key"),
        },
    }


def _summarize_conversation_entry(room_id: str, last_n: int | None = None) -> Dict[str, Any]:
    """
    Wrapper used as the LangChain Tool entrypoint.
    """
    return summarize_conversation(room_id=room_id, last_n=last_n)


SUMMARIZE_CONVERSATION_TOOL: Tool = Tool(
    name="summarize_conversation",
    description=(
        "Summarize the last N messages of a Matrix room and extract task-related "
        "entities such as possible_task_title, assignee, deadline, priority, and jira_key."
    ),
    func=_summarize_conversation_entry,
)

