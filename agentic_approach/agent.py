from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal

import os

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from agentic_approach.config import CONFIG
from agentic_approach.prompts import TASK_DETECTION_SYSTEM_PROMPT
from agentic_approach.tools.classification import (
    CLASSIFY_MESSAGE_TOOL,
    classify_message,
)
from agentic_approach.tools.summarization import (
    SUMMARIZE_CONVERSATION_TOOL,
    summarize_conversation,
)
from agentic_approach.tools.jira_tools import (
    CREATE_JIRA_TASK_TOOL,
    UPDATE_JIRA_TASK_TOOL,
    create_jira_task,
    update_jira_task,
)
from helpers import safe_parse_deadline
from matrix_sender import send_message_reply


# All tools the agent can conceptually use (for external LangChain composition).
AGENT_TOOLS = [
    CLASSIFY_MESSAGE_TOOL,
    SUMMARIZE_CONVERSATION_TOOL,
    CREATE_JIRA_TASK_TOOL,
    UPDATE_JIRA_TASK_TOOL,
]


@dataclass
class AgentDecision:
    action: str
    # For create
    project_key: Optional[str]
    summary: Optional[str]
    description: Optional[str]
    issue_type: Optional[str]
    assignee: Optional[str]
    priority: Optional[str]
    due_date: Optional[str]
    # For update
    jira_key: Optional[str]
    # For clarification
    missing_fields: List[str]
    clarification_message: Optional[str]


@dataclass
class PendingTaskState:
    """
    In-memory per-room state used to accumulate fields across clarification
    turns until a Jira create/update can be executed.
    """

    mode: Literal["create", "update"]
    project_key: Optional[str] = None
    summary: Optional[str] = None
    description: Optional[str] = None
    issue_type: Optional[str] = None
    assignee: Optional[str] = None
    priority: Optional[str] = None
    due_date: Optional[str] = None
    jira_key: Optional[str] = None


# Simple in-process memory keyed by room_id.
_pending_tasks: Dict[str, PendingTaskState] = {}


def _get_pending(room_id: str) -> Optional[PendingTaskState]:
    return _pending_tasks.get(room_id)


def _set_pending(room_id: str, state: PendingTaskState) -> None:
    _pending_tasks[room_id] = state


def _clear_pending(room_id: str) -> None:
    _pending_tasks.pop(room_id, None)


def _parse_decision(raw_json: str) -> AgentDecision:
    import json

    try:
        data = json.loads(raw_json)
    except Exception:
        return AgentDecision(
            action="none",
            project_key=None,
            summary=None,
            description=None,
            issue_type=None,
            assignee=None,
            priority=None,
            due_date=None,
            jira_key=None,
            missing_fields=[],
            clarification_message=None,
        )

    missing = data.get("missing_fields") or []
    if not isinstance(missing, list):
        missing = []

    return AgentDecision(
        action=data.get("action") or "none",
        project_key=data.get("project_key"),
        summary=data.get("summary"),
        description=data.get("description"),
        issue_type=data.get("issue_type"),
        assignee=data.get("assignee"),
        priority=data.get("priority"),
        due_date=data.get("due_date"),
        jira_key=data.get("jira_key"),
        missing_fields=[str(m) for m in missing],
        clarification_message=data.get("clarification_message"),
    )


def _merge_with_pending(room_id: str, decision: AgentDecision) -> AgentDecision:
    """
    Merge the latest LLM decision with any pending per-room state.

    This lets the agent remember fields that were already supplied in
    earlier turns, so it doesn't re-ask for them.
    """
    pending = _get_pending(room_id)
    if not pending:
        # If we are starting a new create/update, seed pending now.
        if decision.action == "create_task":
            _set_pending(
                room_id,
                PendingTaskState(
                    mode="create",
                    project_key=decision.project_key,
                    summary=decision.summary,
                    description=decision.description,
                    issue_type=decision.issue_type,
                    assignee=decision.assignee,
                    priority=decision.priority,
                    due_date=decision.due_date,
                ),
            )
        elif decision.action == "update_task":
            _set_pending(
                room_id,
                PendingTaskState(
                    mode="update",
                    jira_key=decision.jira_key,
                    summary=decision.summary,
                    description=decision.description,
                    assignee=decision.assignee,
                    priority=decision.priority,
                    due_date=decision.due_date,
                ),
            )
        return decision

    # If we have pending state, prefer the newest explicit values from
    # the decision, but fall back to what was previously stored.
    if decision.action == "create_task" and pending.mode == "create":
        merged = AgentDecision(
            action=decision.action,
            project_key=decision.project_key or pending.project_key,
            summary=decision.summary or pending.summary,
            description=decision.description or pending.description,
            issue_type=decision.issue_type or pending.issue_type,
            assignee=decision.assignee or pending.assignee,
            priority=decision.priority or pending.priority,
            due_date=decision.due_date or pending.due_date,
            jira_key=None,
            missing_fields=decision.missing_fields,
            clarification_message=decision.clarification_message,
        )
        _set_pending(
            room_id,
            PendingTaskState(
                mode="create",
                project_key=merged.project_key,
                summary=merged.summary,
                description=merged.description,
                issue_type=merged.issue_type,
                assignee=merged.assignee,
                priority=merged.priority,
                due_date=merged.due_date,
            ),
        )
        return merged

    if decision.action == "update_task" and pending.mode == "update":
        merged = AgentDecision(
            action=decision.action,
            project_key=None,
            summary=decision.summary or pending.summary,
            description=decision.description or pending.description,
            issue_type=None,
            assignee=decision.assignee or pending.assignee,
            priority=decision.priority or pending.priority,
            due_date=decision.due_date or pending.due_date,
            jira_key=decision.jira_key or pending.jira_key,
            missing_fields=decision.missing_fields,
            clarification_message=decision.clarification_message,
        )
        _set_pending(
            room_id,
            PendingTaskState(
                mode="update",
                jira_key=merged.jira_key,
                summary=merged.summary,
                description=merged.description,
                assignee=merged.assignee,
                priority=merged.priority,
                due_date=merged.due_date,
            ),
        )
        return merged

    # If modes don't match, reset to the latest decision.
    if decision.action == "create_task":
        _set_pending(
            room_id,
            PendingTaskState(
                mode="create",
                project_key=decision.project_key,
                summary=decision.summary,
                description=decision.description,
                issue_type=decision.issue_type,
                assignee=decision.assignee,
                priority=decision.priority,
                due_date=decision.due_date,
            ),
        )
    elif decision.action == "update_task":
        _set_pending(
            room_id,
            PendingTaskState(
                mode="update",
                jira_key=decision.jira_key,
                summary=decision.summary,
                description=decision.description,
                assignee=decision.assignee,
                priority=decision.priority,
                due_date=decision.due_date,
            ),
        )

    return decision


async def handle_message(
    room_id: str,
    event_id: str,
    sender: str,
    message_text: str,
) -> None:
    """
    Core agent entrypoint, to be called from the Matrix listener.

    This function is responsible for:
    - classifying the new message
    - short-circuiting on non-task messages
    - summarizing recent room context
    - asking Groq for a strict decision
    - calling Jira tools or replying for missing fields
    """
    if not CONFIG.enabled:
        return

    # --- Step 1: Classification (hard gate) ---
    cls = classify_message(message_text)
    if not cls["is_task_related"] or cls["intent"] == "none":
        # STRICT: never respond, never summarize, never call Jira.
        return

    # --- Step 2: Fetch conversation context ---
    summary_result = summarize_conversation(
        room_id=room_id,
        last_n=CONFIG.last_n_messages,
    )

    structured_summary = summary_result.get("summary") or ""
    entities = summary_result.get("entities") or {}

    # --- Step 3: Decide action via LangChain ChatGroq ---
    system_prompt = TASK_DETECTION_SYSTEM_PROMPT.strip()

    payload = {
        "current_message": message_text,
        "classification_intent": cls["intent"],
        "classification_confidence": cls["confidence"],
        "conversation_summary": structured_summary,
        "entities": entities,
    }

    user_prompt = f"""
        You must output STRICT JSON in this schema:
        {{
        "action": "create_task" | "update_task" | "request_missing_information" | "none",
        "project_key": "string or null",
        "summary": "string or null",
        "description": "string or null",
        "issue_type": "string or null",
        "assignee": "string or null",
        "priority": "string or null",
        "due_date": "string or null",
        "jira_key": "string or null",
        "missing_fields": ["field_name", ...],
        "clarification_message": "string or null"
        }}

        Rules:
        - If the message is unrelated to task management, set action = "none".
        - For create_task, required fields are: project_key, summary, description, issue_type.
        - For update_task, required field is: jira_key.
        - For request_missing_information, you MUST list only the missing fields in missing_fields
        and provide a single concise clarification_message to send to the room.
        - Never invent Jira keys or project keys; use only what appears in the message or summary.

        Input:
        {payload}
"""
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name=CONFIG.groq_model,
        temperature=0.0,
    )
    response = await llm.ainvoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )

    decision_raw = (response.content or "").strip()
    decision = _parse_decision(decision_raw)
    decision = _merge_with_pending(room_id, decision)

    # Safety: if LLM says none, do nothing.
    if decision.action == "none":
        return

    # --- Step 4: Execute action ---
    if decision.action == "create_task":
        await _handle_create_task(room_id, event_id, decision)
    elif decision.action == "update_task":
        await _handle_update_task(room_id, event_id, decision)
    elif decision.action == "request_missing_information":
        await _handle_request_missing(room_id, event_id, decision)


async def _handle_create_task(room_id: str, event_id: str, d: AgentDecision) -> None:
    # Merge with any existing pending state (defensive, in case this is called directly).
    pending = _get_pending(room_id)
    project_key = d.project_key or (pending.project_key if pending and pending.mode == "create" else None)  # type: ignore[union-attr]
    project_key = project_key or CONFIG.default_project_key
    summary = d.summary or (pending.summary if pending and pending.mode == "create" else None)  # type: ignore[union-attr]
    description = d.description or (pending.description if pending and pending.mode == "create" else None)  # type: ignore[union-attr]
    issue_type = d.issue_type or (pending.issue_type if pending and pending.mode == "create" else None)  # type: ignore[union-attr]
    assignee = d.assignee or (pending.assignee if pending and pending.mode == "create" else None)  # type: ignore[union-attr]
    priority = d.priority or (pending.priority if pending and pending.mode == "create" else None)  # type: ignore[union-attr]
    due_date_raw = d.due_date or (pending.due_date if pending and pending.mode == "create" else None)  # type: ignore[union-attr]
    due_date = safe_parse_deadline(due_date_raw) if due_date_raw else None

    _set_pending(
        room_id,
        PendingTaskState(
            mode="create",
            project_key=project_key,
            summary=summary,
            description=description,
            issue_type=issue_type,
            assignee=assignee,
            priority=priority,
            due_date=due_date,
        ),
    )

    # Check required fields ourselves; never trust model blindly.
    required_missing: list[str] = []
    if not project_key:
        required_missing.append("project_key")
    if not summary:
        required_missing.append("summary")
    if not description:
        required_missing.append("description")
    if not issue_type:
        required_missing.append("issue_type")

    if required_missing:
        # Ask only for missing fields.
        msg = "I need the following to create this task: " + ", ".join(required_missing) + "."
        await send_message_reply(room_id, msg, event_id)
        return

    fields = {
        "project_key": project_key,
        "summary": summary,
        "description": description,
        "issue_type": issue_type,
        "assignee": assignee,
        "priority": priority,
        "due_date": due_date,
    }

    jira_response = await create_jira_task(fields)
    if not jira_response:
        await send_message_reply(
            room_id,
            "I could not create the Jira task due to an error.",
            event_id,
        )
        return

    key = jira_response.get("key")
    if key:
        msg = f"Created Jira task {key} for: {d.summary}"
        await send_message_reply(room_id, msg, event_id)
        _clear_pending(room_id)


async def _handle_update_task(room_id: str, event_id: str, d: AgentDecision) -> None:
    if not d.jira_key:
        await send_message_reply(
            room_id,
            "Please provide the Jira issue key (e.g., PROJ-123).",
            event_id,
        )
        return

    fields: Dict[str, Any] = {}
    if d.summary is not None:
        fields["summary"] = d.summary
    if d.description is not None:
        fields["description"] = d.description
    if d.assignee is not None:
        fields["assignee"] = d.assignee
    if d.due_date is not None:
        fields["due_date"] = safe_parse_deadline(d.due_date)

    if not fields:
        # Nothing to update; remain silent to avoid noise.
        return

    success = await update_jira_task(d.jira_key, fields)
    if success:
        await send_message_reply(
            room_id,
            f"Updated Jira task {d.jira_key}.",
            event_id,
        )
    else:
        await send_message_reply(
            room_id,
            f"Failed to update Jira task {d.jira_key}.",
            event_id,
        )


async def _handle_request_missing(room_id: str, event_id: str, d: AgentDecision) -> None:
    # If model gave us a clarification message, use it; otherwise build one
    # based on missing_fields.
    if d.clarification_message:
        await send_message_reply(room_id, d.clarification_message, event_id)
        return

    if not d.missing_fields:
        # Nothing obviously missing; stay silent rather than hallucinate.
        return

    msg = "I need the following details before proceeding: " + ", ".join(d.missing_fields) + "."
    await send_message_reply(room_id, msg, event_id)

