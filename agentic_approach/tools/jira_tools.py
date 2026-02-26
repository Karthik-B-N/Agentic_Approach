from __future__ import annotations

from typing import Any, Dict, Optional, List
import asyncio

from langchain_core.tools import Tool

from jira_client import (
    create_jira_issue,
    update_jira_issue,
    resolve_account_id as _jira_resolve_account_id,
    get_active_issues as _jira_get_active_issues,
)
from agentic_approach.config import CONFIG


async def create_jira_task(fields: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Create a Jira task using the existing jira_client helper.

    Required fields:
        - project_key
        - summary
        - description
        - issue_type  (currently only 'Task' is respected)

    Optional:
        - assignee
        - priority  (ignored for now, but accepted)
        - due_date  (string)
    """
    project_key = fields.get("project_key") or CONFIG.default_project_key
    summary = fields.get("summary")
    description = fields.get("description") or ""
    assignee = fields.get("assignee")
    due_date = fields.get("due_date")

    if not project_key or not summary:
        # Fail closed; caller must have validated required fields.
        return None

    # The underlying jira_client uses a single configured project key,
    # so we currently ignore project_key differences. This can be
    # extended to support multi-project routing if needed.
    return await create_jira_issue(
        title=summary,
        description=description,
        assigned_to=assignee,
        deadline=due_date,
    )


async def update_jira_task(issue_key: str, fields: Dict[str, Any]) -> bool:
    """
    Update an existing Jira task. Only explicitly provided fields are sent.

    Allowed fields:
        - summary
        - description
        - assignee
        - due_date
    """
    if not issue_key:
        return False

    return await update_jira_issue(
        jira_issue_key=issue_key,
        title=fields.get("summary"),
        description=fields.get("description"),
        assigned_to=fields.get("assignee"),
        deadline=fields.get("due_date"),
    )


async def resolve_jira_account(display_name: str) -> Optional[str]:
    """
    Look up a Jira user by display name or email fragment and return
    their accountId, or None if no match is found.
    """
    if not display_name:
        return None
    return await _jira_resolve_account_id(display_name)


async def get_active_issues_agent(room_id: str | None = None) -> List[Dict[str, Any]]:
    """
    Thin wrapper around jira_client.get_active_issues, exposed for tools
    and for the agent to use when deciding create vs update.
    """
    return await _jira_get_active_issues(room_id)

def _create_jira_task_sync(fields: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Synchronous wrapper for create_jira_task, for LangChain tool compatibility.

    Note: this should not be used from within an existing running event loop.
    In async contexts, prefer the coroutine version exposed via Tool.coroutine.
    """
    return asyncio.run(create_jira_task(fields))


def _update_jira_task_sync(issue_key: str, fields: Dict[str, Any]) -> bool:
    """
    Synchronous wrapper for update_jira_task, for LangChain tool compatibility.

    Note: this should not be used from within an existing running event loop.
    In async contexts, prefer the coroutine version exposed via Tool.coroutine.
    """
    return asyncio.run(update_jira_task(issue_key, fields))


def _resolve_jira_account_sync(display_name: str) -> Optional[str]:
    """
    Synchronous wrapper for resolve_jira_account, for LangChain tools.
    """
    return asyncio.run(resolve_jira_account(display_name))


def _get_active_issues_sync(room_id: str | None = None) -> List[Dict[str, Any]]:
    """
    Synchronous wrapper for get_active_issues_agent, for LangChain tools.
    """
    return asyncio.run(get_active_issues_agent(room_id))


CREATE_JIRA_TASK_TOOL: Tool = Tool(
    name="create_jira_task",
    description=(
        "Create a Jira task. Expects a JSON object with fields: "
        "project_key, summary, description, issue_type, and optional "
        "assignee, priority, due_date."
    ),
    func=_create_jira_task_sync,
    coroutine=create_jira_task,
)


UPDATE_JIRA_TASK_TOOL: Tool = Tool(
    name="update_jira_task",
    description=(
        "Update an existing Jira task by issue key. Expects the issue_key and a "
        "JSON object with any of: summary, description, assignee, due_date."
    ),
    func=_update_jira_task_sync,
    coroutine=update_jira_task,
)


RESOLVE_ACCOUNT_ID_TOOL: Tool = Tool(
    name="resolve_jira_account",
    description=(
        "Resolve a Jira user accountId from a display name or email fragment. "
        "Returns the accountId string or null if not found."
    ),
    func=_resolve_jira_account_sync,
    coroutine=resolve_jira_account,
)


GET_ACTIVE_ISSUES_TOOL: Tool = Tool(
    name="get_active_issues",
    description=(
        "Fetch open Jira issues for the configured project. Returns a list of "
        "objects with keys: key, title, description, assigned_to, deadline, status."
    ),
    func=_get_active_issues_sync,
    coroutine=get_active_issues_agent,
)


