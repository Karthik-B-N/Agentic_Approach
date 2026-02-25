from __future__ import annotations

"""
Thin adapter between the existing Matrix NIO listener and the new agent.

This keeps the agent logic independent from nio types and makes it easy
to unit test by calling handle_matrix_message directly.
"""

from agentic_approach.agent import handle_message as agent_handle_message


async def handle_matrix_message(
    room_id: str,
    event_id: str,
    sender: str,
    message_text: str,
) -> None:
    """
    Entrypoint called from websocket.message_callback.

    All heavy lifting is delegated to agentic_approach.agent.
    """

    await agent_handle_message(
        room_id=room_id,
        event_id=event_id,
        sender=sender,
        message_text=message_text,
    )

