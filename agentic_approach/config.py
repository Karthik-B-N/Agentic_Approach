import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConfig:
    """
    Central configuration for the Matrix → Jira task agent.

    Values are primarily sourced from environment variables so they can be
    tuned without code changes.
    """

    # How many recent messages to consider when summarizing a room.
    last_n_messages: int = int(os.getenv("AGENT_LAST_N_MESSAGES", "20"))

    # Minimum confidence for a classification to be considered task-related.
    classification_threshold: float = float(
        os.getenv("AGENT_CLASSIFICATION_THRESHOLD", "0.7")
    )

    # Default Jira project key to use when none is explicitly mentioned.
    default_project_key: str | None = os.getenv("JIRA_PROJECT_KEY") or os.getenv(
        "AGENT_DEFAULT_PROJECT_KEY"
    )

    # Groq model name used for all LangChain calls.
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    # Toggle for enabling the agent from the Matrix listener.
    enabled: bool = os.getenv("AGENT_ENABLED", "true").lower() == "true"


CONFIG = AgentConfig()

