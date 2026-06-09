from .base import Agent
from .random_agent import RandomAgent

AVAILABLE_AGENTS = {
    "random": RandomAgent,
}

__all__ = ["Agent", "RandomAgent", "AVAILABLE_AGENTS"]
