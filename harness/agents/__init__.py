from .base import Agent
from .random_agent import RandomAgent
from .wm_agent import WMMemoAgent, WMTemplateAgent, WorldModelAgent

AVAILABLE_AGENTS = {
    "random": RandomAgent,
    "wm-template": WMTemplateAgent,
    "wm-memo": WMMemoAgent,
}

__all__ = [
    "Agent",
    "RandomAgent",
    "WorldModelAgent",
    "WMTemplateAgent",
    "WMMemoAgent",
    "AVAILABLE_AGENTS",
]
