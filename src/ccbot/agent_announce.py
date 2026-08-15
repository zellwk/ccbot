"""Announce which agent is answering a topic when that agent changes.

A Telegram topic survives an agent switch: switch-agent.sh respawns the pane
under a different --agent while the tmux window id stays put, so the topic
binding holds and nothing in the chat marks the change. The incoming Claude
session cannot announce itself either — bot-launch-ccbot.sh execs `claude`
with no initial prompt, so the session boots and waits, producing no turn
until a message arrives.

The session monitor sees the window's session id change and calls
announce_agent_change(), which posts the new agent's name into the bound topic.

Announcements fire only when the resolved agent differs from the last one
announced for that window, so /clear and compaction stay quiet.

Agent resolution mirrors bot-launch-ccbot.sh: the per-window override wins,
then the working directory's .bot-agent, then shiz.
"""

import json
import logging
from pathlib import Path
from typing import Any

from .utils import atomic_write_json, ccbot_dir

logger = logging.getLogger(__name__)

_DEFAULT_AGENT = "shiz"


def _announced_file() -> Path:
    return ccbot_dir() / "announced_agents.json"


def resolve_agent(window_id: str, cwd: str) -> str:
    """Resolve the agent name for a window, mirroring bot-launch-ccbot.sh."""
    override = ccbot_dir() / "agent" / window_id
    try:
        name = override.read_text().strip()
        if name:
            return name
    except OSError:
        pass

    if cwd:
        try:
            name = (Path(cwd) / ".bot-agent").read_text().strip()
            if name:
                return name
        except OSError:
            pass

    return _DEFAULT_AGENT


def _load_announced() -> dict[str, str]:
    try:
        return json.loads(_announced_file().read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _save_announced(announced: dict[str, str]) -> None:
    try:
        atomic_write_json(_announced_file(), announced)
    except OSError as e:
        logger.error("Failed to persist announced agents: %s", e)


async def announce_agent_change(window_id: str, bot: Any) -> None:
    """Post the agent now answering `window_id` to its bound topic.

    No-ops when the agent is unchanged, or when the window has no topic bound
    yet — an unbound window has nowhere to post, and recording the agent now
    would suppress the announcement once a topic does bind.
    """
    from .session import session_manager

    ws = session_manager.window_states.get(window_id)
    if ws is None:
        logger.debug("No window state for %s, skipping announce", window_id)
        return

    agent = resolve_agent(window_id, ws.cwd or "")

    announced = _load_announced()
    if announced.get(window_id) == agent:
        logger.debug("Agent unchanged for %s (%s), skipping announce", window_id, agent)
        return

    if ws.user_id is None:
        logger.debug("Window %s has no bound topic, skipping announce", window_id)
        return

    chat_id = session_manager.resolve_chat_id(ws.user_id, ws.thread_id)
    try:
        await bot.send_message(
            chat_id=chat_id,
            message_thread_id=ws.thread_id,
            text=f"{agent} is here.",
        )
    except Exception as e:
        logger.error("Failed to announce agent for %s: %s", window_id, e)
        return

    announced[window_id] = agent
    _save_announced(announced)
    logger.info("Announced agent for %s: %s", window_id, agent)
