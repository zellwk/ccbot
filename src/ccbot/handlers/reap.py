"""Reap windows whose Telegram topic no longer exists.

Closing a topic sends an update, and topic_closed_handler ends the session
there. Deleting one sends nothing — no such update exists — so the window
keeps running Claude against a thread nothing can reach. This module covers
that silent case, and the windows left bound to no topic at all.

Telegram has no call that asks whether a thread exists. sendChatAction and
unpinAllForumTopicMessages both answer ok for a thread long gone.
reopenForumTopic answers truthfully and does nothing either way: it refuses a
missing topic with TOPIC_ID_INVALID and an open one with TOPIC_NOT_MODIFIED,
posting no service message for either. The call is read for its error, never
for its effect.

Idle time decides who is worth asking about, never who dies. A window that is
mid-turn or was used moments ago is skipped, and every kill of a bound window
comes from Telegram's own answer. An unbound window has no topic to ask about,
so only long idleness reaps it.

Key function: reap_dead_topics().
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Literal

from telegram import Bot
from telegram.error import BadRequest

from ..session import session_manager
from ..terminal_parser import parse_status_line
from ..tmux_manager import tmux_manager
from .cleanup import clear_topic_state, close_topic

logger = logging.getLogger(__name__)

# How long between passes.
REAP_INTERVAL_SECONDS = 300.0

# A window used this recently is alive; asking about it spends a call to learn
# what its own transcript already said.
RECENT_ACTIVITY_SECONDS = 120.0

# An unbound window has no topic to ask Telegram about, so nothing but a long
# silence separates one that was abandoned from one between sessions.
UNBOUND_IDLE_SECONDS = 900.0

# A burst of probes earns a 429, which answers nothing about any of them.
MAX_PROBES_PER_PASS = 5

TopicVerdict = Literal["gone", "closed", "keep"]


async def reap_dead_topics(bot: Bot) -> int:
    """Kill windows whose topic is gone. Returns how many were reaped."""
    windows = await tmux_manager.list_windows()
    bound = {
        window_id: (user_id, thread_id)
        for user_id, thread_id, window_id in session_manager.iter_thread_bindings()
    }

    reaped = 0
    probes = 0
    for window in windows:
        window_id = window.window_id
        if await _is_working(window_id):
            continue

        idle = await _idle_seconds(window_id)
        if idle is None or idle < RECENT_ACTIVITY_SECONDS:
            continue

        binding = bound.get(window_id)
        if binding is None:
            if idle >= UNBOUND_IDLE_SECONDS:
                await _reap_window(bot, window_id, None, None, "bound to no topic")
                reaped += 1
            continue

        if probes >= MAX_PROBES_PER_PASS:
            continue
        probes += 1

        user_id, thread_id = binding
        verdict = await _topic_verdict(bot, user_id, thread_id)
        if verdict == "gone":
            await _reap_window(
                bot, window_id, user_id, thread_id, "topic no longer exists"
            )
            reaped += 1
        elif verdict == "closed":
            # The probe reopened it. Closing it again puts it back and ends the
            # session through topic_closed_handler, the one path that does.
            await close_topic(bot, user_id, thread_id)
            reaped += 1

    return reaped


async def reap_loop(bot: Bot) -> None:
    """Run a reap pass on an interval, for as long as the bot runs."""
    logger.info("Reaping started (interval: %ss)", REAP_INTERVAL_SECONDS)
    while True:
        await asyncio.sleep(REAP_INTERVAL_SECONDS)
        try:
            reaped = await reap_dead_topics(bot)
            if reaped:
                logger.info("Reap pass: %d window(s) reaped", reaped)
        except Exception:
            logger.exception("Reap pass failed")


async def _topic_verdict(bot: Bot, user_id: int, thread_id: int) -> TopicVerdict:
    """Ask Telegram what became of a topic, changing nothing.

    "keep" covers every answer that is not a clear death, including a rate
    limit and an unreachable network — an unanswered question reaps nothing.
    """
    chat_id = session_manager.resolve_chat_id(user_id, thread_id)
    try:
        await bot.reopen_forum_topic(chat_id=chat_id, message_thread_id=thread_id)
    except BadRequest as e:
        if "topic_id_invalid" in str(e).lower():
            return "gone"
        return "keep"
    except Exception as e:  # noqa: BLE001 - a probe must never break the loop
        logger.debug("Reap probe error (thread=%d): %s", thread_id, e)
        return "keep"
    return "closed"


async def _reap_window(
    bot: Bot,
    window_id: str,
    user_id: int | None,
    thread_id: int | None,
    reason: str,
) -> None:
    """Kill a window and drop the state that pointed at it."""
    window = await tmux_manager.find_window_by_id(window_id)
    if window:
        await tmux_manager.kill_window(window.window_id)
    if user_id is not None and thread_id is not None:
        session_manager.unbind_thread(user_id, thread_id)
        await clear_topic_state(user_id, thread_id, bot)
    logger.info("Reaped window %s: %s", window_id, reason)


async def _is_working(window_id: str) -> bool:
    """Report whether Claude is mid-turn in this window."""
    pane = await tmux_manager.capture_pane(window_id)
    return bool(pane and parse_status_line(pane))


async def _idle_seconds(window_id: str) -> float | None:
    """Seconds since this window's transcript was last written.

    None when the question cannot be answered — no session started yet, or an
    unreadable file — which keeps a window that is still starting up alive.
    """
    session = await session_manager.resolve_session_for_window(window_id)
    if not session or not session.file_path:
        return None
    try:
        return time.time() - Path(session.file_path).stat().st_mtime
    except OSError:
        return None
