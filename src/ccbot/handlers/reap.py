"""Reap windows whose Telegram topic no longer exists.

Telegram sends nothing when a topic is closed or deleted — both are silent,
and both leave a tmux window running Claude against a thread that can no
longer be reached. There is also no read-only way to ask whether a thread
still exists: ``sendChatAction`` and ``unpinAllForumTopicMessages`` both
return ok for threads that are long gone.

The one call that answers truthfully is ``editForumTopic`` with a name. On a
dead thread it fails with ``TOPIC_ID_INVALID`` and posts nothing, because
there is no topic to post into. On a live thread it succeeds and posts a
rename service message — so the probe re-sends the name the topic already
has, and deletes the echo when it arrives (see ``rename_echo``).

Probing runs when a new topic is created, which is the only moment the
window population grows. Idle time decides who is worth asking about, never
who dies: a window that is mid-turn or has just been used is skipped as
obviously alive, and every kill comes from Telegram's own answer.
"""

import logging
import time
from pathlib import Path

from telegram import Bot
from telegram.error import BadRequest

from ..rename_echo import expect_rename_echo
from ..session import session_manager
from ..terminal_parser import parse_status_line
from ..tmux_manager import tmux_manager
from .cleanup import clear_topic_state

logger = logging.getLogger(__name__)

# A window used this recently is alive; asking about it would only produce a
# service message we would then have to delete.
RECENT_ACTIVITY_SECONDS = 120.0

# How long a status line is taken at face value. A turn that dies mid-request
# leaves the spinner painted in the pane indefinitely, which would otherwise
# make the window permanently unprobeable.
STATUS_LINE_TRUST_SECONDS = 600.0


async def _idle_seconds(window_id: str) -> float | None:
    """Seconds since the window's transcript last grew, or None if unknown."""
    session = await session_manager.resolve_session_for_window(window_id)
    if not (session and session.file_path):
        return None
    try:
        return time.time() - Path(session.file_path).stat().st_mtime
    except OSError:
        return None


async def _is_obviously_alive(window_id: str) -> bool:
    """Check whether a window is mid-turn or was used moments ago."""
    idle = await _idle_seconds(window_id)
    if idle is None:
        return False

    pane = await tmux_manager.capture_pane(window_id)
    if pane and parse_status_line(pane):
        # Claude is working — do not disturb it, unless the transcript stopped
        # growing long ago and the status line is just a stuck spinner.
        return idle < STATUS_LINE_TRUST_SECONDS

    return idle < RECENT_ACTIVITY_SECONDS


async def reap_dead_topics(
    bot: Bot, user_id: int, skip_thread: int | None = None
) -> int:
    """Kill windows whose topic is gone. Returns how many were reaped."""
    reaped = 0
    for uid, thread_id, wid in list(session_manager.iter_thread_bindings()):
        if uid != user_id or thread_id == skip_thread:
            continue

        name = session_manager.get_display_name(wid)
        if name.startswith("@"):
            # No real name to send back — renaming a live topic to "@26"
            # would be worse than skipping it this round.
            continue
        if await _is_obviously_alive(wid):
            continue

        chat_id = session_manager.resolve_chat_id(uid, thread_id)
        try:
            await bot.edit_forum_topic(
                chat_id=chat_id,
                message_thread_id=thread_id,
                name=name,
            )
        except BadRequest as e:
            if "topic_id_invalid" not in str(e).lower():
                logger.debug("Reap probe failed for %s: %s", wid, e)
                continue
            w = await tmux_manager.find_window_by_id(wid)
            if w:
                await tmux_manager.kill_window(w.window_id)
            session_manager.unbind_thread(uid, thread_id)
            await clear_topic_state(uid, thread_id, bot)
            logger.info(
                "Reaped '%s' (window %s, thread %d): topic no longer exists",
                name,
                wid,
                thread_id,
            )
            reaped += 1
            continue
        except Exception as e:  # noqa: BLE001 - a probe must never break delivery
            logger.debug("Reap probe error for %s: %s", wid, e)
            continue

        # Alive. The rename echo is on its way; mark it so it gets deleted
        # rather than shown.
        expect_rename_echo(chat_id, thread_id, name)

    return reaped
