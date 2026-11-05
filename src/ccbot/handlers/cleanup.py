"""Unified cleanup API for topic state.

Provides centralized cleanup functions that coordinate state cleanup across
all modules, preventing memory leaks when topics are deleted.

Functions:
  - clear_topic_state: Clean up all memory state for a specific topic
  - end_session: End the session a topic holds, leaving the topic itself alone
  - close_topic: Close the Telegram topic whose session has ended
"""

import logging
from typing import Any

from telegram import Bot

from ..session import session_manager
from ..tmux_manager import tmux_manager
from .interactive_ui import clear_interactive_msg
from .message_queue import clear_status_msg_info, clear_tool_msg_ids_for_topic

logger = logging.getLogger(__name__)


async def clear_topic_state(
    user_id: int,
    thread_id: int,
    bot: Bot | None = None,
    user_data: dict[str, Any] | None = None,
) -> None:
    """Clear all memory state associated with a topic.

    This should be called when:
      - A topic is closed or deleted
      - A thread binding becomes stale (window deleted externally)

    Cleans up:
      - _status_msg_info (status message tracking)
      - _tool_msg_ids (tool_use → message_id mapping)
      - _interactive_msgs and _interactive_mode (interactive UI state)
      - user_data pending state (_pending_thread_id, _pending_thread_text)
    """
    # Clear status message tracking
    clear_status_msg_info(user_id, thread_id)

    # Clear tool message ID tracking
    clear_tool_msg_ids_for_topic(user_id, thread_id)

    # Clear interactive UI state (also deletes message from chat)
    await clear_interactive_msg(user_id, bot, thread_id)

    # Clear pending thread state from user_data
    if user_data is not None:
        if user_data.get("_pending_thread_id") == thread_id:
            user_data.pop("_pending_thread_id", None)
            user_data.pop("_pending_thread_text", None)


async def end_session(
    bot: Bot, thread_id: int, user_data: dict[str, Any] | None = None
) -> None:
    """LOCAL PATCH: end the session a topic holds. The topic itself stays.

    Killing the window is what ends a session; dropping the binding and clearing
    the topic's memory state follow from it. A closed topic, /done and a job's
    next run all arrive here, so none grows its own version of the teardown.

    The binding goes before the window dies. The status poller closes the topic
    of any binding whose window is gone, so the other order lets it close a
    job's standing topic in the moment between the two.

    The binding is read by thread rather than by the acting user. ccbot ends
    sessions itself, and the service message for a close it authored carries
    the bot as its sender.
    """
    binding = session_manager.get_binding_for_thread(thread_id)
    if not binding:
        logger.debug("Session ended: no binding (thread=%d)", thread_id)
        return

    user_id, window_id = binding
    display = session_manager.get_display_name(window_id)
    session_manager.unbind_thread(user_id, thread_id)
    await clear_topic_state(user_id, thread_id, bot, user_data)

    window = await tmux_manager.find_window_by_id(window_id)
    if window:
        await tmux_manager.kill_window(window.window_id)
        logger.info(
            "Session ended: killed window %s (user=%d, thread=%d)",
            display,
            user_id,
            thread_id,
        )
    else:
        logger.info(
            "Session ended: window %s already gone (user=%d, thread=%d)",
            display,
            user_id,
            thread_id,
        )


async def close_topic(bot: Bot, user_id: int, thread_id: int) -> None:
    """Close the Telegram topic whose session has ended.

    Closing a topic is what ends a session, so a window that exited on its own
    leaves its topic open with nothing behind it. Closing it here keeps the
    open topics equal to the live sessions. A topic already closed or already
    gone rejects the call, which is not worth reporting.
    """
    chat_id = session_manager.resolve_chat_id(user_id, thread_id)
    try:
        await bot.close_forum_topic(chat_id=chat_id, message_thread_id=thread_id)
    except Exception as e:  # noqa: BLE001 - a topic that won't close is not fatal
        logger.debug("Could not close topic %d: %s", thread_id, e)
