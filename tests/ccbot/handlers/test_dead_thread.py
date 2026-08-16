"""Tests for dead-topic handling on the send path.

Telegram answers a send into a deleted topic with "Message thread not found".
Swallowing that answer leaves the status poller re-sending once a second for as
long as the window shows a status line, which earns a flood ban on the chat.
"""

import pytest
from telegram.error import BadRequest

import ccbot.handlers.message_queue as mq
from ccbot.handlers.message_queue import MessageTask
from ccbot.handlers.message_sender import DeadThread, safe_send, send_with_fallback
from ccbot.session import SessionManager

USER = 100
CHAT = 555


class DeadThreadBot:
    """Rejects every send the way Telegram rejects a deleted topic."""

    def __init__(self) -> None:
        self.sends = 0

    async def send_message(self, **kwargs) -> None:
        self.sends += 1
        raise BadRequest("Message thread not found")


@pytest.fixture
def mgr(monkeypatch) -> SessionManager:
    monkeypatch.setattr(SessionManager, "_load_state", lambda self: None)
    monkeypatch.setattr(SessionManager, "_save_state", lambda self: None)
    m = SessionManager()
    monkeypatch.setattr(mq, "session_manager", m)
    return m


class TestSendRaises:
    @pytest.mark.asyncio
    async def test_send_with_fallback_raises(self) -> None:
        bot = DeadThreadBot()
        with pytest.raises(DeadThread):
            await send_with_fallback(bot, CHAT, "hello", message_thread_id=7)
        assert bot.sends == 1  # plain-text retry is pointless and skipped

    @pytest.mark.asyncio
    async def test_safe_send_raises(self) -> None:
        bot = DeadThreadBot()
        with pytest.raises(DeadThread):
            await safe_send(bot, CHAT, "hello", message_thread_id=7)
        assert bot.sends == 1

    @pytest.mark.asyncio
    async def test_other_bad_request_still_falls_back(self) -> None:
        """A formatting rejection keeps the existing plain-text retry."""
        calls: list[dict] = []

        class PickyBot:
            async def send_message(self, **kwargs):
                calls.append(kwargs)
                if "parse_mode" in kwargs:
                    raise BadRequest("Can't parse entities")
                return "sent"

        assert await send_with_fallback(PickyBot(), CHAT, "*x", message_thread_id=7)
        assert len(calls) == 2


class TestDropDeadThread:
    @pytest.mark.asyncio
    async def test_unbinds_the_thread(self, mgr) -> None:
        mgr.bind_thread(USER, 42, "@9", window_name="gone")

        await mq._drop_dead_thread(
            None, USER, 42, BadRequest("Message thread not found")
        )

        assert mgr.get_window_for_thread(USER, 42) is None

    @pytest.mark.asyncio
    async def test_keeps_the_window(self, mgr) -> None:
        """The window survives unbound, so it stays in the window picker."""
        mgr.bind_thread(USER, 42, "@9", window_name="gone")

        await mq._drop_dead_thread(
            None, USER, 42, BadRequest("Message thread not found")
        )

        assert "@9" in mgr.window_states

    @pytest.mark.asyncio
    async def test_status_update_surfaces_dead_thread(self, mgr) -> None:
        """The status path is where the retry loop lived — it must raise."""
        mgr.bind_thread(USER, 42, "@9", window_name="gone")
        task = MessageTask(
            task_type="status_update", text="Thinking…", window_id="@9", thread_id=42
        )

        with pytest.raises(DeadThread):
            await mq._process_status_update_task(DeadThreadBot(), USER, task)
