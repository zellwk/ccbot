"""Tests for the topic-liveness probe that decides which windows get reaped.

reopenForumTopic is the one Telegram call that answers truthfully whether a
thread still exists, and it is read for its error rather than its effect. A
wrong reading here kills a live Claude session, so each answer Telegram can
give is pinned to the verdict it produces.
"""

import pytest
from telegram.error import BadRequest, RetryAfter

from ccbot.handlers.reap import _topic_verdict

USER = 100
THREAD = 42


class ReopenBot:
    """Answers reopen_forum_topic the way Telegram would for one topic state."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls = 0

    async def reopen_forum_topic(self, chat_id: int, message_thread_id: int) -> bool:
        self.calls += 1
        if self.error:
            raise self.error
        return True


@pytest.mark.asyncio
async def test_deleted_topic_reads_as_gone():
    bot = ReopenBot(BadRequest("Bad Request: TOPIC_ID_INVALID"))
    assert await _topic_verdict(bot, USER, THREAD) == "gone"


@pytest.mark.asyncio
async def test_open_topic_is_kept():
    bot = ReopenBot(BadRequest("Bad Request: TOPIC_NOT_MODIFIED"))
    assert await _topic_verdict(bot, USER, THREAD) == "keep"


@pytest.mark.asyncio
async def test_rate_limit_kills_nothing():
    bot = ReopenBot(RetryAfter(30))
    assert await _topic_verdict(bot, USER, THREAD) == "keep"


@pytest.mark.asyncio
async def test_unrecognised_rejection_kills_nothing():
    bot = ReopenBot(BadRequest("Bad Request: CHAT_ADMIN_REQUIRED"))
    assert await _topic_verdict(bot, USER, THREAD) == "keep"


@pytest.mark.asyncio
async def test_success_means_the_topic_was_closed():
    bot = ReopenBot()
    assert await _topic_verdict(bot, USER, THREAD) == "closed"


@pytest.mark.asyncio
async def test_probe_asks_once_per_thread():
    bot = ReopenBot(BadRequest("Bad Request: TOPIC_ID_INVALID"))
    await _topic_verdict(bot, USER, THREAD)
    assert bot.calls == 1
