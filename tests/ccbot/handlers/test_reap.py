"""Tests for reap — killing windows whose Telegram topic is gone."""

import os
import time
from types import SimpleNamespace

import pytest
from telegram.error import BadRequest

import ccbot.handlers.reap as reap_mod
from ccbot.handlers.reap import PENDING_ECHOES, consume_probe_echo, reap_dead_topics
from ccbot.session import SessionManager
from ccbot.tmux_manager import TmuxWindow

USER = 100


class FakeBot:
    """Records edit_forum_topic calls and fails the threads named dead."""

    def __init__(self, dead: set[int]) -> None:
        self.dead = dead
        self.probed: list[tuple[int, str]] = []

    async def edit_forum_topic(self, chat_id, message_thread_id, name):
        self.probed.append((message_thread_id, name))
        if message_thread_id in self.dead:
            raise BadRequest("Bad Request: TOPIC_ID_INVALID")
        return True


@pytest.fixture
def mgr(monkeypatch) -> SessionManager:
    monkeypatch.setattr(SessionManager, "_load_state", lambda self: None)
    monkeypatch.setattr(SessionManager, "_save_state", lambda self: None)
    m = SessionManager()
    monkeypatch.setattr(reap_mod, "session_manager", m)
    return m


@pytest.fixture
def killed(monkeypatch) -> list[str]:
    """Capture kill_window calls; treat every window as live in tmux."""
    out: list[str] = []

    async def fake_find(window_id: str):
        return TmuxWindow(window_id, "w", "/tmp")

    async def fake_kill(window_id: str) -> bool:
        out.append(window_id)
        return True

    monkeypatch.setattr(reap_mod.tmux_manager, "find_window_by_id", fake_find)
    monkeypatch.setattr(reap_mod.tmux_manager, "kill_window", fake_kill)

    async def not_alive(window_id: str) -> bool:
        return False

    monkeypatch.setattr(reap_mod, "_is_obviously_alive", not_alive)

    async def noop_clear(*a, **kw) -> None:
        return None

    monkeypatch.setattr(reap_mod, "clear_topic_state", noop_clear)
    return out


@pytest.fixture(autouse=True)
def clear_echoes():
    PENDING_ECHOES.clear()
    yield
    PENDING_ECHOES.clear()


class TestReapDeadTopics:
    @pytest.mark.asyncio
    async def test_dead_topic_is_killed(self, mgr, killed) -> None:
        mgr.bind_thread(USER, 11, "@1", window_name="dead-one")
        bot = FakeBot(dead={11})

        assert await reap_dead_topics(bot, USER) == 1
        assert killed == ["@1"]
        assert mgr.get_window_for_thread(USER, 11) is None

    @pytest.mark.asyncio
    async def test_live_topic_survives(self, mgr, killed) -> None:
        mgr.bind_thread(USER, 22, "@2", window_name="alive")
        bot = FakeBot(dead=set())

        assert await reap_dead_topics(bot, USER) == 0
        assert killed == []
        assert mgr.get_window_for_thread(USER, 22) == "@2"

    @pytest.mark.asyncio
    async def test_live_probe_queues_its_echo_for_deletion(self, mgr, killed) -> None:
        """The rename notice Telegram posts must be cleaned up, not shown."""
        mgr.bind_thread(USER, 22, "@2", window_name="alive")
        await reap_dead_topics(FakeBot(dead=set()), USER)

        assert (USER, 22, "alive") in PENDING_ECHOES

    @pytest.mark.asyncio
    async def test_dead_probe_queues_nothing(self, mgr, killed) -> None:
        """A dead thread posts no service message, so nothing to clean up."""
        mgr.bind_thread(USER, 11, "@1", window_name="dead-one")
        await reap_dead_topics(FakeBot(dead={11}), USER)

        assert not PENDING_ECHOES

    @pytest.mark.asyncio
    async def test_probe_sends_the_name_the_topic_already_has(
        self, mgr, killed
    ) -> None:
        """Any other name would rename a live topic instead of testing it."""
        mgr.bind_thread(USER, 22, "@2", window_name="Real Name")
        bot = FakeBot(dead=set())

        await reap_dead_topics(bot, USER)
        assert bot.probed == [(22, "Real Name")]

    @pytest.mark.asyncio
    async def test_new_topic_is_skipped(self, mgr, killed) -> None:
        """The topic that triggered the sweep is alive by definition."""
        mgr.bind_thread(USER, 33, "@3", window_name="brand-new")
        bot = FakeBot(dead={33})

        assert await reap_dead_topics(bot, USER, skip_thread=33) == 0
        assert bot.probed == []
        assert killed == []

    @pytest.mark.asyncio
    async def test_busy_window_never_probed(self, mgr, monkeypatch, killed) -> None:
        """A window mid-turn is alive; asking would only make noise."""
        mgr.bind_thread(USER, 44, "@4", window_name="working")

        async def alive(window_id: str) -> bool:
            return True

        monkeypatch.setattr(reap_mod, "_is_obviously_alive", alive)
        bot = FakeBot(dead={44})

        assert await reap_dead_topics(bot, USER) == 0
        assert bot.probed == []
        assert killed == []

    @pytest.mark.asyncio
    async def test_window_id_fallback_name_never_probed(self, mgr, killed) -> None:
        """Renaming a live topic to "@4" is worse than missing a sweep."""
        mgr.bind_thread(USER, 55, "@5")
        bot = FakeBot(dead={55})

        assert await reap_dead_topics(bot, USER) == 0
        assert bot.probed == []

    @pytest.mark.asyncio
    async def test_other_errors_do_not_kill(self, mgr, killed) -> None:
        """Only TOPIC_ID_INVALID means gone — a flaky call must not reap."""
        mgr.bind_thread(USER, 66, "@6", window_name="flaky")

        class Flaky(FakeBot):
            async def edit_forum_topic(self, chat_id, message_thread_id, name):
                raise BadRequest("Bad Request: CHAT_WRITE_FORBIDDEN")

        assert await reap_dead_topics(Flaky(dead=set()), USER) == 0
        assert killed == []
        assert mgr.get_window_for_thread(USER, 66) == "@6"

    @pytest.mark.asyncio
    async def test_other_users_untouched(self, mgr, killed) -> None:
        mgr.bind_thread(999, 77, "@7", window_name="theirs")
        bot = FakeBot(dead={77})

        assert await reap_dead_topics(bot, USER) == 0
        assert killed == []


class TestConsumeProbeEcho:
    @pytest.mark.asyncio
    async def test_matching_echo_is_consumed_once(self) -> None:
        PENDING_ECHOES.add((USER, 22, "alive"))
        assert await consume_probe_echo(None, USER, 22, "alive") is True
        assert await consume_probe_echo(None, USER, 22, "alive") is False

    @pytest.mark.asyncio
    async def test_real_rename_passes_through(self) -> None:
        """A rename Zell made must reach the normal sync path."""
        PENDING_ECHOES.add((USER, 22, "alive"))
        assert await consume_probe_echo(None, USER, 22, "a name he chose") is False


class TestIsObviouslyAlive:
    """A status line is only trusted while the transcript is still growing."""

    STATUS_PANE = "✳ Gitifying… (2m 1s · ↓ 6.5k tokens)\n"

    @staticmethod
    def _setup(monkeypatch, tmp_path, pane: str, age_seconds: float) -> None:
        transcript = tmp_path / "session.jsonl"
        transcript.write_text("{}")
        os.utime(transcript, (time.time(), time.time() - age_seconds))

        async def fake_session(window_id: str):
            return SimpleNamespace(file_path=str(transcript))

        async def fake_capture(window_id: str) -> str:
            return pane

        monkeypatch.setattr(
            reap_mod.session_manager, "resolve_session_for_window", fake_session
        )
        monkeypatch.setattr(reap_mod.tmux_manager, "capture_pane", fake_capture)

    @pytest.mark.asyncio
    async def test_working_window_is_alive(self, mgr, monkeypatch, tmp_path) -> None:
        self._setup(monkeypatch, tmp_path, self.STATUS_PANE, age_seconds=5)
        assert await reap_mod._is_obviously_alive("@1") is True

    @pytest.mark.asyncio
    async def test_stale_spinner_is_not_alive(self, mgr, monkeypatch, tmp_path) -> None:
        """A turn that dies mid-request leaves its status line painted for hours."""
        self._setup(monkeypatch, tmp_path, self.STATUS_PANE, age_seconds=4 * 3600)
        assert await reap_mod._is_obviously_alive("@1") is False

    @pytest.mark.asyncio
    async def test_recently_used_idle_window_is_alive(
        self, mgr, monkeypatch, tmp_path
    ) -> None:
        self._setup(monkeypatch, tmp_path, "❯ \n", age_seconds=5)
        assert await reap_mod._is_obviously_alive("@1") is True
