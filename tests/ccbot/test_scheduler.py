"""Tests for a job's standing topic and the chain that follows its report.

A job reuses one open topic for every run. A closed topic is done with and a
deleted one is unreachable, so either gets a new topic, and the closed one is
never reopened. The chain to a job's
followers fires only from a delivered report and at most once a day. A chain
fired twice works the same customer case twice, so each trigger that must stay
silent is pinned here.
"""

import asyncio
import json
from types import SimpleNamespace

import pytest
from telegram.error import BadRequest

from ccbot import scheduler
from ccbot.handlers import cleanup
from ccbot.scheduler import JobReport
from ccbot.session import SessionManager

TRIAGE = {
    "name": "fake-triage",
    "cwd": "/tmp",
    "topic": "Fake triage",
    "prompt": "say hi",
    "next": ["fake-support", "fake-action"],
}
SUPPORT = {
    "name": "fake-support",
    "enabled": False,
    "cwd": "/tmp",
    "topic": "S",
    "prompt": "hi",
}
ACTION = {
    "name": "fake-action",
    "enabled": False,
    "cwd": "/tmp",
    "topic": "A",
    "prompt": "hi",
}


class ForumBot:
    """A forum where each topic is open, closed or gone, as Telegram reports it.

    An admin bot can still post in a closed topic, so only a gone one rejects sends.
    """

    def __init__(self) -> None:
        self.topics: dict[int, str] = {}
        self.created: list[str] = []
        self.sent: list[tuple[int, str]] = []
        self.next_thread = 500

    async def create_forum_topic(self, chat_id: int, name: str):
        self.next_thread += 1
        self.topics[self.next_thread] = "open"
        self.created.append(name)
        return SimpleNamespace(message_thread_id=self.next_thread)

    async def reopen_forum_topic(self, chat_id: int, message_thread_id: int) -> bool:
        self.topics[message_thread_id] = "reopened"
        return True

    async def send_message(self, chat_id: int, text: str, **kwargs) -> None:
        thread = kwargs["message_thread_id"]
        if thread not in self.topics:
            raise BadRequest("Message thread not found")
        self.sent.append((thread, text))


class FakeTmux:
    """Windows by id. Creating one fails, so a run stops once its report is posted."""

    def __init__(self, live: set[str] | None = None) -> None:
        self.live = live or set()
        self.killed: list[str] = []

    async def find_window_by_id(self, window_id: str):
        return SimpleNamespace(window_id=window_id) if window_id in self.live else None

    async def kill_window(self, window_id: str) -> None:
        self.live.discard(window_id)
        self.killed.append(window_id)

    async def create_window(self, cwd: str, resume_session_id: str | None = None):
        return False, "no tmux in tests", "", ""


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("CCBOT_DIR", str(tmp_path))
    (tmp_path / "jobs.json").write_text(json.dumps([TRIAGE, SUPPORT, ACTION]))

    monkeypatch.setattr(SessionManager, "_load_state", lambda self: None)
    monkeypatch.setattr(SessionManager, "_save_state", lambda self: None)
    mgr = SessionManager()
    tmux = FakeTmux()
    for module in (scheduler, cleanup):
        monkeypatch.setattr(module, "session_manager", mgr)
    monkeypatch.setattr(scheduler, "tmux_manager", tmux)
    monkeypatch.setattr(cleanup, "tmux_manager", tmux)

    worked: list[str] = []
    reports: dict[str, JobReport | None] = {}

    async def fake_report(job):
        worked.append(job["name"])
        return reports.get(job["name"], JobReport(f"{job['name']} report", "sid"))

    monkeypatch.setattr(scheduler, "_get_job_report", fake_report)
    return SimpleNamespace(
        bot=ForumBot(), mgr=mgr, tmux=tmux, worked=worked, reports=reports, dir=tmp_path
    )


async def run(bot, job) -> None:
    """Runs one job and every chain it starts, to the end."""
    await scheduler._open_session_for_job(bot, job)
    while pending := [
        t for t in asyncio.all_tasks() if t is not asyncio.current_task()
    ]:
        await asyncio.gather(*pending)


def stored_topics(env) -> dict[str, int]:
    return json.loads((env.dir / "jobs-topics.json").read_text())


class TestStandingTopic:
    async def test_first_run_makes_the_topic_under_the_jobs_name(self, env) -> None:
        await run(env.bot, SUPPORT)
        assert env.bot.created == ["S"]
        assert stored_topics(env) == {"fake-support": 501}

    async def test_open_topic_is_reused(self, env) -> None:
        await run(env.bot, SUPPORT)
        await run(env.bot, SUPPORT)
        assert env.bot.created == ["S"]
        assert {thread for thread, _ in env.bot.sent} == {501}

    async def test_closed_topic_is_left_and_replaced(self, env) -> None:
        await run(env.bot, SUPPORT)
        env.bot.topics[501] = "closed"
        scheduler.forget_closed_topic(501)  # what topic_closed_handler does

        await run(env.bot, SUPPORT)

        assert env.bot.created == ["S", "S"]
        assert stored_topics(env) == {"fake-support": 502}
        assert env.bot.topics[501] == "closed"
        assert [thread for thread, _ in env.bot.sent].count(501) == 2  # first run only

    async def test_closing_a_topic_no_job_owns_changes_nothing(self, env) -> None:
        await run(env.bot, SUPPORT)
        scheduler.forget_closed_topic(999)
        assert stored_topics(env) == {"fake-support": 501}

    async def test_deleted_topic_is_replaced(self, env) -> None:
        await run(env.bot, SUPPORT)
        del env.bot.topics[501]
        await run(env.bot, SUPPORT)
        assert env.bot.created == ["S", "S"]
        assert stored_topics(env) == {"fake-support": 502}
        assert env.bot.sent[-1][0] == 502

    async def test_last_runs_session_ends_and_the_topic_stays(self, env) -> None:
        await run(env.bot, SUPPORT)
        env.tmux.live.add("@7")
        env.mgr.bind_thread(12345, 501, "@7")

        await run(env.bot, SUPPORT)

        assert env.tmux.killed == ["@7"]
        assert env.mgr.get_binding_for_thread(501) is None
        assert env.bot.topics[501] == "open"


class TestChain:
    async def test_delivered_report_starts_each_follower_in_order(self, env) -> None:
        await run(env.bot, TRIAGE)
        assert env.worked == ["fake-triage", "fake-support", "fake-action"]

    async def test_followers_run_once_a_day_however_often_triggered(self, env) -> None:
        await run(env.bot, TRIAGE)
        await run(env.bot, TRIAGE)
        assert env.worked.count("fake-support") == 1
        assert env.worked.count("fake-action") == 1

    async def test_failed_pass_starts_no_follower(self, env) -> None:
        env.reports["fake-triage"] = JobReport("❌ failed", "", True)
        await run(env.bot, TRIAGE)
        assert env.worked == ["fake-triage"]

    async def test_nothing_to_report_starts_no_follower(self, env) -> None:
        env.reports["fake-triage"] = None
        await run(env.bot, TRIAGE)
        assert env.worked == ["fake-triage"]

    async def test_ending_the_triage_session_starts_no_follower(self, env) -> None:
        """Closing a topic, /done and a reap all end a session through here."""
        (env.dir / "jobs-topics.json").write_text(json.dumps({"fake-triage": 501}))
        env.tmux.live.add("@9")
        env.mgr.bind_thread(12345, 501, "@9")

        await cleanup.end_session(env.bot, 501)
        await asyncio.sleep(0)

        assert env.tmux.killed == ["@9"]
        assert env.worked == []
