"""Scheduled jobs — opens a topic, starts a session in it, sends it a prompt.

Reads ``~/.ccbot/jobs.json`` every tick so edits land without a restart. A job
fires once its time has passed today and it has not already run; the last fire
date lives in ``~/.ccbot/jobs-fired.json`` so a restart neither double-fires
nor skips. A job whose time passed while the bot was down fires on the next
tick — late beats never for a morning report.

Job shape::

    {"name": "email-triage", "at": "08:00", "days": [0,1,2,3,4,5,6],
     "machines": ["headless"], "cwd": "/Users/zellwk/projects",
     "topic": "Email triage", "prompt": "triage emails"}

``days`` uses Python weekdays, Monday 0. Omit it for every day.

``machines`` lists the roles this job runs on, matched against ``~/.claude/.machine``
— ``headless`` on the Mac mini, ``main`` on the laptop. Omit it to run everywhere.

``next`` names a job to start when ``/done`` is sent in this job's topic, which
is how a stage says it is finished. The follower needs no ``at`` and should
carry ``"enabled": false`` so the clock never starts it on its own — chaining
calls it directly. Chains can run any depth: triage → support → action.

``/done`` rather than closing the topic because a private chat is not a forum:
Telegram sends no ``forum_topic_closed`` update there, so nothing observes a
topic closing. ``topic_closed_handler`` chains too, for a supergroup setup.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from telegram import Bot

from .config import config
from .session import session_manager
from .tmux_manager import tmux_manager
from .utils import atomic_write_json, ccbot_dir

logger = logging.getLogger(__name__)

TICK_SECONDS = 30.0

# Long enough for Claude Code's SessionStart hook to register the window.
SESSION_MAP_TIMEOUT = 5.0


async def scheduled_jobs_loop(bot: Bot) -> None:
    """Runs each due job in a topic of its own, once a day."""
    while True:
        await asyncio.sleep(TICK_SECONDS)
        try:
            fired = _load_fired_dates()
            now = datetime.now()
            for job in _due_jobs(_load_jobs(), now, fired):
                await _open_session_for_job(bot, job)
                fired[job["name"]] = now.date().isoformat()
                atomic_write_json(_fired_path(), fired)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Scheduled jobs tick failed")


def _load_fired_dates() -> dict[str, str]:
    path = _fired_path()
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _load_jobs() -> list[dict[str, Any]]:
    path = ccbot_dir() / "jobs.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _this_machine() -> str:
    """Reads the machine role Claude Code writes to ~/.claude/.machine."""
    path = Path.home() / ".claude" / ".machine"
    if not path.exists():
        return ""
    return path.read_text().strip()


def _due_jobs(
    jobs: list[dict[str, Any]], now: datetime, fired: dict[str, str]
) -> list[dict[str, Any]]:
    """Picks the jobs whose time has passed today and that have not run."""
    today = now.date().isoformat()
    machine = _this_machine()
    due = []
    for job in jobs:
        if not job.get("enabled", True):
            continue
        if machine not in job.get("machines", [machine]):
            continue
        if now.weekday() not in job.get("days", list(range(7))):
            continue
        if fired.get(job["name"]) == today:
            continue
        hour, minute = (int(part) for part in job["at"].split(":"))
        if (now.hour, now.minute) < (hour, minute):
            continue
        due.append(job)
    return due


async def _open_session_for_job(bot: Bot, job: dict[str, Any]) -> None:
    """Opens the topic, starts the session, sends the prompt."""
    user_id = next(iter(config.allowed_users))
    chat_id = session_manager.resolve_chat_id(user_id)
    title = f"{job['topic']} · {datetime.now():%b %d}"

    topic = await bot.create_forum_topic(chat_id=chat_id, name=title)
    thread_id = topic.message_thread_id

    created, detail, window_name, window_id = await tmux_manager.create_window(
        job["cwd"]
    )
    if not created:
        logger.error("Job %s: window failed: %s", job["name"], detail)
        await bot.send_message(
            chat_id, f"❌ {job['name']}: {detail}", message_thread_id=thread_id
        )
        return

    await session_manager.wait_for_session_map_entry(
        window_id, timeout=SESSION_MAP_TIMEOUT
    )
    session_manager.bind_thread(
        user_id, thread_id, window_id, window_name=window_name
    )
    # A job's topic is named on purpose. Setting auto_named here is what
    # topic_namer.maybe_autoname() checks before it renames anything, so the
    # local model leaves these rooms alone.
    session_manager.mark_auto_named(window_id, True)
    _remember_job_topic(thread_id, job["name"])

    sent, detail = await session_manager.send_to_window(window_id, job["prompt"])
    if not sent:
        logger.error("Job %s: prompt failed: %s", job["name"], detail)
        await bot.send_message(
            chat_id, f"❌ {job['name']}: {detail}", message_thread_id=thread_id
        )
        return

    logger.info(
        "Job %s: started in topic %d (window %s)", job["name"], thread_id, window_id
    )


async def open_next_job(bot: Bot, thread_id: int) -> str | None:
    """Opens the job chained after the one that owns ``thread_id``.

    Called when a topic closes. Returns the name of the job it started, or
    None when this topic came from no job, or that job names no successor.
    """
    name = _job_topics().get(str(thread_id))
    _forget_job_topic(thread_id)
    if not name:
        return None

    job = next((j for j in _load_jobs() if j["name"] == name), None)
    nxt = job and job.get("next")
    if not nxt:
        return None

    follower = next((j for j in _load_jobs() if j["name"] == nxt), None)
    if not follower:
        logger.error("Job %s: next job %r not found", name, nxt)
        return None

    await _open_session_for_job(bot, follower)
    return nxt


def job_name_for_topic(thread_id: int) -> str | None:
    """Name of the job whose topic this is, or None when it came from no job."""
    return _job_topics().get(str(thread_id))


def _job_topics() -> dict[str, str]:
    path = _job_topics_path()
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _remember_job_topic(thread_id: int, name: str) -> None:
    topics = _job_topics()
    topics[str(thread_id)] = name
    atomic_write_json(_job_topics_path(), topics)


def _forget_job_topic(thread_id: int) -> None:
    topics = _job_topics()
    if topics.pop(str(thread_id), None) is not None:
        atomic_write_json(_job_topics_path(), topics)


def _job_topics_path() -> Path:
    return ccbot_dir() / "jobs-topics.json"


def _fired_path() -> Path:
    return ccbot_dir() / "jobs-fired.json"
