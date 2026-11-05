"""Scheduled jobs — works each job headless, and posts only when it has something.

A job is worked by ``claude -p`` first, with nothing on Telegram. When that pass
reports nothing, the job ends there: no post, no notification, no tmux window to
close by hand. When it does have something, the report is posted in the job's
topic and a window resumes that same session — so a reply lands in the
conversation that already did the work rather than repeating it.

Each job owns one standing topic, named by its ``topic`` field and reused on every
run. ``jobs-topics.json`` maps job name to thread. A run ends the session the
topic still holds from last time, then binds its own; the topic itself stays.

Closing a topic means it is done with. ``topic_closed_handler`` drops the job's
entry, and the next run makes a new topic, leaving the closed one as it is.
Nothing probes a topic for its state: the one call that reveals it reopens a
closed topic as it asks. A deleted topic shows when the report's send fails,
and the report goes to a new topic instead.

``when`` names a shell command that decides due-ness before the headless pass runs.
Exit 0 opens the gate, anything else ends the job. It exists because a headless pass
costs real money whether or not it finds anything, so a job whose due-ness is plain
from a file should answer that in code. Omit it for a job only the model can judge.

Reads ``~/.ccbot/jobs.json`` every tick so edits land without a restart. A job
fires once its time has passed today and it has not already run; the last fire
date lives in ``~/.ccbot/jobs-fired.json`` so a restart neither double-fires
nor skips. A job whose time passed while the bot was down fires on the next
tick — late beats never for a morning report.

A job with no entry in ``jobs-fired.json`` at all has never been seen, so it is
new rather than late. It records today's date without running and starts at its
next slot. Without this, adding a job whose time is earlier in the day fires it
the moment the file lands.

Job shape::

    {"name": "email-triage", "at": "08:00", "days": [0,1,2,3,4,5,6],
     "machines": ["headless"], "cwd": "/Users/zellwk/projects",
     "topic": "Email triage", "prompt": "triage emails",
     "when": "node check-due.js"}

``days`` uses Python weekdays, Monday 0. Omit it for every day.

``machines`` lists the roles this job runs on, matched against ``~/.claude/.machine``
— ``headless`` on the Mac mini, ``main`` on the laptop. Omit it to run everywhere.

``next`` names a job, or a list of jobs, to start once this job's report has been
posted. That delivery is the chain's one trigger. Closing a topic, ``/done``, a
reap or a killed window never starts a follower, so tidying old topics sets
nothing off. A list runs its followers one after another, each in its own topic.
Each follower needs no ``at`` and should carry ``"enabled": false`` so the clock
never starts it on its own — chaining calls it directly. Chains can run any
depth: triage → support → action.

A follower runs at most once a day. Its date is claimed in ``jobs-fired.json``
before it starts, so a second trigger the same day finds it taken. A pass that
fails or finds nothing posts no report, and starts no follower.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

from telegram import Bot

from .config import config
from .handlers.cleanup import end_session
from .handlers.message_sender import DeadThread, safe_send
from .session import session_manager
from .telegram_sender import split_message
from .tmux_manager import tmux_manager
from .utils import atomic_write_json, ccbot_dir

logger = logging.getLogger(__name__)

TICK_SECONDS = 30.0

# Long enough for Claude Code's SessionStart hook to register the window.
SESSION_MAP_TIMEOUT = 5.0

# A morning report can take a while to assemble; past this the pass is stuck.
HEADLESS_TIMEOUT_SECONDS = 1200.0

# `has_report` decides whether a topic opens at all, and its description is the only
# place a job learns which way to lean, so the bias is written into the schema.
REPORT_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "has_report": {
                "type": "boolean",
                "description": (
                    "True only when there is something Zell has to see. This job runs "
                    "again tomorrow and anything still outstanding comes back then, so "
                    "leave it false when in doubt — staying quiet costs a day, while a "
                    "topic opened over nothing costs him a notification every morning."
                ),
            },
            "report": {
                "type": "string",
                "description": "The message to post, written as it should read on a phone.",
            },
        },
        "required": ["has_report"],
    }
)


class JobReport(NamedTuple):
    """A headless pass's report, and the session a reply can carry on in.

    ``session_id`` is empty when the pass left nothing resumable — a timeout or a
    crash — and the topic then opens on a fresh session. ``failed`` marks the
    notice posted in place of a report, which starts no follower.
    """

    text: str
    session_id: str
    failed: bool = False


async def scheduled_jobs_loop(bot: Bot) -> None:
    """Runs each due job in a topic of its own, once a day."""
    while True:
        await asyncio.sleep(TICK_SECONDS)
        try:
            fired = _load_fired_dates()
            now = datetime.now()
            for job in _due_jobs(_load_jobs(), now, fired):
                first_seen = job["name"] not in fired
                # Recorded before the job starts, not after. A headless pass takes
                # minutes and the next tick is 30 seconds away, so a date written
                # afterwards leaves a gap that fires the same job again.
                fired[job["name"]] = now.date().isoformat()
                atomic_write_json(_fired_path(), fired)
                if first_seen:
                    logger.info(
                        "Job %s seen for the first time past its time; "
                        "first run is its next slot",
                        job["name"],
                    )
                    continue
                await _open_session_for_job(bot, job)
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
    """Works the job headless, and posts in its topic only when it has something."""
    if not await _is_gate_open(job):
        logger.info("Job %s: gate closed, nothing opened", job["name"])
        return

    report = await _get_job_report(job)
    if report is None:
        logger.info("Job %s: nothing to report, nothing opened", job["name"])
        return

    user_id = next(iter(config.allowed_users))
    chat_id = config.forum_chat_id or session_manager.resolve_chat_id(user_id)
    thread_id = _job_topics().get(job["name"])
    if thread_id is None:
        thread_id = await _new_topic(bot, job, user_id, chat_id)
    # Last run's session ends only now, when the next run replaces it. A reply
    # sent between runs still reaches it.
    await end_session(bot, thread_id)

    # Posted before the window exists. The report is the thing worth delivering, and
    # a window that fails to open must not take the findings down with it.
    try:
        await _post_report(bot, chat_id, thread_id, report)
    except DeadThread:
        logger.info(
            "Job %s: topic %d is gone, making a new one", job["name"], thread_id
        )
        thread_id = await _new_topic(bot, job, user_id, chat_id)
        await _post_report(bot, chat_id, thread_id, report)

    # The report has landed, so this run is done as far as its followers care.
    # A task of their own, because each works headless for minutes and the
    # clock must keep ticking for every other job meanwhile.
    if not report.failed and job.get("next"):
        asyncio.create_task(_start_followers(bot, job))

    # The work is already done, so the window carries on that same session rather
    # than starting cold and being asked the same question twice.
    created, detail, window_name, window_id = await tmux_manager.create_window(
        job["cwd"], resume_session_id=report.session_id or None
    )
    if not created:
        logger.error("Job %s: window failed: %s", job["name"], detail)
        await safe_send(
            bot, chat_id, f"❌ {job['name']}: {detail}", message_thread_id=thread_id
        )
        return

    await session_manager.wait_for_session_map_entry(
        window_id, timeout=SESSION_MAP_TIMEOUT
    )
    session_manager.bind_thread(user_id, thread_id, window_id, window_name=window_name)
    # A job's topic is named on purpose. Setting auto_named here is what
    # topic_namer.maybe_autoname() checks before it renames anything, so the
    # local model leaves these rooms alone.
    session_manager.mark_auto_named(window_id, True)

    logger.info(
        "Job %s: reported in topic %d (window %s)", job["name"], thread_id, window_id
    )


async def _is_gate_open(job: dict[str, Any]) -> bool:
    """Runs the job's ``when`` command. True when it exits 0, or names no gate."""
    command = job.get("when")
    if not command:
        return True

    proc = await asyncio.create_subprocess_shell(
        command,
        cwd=job["cwd"],
        env=_job_env(),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    logger.info(
        "Job %s: gate exit %s (%s)",
        job["name"],
        proc.returncode,
        stdout.decode().strip() or stderr.decode().strip(),
    )
    return proc.returncode == 0


async def _get_job_report(job: dict[str, Any]) -> JobReport | None:
    """Works the job headless. None when it found nothing worth opening a topic for."""
    proc = await asyncio.create_subprocess_exec(
        config.claude_command,
        "-p",
        job["prompt"],
        "--json-schema",
        REPORT_SCHEMA,
        "--output-format",
        "json",
        cwd=job["cwd"],
        env=_job_env(),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), HEADLESS_TIMEOUT_SECONDS
        )
    except TimeoutError:
        proc.kill()
        logger.error("Job %s: headless pass timed out", job["name"])
        return JobReport(f"❌ {job['name']}: timed out before it reported.", "", True)

    if proc.returncode != 0:
        logger.error(
            "Job %s: headless pass exited %s: %s",
            job["name"],
            proc.returncode,
            stderr.decode().strip()[-500:],
        )
        return JobReport(f"❌ {job['name']}: the headless pass failed.", "", True)

    result = json.loads(stdout)
    logger.info(
        "Job %s: headless pass cost $%.4f", job["name"], result.get("total_cost_usd", 0)
    )

    output = result.get("structured_output") or {}
    if not output.get("has_report"):
        return None
    return JobReport(output.get("report", ""), result.get("session_id", ""))


def _job_env() -> dict[str, str]:
    """Environment for a job's subprocesses, with the PATH launchd leaves out."""
    home = str(Path.home())
    env = dict(os.environ)
    env["PATH"] = (
        f"{home}/.local/bin:{home}/n/bin:/opt/homebrew/bin:{env.get('PATH', '')}"
    )
    return env


async def _new_topic(bot: Bot, job: dict[str, Any], user_id: int, chat_id: int) -> int:
    """Makes the job's topic and stores it as the one its runs reuse."""
    topic = await bot.create_forum_topic(chat_id=chat_id, name=job["topic"])
    thread_id = topic.message_thread_id
    # Nothing has arrived from this topic yet, so record where it lives now.
    # Every outbound call for the thread resolves through this mapping.
    session_manager.set_group_chat_id(user_id, thread_id, chat_id)
    _remember_job_topic(job["name"], thread_id)
    return thread_id


async def _post_report(
    bot: Bot, chat_id: int, thread_id: int, report: JobReport
) -> None:
    """Posts the report in chunks. Raises DeadThread when the topic is gone.

    Split with room for MarkdownV2 escaping, as response_builder does: sent
    whole, a report past Telegram's limit fails outright and leaves the topic
    empty.
    """
    for chunk in split_message(report.text, max_length=3000):
        await safe_send(bot, chat_id, chunk, message_thread_id=thread_id)


async def _start_followers(bot: Bot, job: dict[str, Any]) -> None:
    """Runs the jobs chained after ``job``, in order, each at most once a day.

    A failure here must not reach the scheduler's tick.
    """
    nxt = job["next"]
    followers = [nxt] if isinstance(nxt, str) else list(nxt)
    jobs = _load_jobs()
    try:
        for name in followers:
            follower = next((j for j in jobs if j["name"] == name), None)
            if not follower:
                logger.error("Job %s: next job %r not found", job["name"], name)
                continue
            if not _claim_today(name):
                logger.warning(
                    "Job %s: next job %s already ran today, not starting it again",
                    job["name"],
                    name,
                )
                continue
            logger.info("Job %s: chaining to %s", job["name"], name)
            await _open_session_for_job(bot, follower)
    except Exception:
        logger.exception("Job %s: chaining failed", job["name"])


def _claim_today(name: str) -> bool:
    """Records today as ``name``'s run date. False when today is already taken.

    Read and written with no await between, so two triggers on the one event
    loop cannot both claim the same day.
    """
    fired = _load_fired_dates()
    today = datetime.now().date().isoformat()
    if fired.get(name) == today:
        return False
    fired[name] = today
    atomic_write_json(_fired_path(), fired)
    return True


def _job_topics() -> dict[str, int]:
    path = _job_topics_path()
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def forget_closed_topic(thread_id: int) -> None:
    """Drops the job whose topic this was, so its next run makes a new one.

    Called when a topic closes. A thread that belongs to no job changes nothing.
    """
    topics = _job_topics()
    kept = {name: tid for name, tid in topics.items() if tid != thread_id}
    if kept != topics:
        atomic_write_json(_job_topics_path(), kept)


def _remember_job_topic(name: str, thread_id: int) -> None:
    topics = _job_topics()
    topics[name] = thread_id
    atomic_write_json(_job_topics_path(), topics)


def _job_topics_path() -> Path:
    return ccbot_dir() / "jobs-topics.json"


def _fired_path() -> Path:
    return ccbot_dir() / "jobs-fired.json"
