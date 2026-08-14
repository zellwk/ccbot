"""Auto-name a Telegram topic once the conversation has a direction.

A topic starts life as a placeholder ("New chat") and stays that way. After a
few exchanges the transcript already says what the thread is about, so a local
Ollama model writes a short title and the bot renames the topic and its tmux
window to match. No Claude tokens, no network.

Naming happens once per window. The rename sets ``WindowState.auto_named``, and
a manual rename in Telegram sets it too, so a title Zell chose is never
overwritten. Renaming a topic back to a placeholder clears the flag and re-arms
auto-naming for that thread.

Key function: maybe_autoname().
"""

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any

import httpx
from telegram import Bot

from .config import config
from .session import session_manager
from .token_usage import find_transcript

logger = logging.getLogger(__name__)

# (user_id, thread_id) pairs with a naming attempt in flight
_in_flight: set[tuple[int, int]] = set()

_SYSTEM_BLOCK_RE = re.compile(r"<(system-reminder|command-[a-z-]+|local-command-[a-z-]+)>.*?</\1>", re.S)
_THINK_RE = re.compile(r"<think>.*?</think>", re.S)
_TAG_RE = re.compile(r"</?[a-z-]+>")

_PROMPT = """You are titling a chat thread so it can be found again in a list.

Read the excerpt and write the title: 3 to 6 words naming the subject being worked on or talked about. Be specific — use the concrete things mentioned. Never describe the participants or the act of chatting. No colons or prefixes, no quotes, no trailing punctuation, and none of these words: chat, session, conversation, discussion, user, assistant, AI.

---
{digest}
---

Title:"""


def is_placeholder(name: str) -> bool:
    """True when a topic name carries no meaning yet (e.g. 'New chat')."""
    return name.strip().lower() in config.autoname_placeholders


def _entry_text(entry: dict[str, Any]) -> str:
    """Pull plain text out of a transcript entry, or '' when it has none."""
    content = (entry.get("message") or {}).get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = [
        b.get("text", "")
        for b in content
        if isinstance(b, dict) and b.get("type") == "text"
    ]
    return "\n".join(p for p in parts if p)


def _clean(text: str) -> str:
    text = _SYSTEM_BLOCK_RE.sub("", text)
    text = _THINK_RE.sub("", text)
    text = _TAG_RE.sub("", text)
    return " ".join(text.split())


def read_digest(path: Path, max_turns: int) -> tuple[str, int]:
    """Read a transcript into a prompt digest. Returns (digest, user_turn_count).

    Counts only real user turns — tool results and sidechain (subagent) entries
    are skipped, as are the hook and slash-command envelopes Claude Code writes
    into the user stream.
    """
    lines: list[str] = []
    turns = 0
    pending_user = False

    with path.open(errors="replace") as f:
        for raw in f:
            try:
                entry = json.loads(raw)
            except ValueError:
                continue
            if entry.get("isSidechain") or entry.get("isMeta"):
                continue

            kind = entry.get("type")
            if kind == "user":
                text = _clean(_entry_text(entry))
                if not text:
                    continue
                turns += 1
                pending_user = True
                if turns <= max_turns:
                    lines.append(f"User: {text[:400]}")
            elif kind == "assistant" and pending_user and turns <= max_turns:
                text = _clean(_entry_text(entry))
                if not text:
                    continue
                lines.append(f"Assistant: {text[:400]}")
                pending_user = False

    return "\n".join(lines)[:3000], turns


def sanitize(raw: str) -> str | None:
    """Reduce a model's reply to a usable topic name, or None if unusable."""
    text = _THINK_RE.sub("", raw or "")
    for line in text.splitlines():
        line = line.strip().strip("*_`#").strip()
        line = re.sub(r"^(title|answer)\s*[:\-]\s*", "", line, flags=re.I)
        line = line.strip().strip('"“”‘’\'').strip()
        line = line.rstrip(".!,;:").strip()
        line = " ".join(line.split())
        if not line:
            continue
        if len(line) > 60:
            line = line[:60].rsplit(" ", 1)[0]
        return line or None
    return None


async def generate_title(digest: str) -> str | None:
    """Ask the local Ollama model for a title. Returns None on any failure."""
    payload = {
        "model": config.autoname_model,
        "prompt": _PROMPT.format(digest=digest),
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 32},
    }
    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(
                f"{config.autoname_ollama_url}/api/generate", json=payload
            )
            resp.raise_for_status()
            return sanitize(resp.json().get("response", ""))
    except Exception as e:
        logger.warning("Auto-name: Ollama call failed (%s): %s", type(e).__name__, e)
        return None


async def maybe_autoname(
    bot: Bot, user_id: int, thread_id: int | None, window_id: str
) -> None:
    """Rename a still-unnamed topic once its transcript has enough turns.

    Safe to fire on every message — it returns early unless this window is
    unnamed and the turn threshold has just been crossed.
    """
    if thread_id is None or not config.autoname_after_turns:
        return

    state = session_manager.get_window_state(window_id)
    if state.auto_named or not state.session_id:
        return

    key = (user_id, thread_id)
    if key in _in_flight:
        return
    _in_flight.add(key)
    try:
        path = await asyncio.to_thread(find_transcript, state.session_id)
        if not path:
            return
        digest, turns = await asyncio.to_thread(
            read_digest, path, config.autoname_after_turns
        )
        if turns < config.autoname_after_turns or not digest:
            return

        title = await generate_title(digest)
        if not title:
            return

        # Re-check: a manual rename may have landed while the model was thinking.
        if session_manager.get_window_state(window_id).auto_named:
            return

        chat_id = session_manager.resolve_chat_id(user_id, thread_id)
        await bot.edit_forum_topic(
            chat_id=chat_id, message_thread_id=thread_id, name=title
        )
        session_manager.mark_auto_named(window_id, True)
        # Keep the tmux window in step. The topic_edited service message does
        # this too, but only if Telegram echoes the bot's own edit back.
        from .tmux_manager import tmux_manager

        await tmux_manager.rename_window(window_id, title)
        session_manager.update_display_name(window_id, title)
        logger.info(
            "Auto-named topic: '%s' (window=%s, user=%d, thread=%d, turns=%d)",
            title,
            window_id,
            user_id,
            thread_id,
            turns,
        )
    except Exception as e:
        logger.warning("Auto-name failed (window=%s): %s", window_id, e)
    finally:
        _in_flight.discard(key)
