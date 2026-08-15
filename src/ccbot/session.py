"""Claude Code session management — the core state hub.

One record per tmux window, keyed by window_id, holds everything known about
that window: the Claude session it runs, its working directory, its display
name, and the Telegram topic bound to it (1 topic = 1 window).

Responsibilities:
  - Persist/load state to ~/.ccbot/state.json.
  - Sync window↔session bindings from session_map.json (written by hook).
  - Resolve window IDs to ClaudeSession objects (JSONL file reading).
  - Manage thread↔window bindings for Telegram topic routing.
  - Send keystrokes to tmux windows and retrieve message history.
  - Re-resolve stale window IDs on startup (tmux server restart recovery).

Key class: SessionManager (singleton instantiated as `session_manager`).
Key methods for thread binding access:
  - resolve_window_for_thread: Get window_id for a user's thread
  - iter_thread_bindings: Generator for iterating all (user_id, thread_id, window_id)
  - find_users_for_session: Find all users bound to a session_id
"""

import asyncio
import fcntl
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Callable, Iterator
from typing import Any

import aiofiles

from .config import config
from .terminal_parser import is_blocking_dialog, is_prompt_ready
from .tmux_manager import tmux_manager
from .transcript_parser import TranscriptParser
from .utils import atomic_write_json

logger = logging.getLogger(__name__)


@dataclass
class WindowState:
    """Persistent state for a tmux window.

    One record per tmux window holds everything known about it. Nothing about
    a window is stored anywhere else in this file.

    Attributes:
        session_id: Associated Claude session ID (empty if not yet detected)
        cwd: Working directory for direct file path construction
        window_name: Display name — mirrors the tmux window name
        auto_named: Topic has a real name — skip auto-naming from now on
        user_id: Telegram user whose topic is bound here (None if unbound)
        thread_id: Telegram topic thread bound here (None if unbound)
    """

    session_id: str = ""
    cwd: str = ""
    window_name: str = ""
    auto_named: bool = False
    user_id: int | None = None
    thread_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "session_id": self.session_id,
            "cwd": self.cwd,
        }
        if self.window_name:
            d["window_name"] = self.window_name
        if self.auto_named:
            d["auto_named"] = True
        if self.user_id is not None:
            d["user_id"] = self.user_id
        if self.thread_id is not None:
            d["thread_id"] = self.thread_id
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WindowState":
        uid = data.get("user_id")
        tid = data.get("thread_id")
        return cls(
            session_id=data.get("session_id", ""),
            cwd=data.get("cwd", ""),
            window_name=data.get("window_name", ""),
            auto_named=bool(data.get("auto_named", False)),
            user_id=int(uid) if uid is not None else None,
            thread_id=int(tid) if tid is not None else None,
        )


@dataclass
class ClaudeSession:
    """Information about a Claude Code session."""

    session_id: str
    summary: str
    message_count: int
    file_path: str


@dataclass
class SessionManager:
    """Manages session state for Claude Code.

    All internal keys use window_id (e.g. '@0', '@12') for uniqueness.
    Display names (window_name) are stored separately for UI presentation.

    window_states: window_id -> WindowState. The single record for a window —
    its session, cwd, display name, and the topic bound to it. Look up
    anything about a window here; there is no second place to check.

    group_chat_ids: "user_id:thread_id" -> group chat_id (for supergroup
    routing). Stays separate because it is keyed by thread, is written before
    a window exists, and must outlive unbinding.
    """

    window_states: dict[str, WindowState] = field(default_factory=dict)
    # "user_id:thread_id" -> group chat_id (for supergroup forum topic routing)
    # IMPORTANT: This mapping is essential for supergroup/forum topic support.
    # Telegram Bot API requires group chat_id (negative number like -100xxx)
    # as the chat_id parameter when sending messages to forum topics.
    # Using user_id as chat_id will fail with "Message thread not found".
    # See: https://core.telegram.org/bots/api#sendmessage
    # History: originally added in 5afc111, erroneously removed in 26cb81f,
    # restored in PR #23.
    group_chat_ids: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._load_state()

    def _save_state(self) -> None:
        state: dict[str, Any] = {
            "version": 2,
            "window_states": {k: v.to_dict() for k, v in self.window_states.items()},
            "group_chat_ids": self.group_chat_ids,
        }
        atomic_write_json(config.state_file, state)
        logger.debug("State saved to %s", config.state_file)

    def _is_window_id(self, key: str) -> bool:
        """Check if a key looks like a tmux window ID (e.g. '@0', '@12')."""
        return key.startswith("@") and len(key) > 1 and key[1:].isdigit()

    @staticmethod
    def _fold_v1(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """Fold the pre-v2 state layout into one record per window.

        v1 spread a window across four maps: window_states, thread_bindings,
        window_display_names and user_window_offsets. The offsets were never
        read — monitor_state.json tracks the byte offsets that actually
        matter — so they are dropped rather than carried over.
        """
        records: dict[str, dict[str, Any]] = {
            k: dict(v) for k, v in state.get("window_states", {}).items()
        }
        for wid, name in state.get("window_display_names", {}).items():
            records.setdefault(wid, {})["window_name"] = name
        for uid, bindings in state.get("thread_bindings", {}).items():
            for tid, wid in bindings.items():
                rec = records.setdefault(wid, {})
                rec["user_id"] = int(uid)
                rec["thread_id"] = int(tid)
        return records

    def _load_state(self) -> None:
        """Load state synchronously during initialization.

        Reads either layout: v2 stores one record per window, v1 spread the
        same fields across four maps. Old-format keys (window_name rather
        than window_id) are re-resolved against live tmux on startup.
        """
        if config.state_file.exists():
            try:
                state = json.loads(config.state_file.read_text())
                if state.get("version", 1) < 2:
                    logger.info("Migrating state to the single-record layout")
                    raw = self._fold_v1(state)
                else:
                    raw = state.get("window_states", {})
                self.window_states = {
                    k: WindowState.from_dict(v) for k, v in raw.items()
                }
                self.group_chat_ids = {
                    k: int(v) for k, v in state.get("group_chat_ids", {}).items()
                }

                if any(not self._is_window_id(k) for k in self.window_states):
                    logger.info(
                        "Detected old-format state (window_name keys), "
                        "will re-resolve on startup"
                    )

            except (json.JSONDecodeError, ValueError) as e:
                logger.warning("Failed to load state: %s", e)
                self.window_states = {}
                self.group_chat_ids = {}
                pass

    async def resolve_stale_ids(self) -> None:
        """Re-resolve persisted window IDs against live tmux windows.

        Called on startup. Handles two cases:
        1. Old-format migration: window_name keys → window_id keys
        2. Stale IDs: window_id no longer exists but display name matches a live window

        Builds {window_name: window_id} from live windows, then remaps or drops entries.
        """
        windows = await tmux_manager.list_windows()
        live_by_name: dict[str, str] = {}  # window_name -> window_id
        live_ids: set[str] = set()
        for w in windows:
            live_by_name[w.window_name] = w.window_id
            live_ids.add(w.window_id)

        changed = False
        resolved: dict[str, WindowState] = {}
        # Records already on a live id are authoritative and are taken first.
        # A stale record may share its name with a live window (tmux reuses
        # "projects-2" once the original is gone), and must never overwrite
        # the live one — that would hand the live window a dead session and
        # drop the topic bound to it.
        ordered = sorted(
            self.window_states.items(), key=lambda kv: kv[0] not in live_ids
        )
        for key, ws in ordered:
            if self._is_window_id(key):
                if key in live_ids:
                    resolved[key] = ws
                    continue
                # Stale ID — the tmux server restarted and handed out new ones.
                # The display name is what survives, so re-resolve through it.
                name = ws.window_name or key
                new_id = live_by_name.get(name)
                if new_id in resolved:
                    logger.info(
                        "Dropping stale window_state %s: %s already holds '%s'",
                        key,
                        new_id,
                        name,
                    )
                elif new_id:
                    logger.info(
                        "Re-resolved stale window_id %s -> %s (name=%s)",
                        key,
                        new_id,
                        name,
                    )
                    resolved[new_id] = ws
                else:
                    logger.info("Dropping stale window_state: %s (name=%s)", key, name)
            else:
                # Old format: the key was the window name.
                new_id = live_by_name.get(key)
                if new_id and new_id not in resolved:
                    logger.info("Migrating window_state key %s -> %s", key, new_id)
                    ws.window_name = key
                    resolved[new_id] = ws
                else:
                    logger.info(
                        "Dropping old-format window_state: %s (no live window)", key
                    )
            changed = True

        self.window_states = resolved
        if changed:
            self._save_state()
            logger.info("Startup re-resolution complete")

        # Clean up session_map.json: stale window IDs, migrate old-format keys
        await self._cleanup_stale_session_map_entries(live_ids)
        await self._migrate_old_format_session_map_keys(live_by_name)

    def _migrate_old_format_map(
        self, session_map: dict[str, dict], live_by_name: dict[str, str]
    ) -> bool:
        """Migrate old-format session_map keys to the @window_id form in place.

        Old hook versions keyed session_map by window_name (e.g. "ccbot:ccmux")
        instead of window_id ("ccbot:@4"). Such keys are invisible to the
        window_id-based delivery path (load_session_map skips them), which
        silently drops inbound messages. This resolves each old-format key's
        window_name against live tmux windows and rewrites it to the @window_id
        form, preserving session_id/cwd and backfilling window_name. Keys with
        no matching live window are dropped as orphans; if the @window_id key
        already exists it wins and the old-format one is discarded.

        Mutates session_map in place. Returns True if anything changed.
        """
        prefix = f"{config.tmux_session_name}:"
        old_keys = [
            key
            for key in session_map
            if key.startswith(prefix) and not self._is_window_id(key[len(prefix) :])
        ]
        changed = False
        for key in old_keys:
            window_name = key[len(prefix) :]
            info = session_map.pop(key)
            changed = True
            new_id = live_by_name.get(window_name)
            if not new_id:
                logger.info("Dropping orphan old-format session_map key: %s", key)
                continue
            new_key = f"{prefix}{new_id}"
            if new_key in session_map:
                logger.info(
                    "Discarding old-format session_map key %s (superseded by %s)",
                    key,
                    new_key,
                )
                continue
            info.setdefault("window_name", window_name)
            session_map[new_key] = info
            logger.info("Migrated old-format session_map key %s -> %s", key, new_key)
        return changed

    def _mutate_session_map_locked(
        self, mutate: Callable[[dict[str, dict]], bool]
    ) -> bool:
        """Read-modify-write session_map.json under the same flock the hook uses.

        The SessionStart hook serializes its writes via session_map.lock;
        any bot-side read-modify-write MUST take the same lock or it can
        overwrite a concurrent hook write (lost update). Synchronous —
        call via asyncio.to_thread from async code.

        Returns True if `mutate` reported changes and the file was rewritten.
        """
        map_file = config.session_map_file
        lock_path = map_file.with_suffix(".lock")
        try:
            with open(lock_path, "w") as lock_f:
                fcntl.flock(lock_f, fcntl.LOCK_EX)
                try:
                    session_map: dict[str, dict] = {}
                    if map_file.exists():
                        try:
                            session_map = json.loads(map_file.read_text())
                        except (json.JSONDecodeError, OSError):
                            logger.warning(
                                "Unreadable session_map.json, skipping mutation"
                            )
                            return False
                    if not mutate(session_map):
                        return False
                    atomic_write_json(map_file, session_map)
                    return True
                finally:
                    fcntl.flock(lock_f, fcntl.LOCK_UN)
        except OSError as e:
            logger.error("Failed to update session_map.json: %s", e)
            return False

    async def override_session_map_entry(
        self, window_id: str, session_id: str, cwd: str = "", window_name: str = ""
    ) -> None:
        """Force a window's session_map entry to a specific session_id.

        Used after `--resume`: session_map drives both the monitor's watch
        list and load_session_map()'s sync into window_states, so overriding
        window_state alone would be reverted on the next poll cycle. Creates
        the entry if missing (hook timed out); no-op if already consistent.
        """
        key = f"{config.tmux_session_name}:{window_id}"

        def mutate(session_map: dict[str, dict]) -> bool:
            info = session_map.get(key)
            if info is None:
                session_map[key] = {
                    "session_id": session_id,
                    "cwd": cwd,
                    "window_name": window_name,
                }
                return True
            if info.get("session_id") == session_id:
                return False
            info["session_id"] = session_id
            return True

        if await asyncio.to_thread(self._mutate_session_map_locked, mutate):
            logger.info("session_map override: %s -> session_id=%s", key, session_id)

    async def _migrate_old_format_session_map_keys(
        self, live_by_name: dict[str, str]
    ) -> None:
        """Migrate old-format keys in session_map.json to @window_id form (startup)."""
        if not config.session_map_file.exists():
            return
        changed = await asyncio.to_thread(
            self._mutate_session_map_locked,
            lambda session_map: self._migrate_old_format_map(session_map, live_by_name),
        )
        if changed:
            logger.info("Migrated old-format session_map keys to @window_id form")

    async def _cleanup_stale_session_map_entries(self, live_ids: set[str]) -> None:
        """Remove entries for tmux windows that no longer exist.

        When windows are closed externally (outside ccbot), session_map.json
        retains orphan references. This cleanup removes entries whose window_id
        is not in the current set of live tmux windows.
        """
        if not config.session_map_file.exists():
            return

        prefix = f"{config.tmux_session_name}:"

        def mutate(session_map: dict[str, dict]) -> bool:
            stale_keys = [
                key
                for key in session_map
                if key.startswith(prefix)
                and self._is_window_id(key[len(prefix) :])
                and key[len(prefix) :] not in live_ids
            ]
            for key in stale_keys:
                del session_map[key]
                logger.info("Removed stale session_map entry: %s", key)
            return bool(stale_keys)

        if await asyncio.to_thread(self._mutate_session_map_locked, mutate):
            logger.info(
                "Cleaned up stale session_map entries (windows no longer in tmux)"
            )

    # --- Display name management ---

    def get_display_name(self, window_id: str) -> str:
        """Get display name for a window_id, fallback to window_id itself."""
        ws = self.window_states.get(window_id)
        return ws.window_name if ws and ws.window_name else window_id

    def update_display_name(self, window_id: str, new_name: str) -> None:
        """Update the display name for a window and persist state."""
        self.get_window_state(window_id).window_name = new_name
        self._save_state()
        logger.info("Updated display name: window_id %s -> '%s'", window_id, new_name)

    # --- Group chat ID management (supergroup forum topic routing) ---

    def set_group_chat_id(
        self, user_id: int, thread_id: int | None, chat_id: int
    ) -> None:
        """Store the group chat_id for a user+thread combination.

        In supergroups with forum topics, messages must be sent to the group's
        chat_id (negative number like -100xxx) rather than the user's personal ID.
        Telegram's Bot API rejects message_thread_id when chat_id is a private
        user ID — the thread only exists within the group context.

        DO NOT REMOVE this method or the group_chat_ids mapping.
        Without it, all outbound messages in forum topics fail with
        "Message thread not found". See commit history: 5afc111 → 26cb81f → PR #23.
        """
        tid = thread_id or 0
        key = f"{user_id}:{tid}"
        if self.group_chat_ids.get(key) != chat_id:
            self.group_chat_ids[key] = chat_id
            self._save_state()
            logger.debug(
                "Stored group chat_id: user=%d, thread=%s, chat_id=%d",
                user_id,
                thread_id,
                chat_id,
            )

    def resolve_chat_id(self, user_id: int, thread_id: int | None = None) -> int:
        """Resolve the correct chat_id for sending messages.

        Returns the stored group chat_id when a thread_id is present and a
        mapping exists, otherwise falls back to user_id (for private chats).

        Every outbound Telegram API call (send_message, edit_message_text,
        delete_message, send_chat_action, edit_forum_topic, etc.) MUST use
        this method instead of raw user_id. Using user_id directly breaks
        supergroup forum topic routing.
        """
        if thread_id is not None:
            key = f"{user_id}:{thread_id}"
            group_id = self.group_chat_ids.get(key)
            if group_id is not None:
                return group_id
        return user_id

    async def wait_for_session_map_entry(
        self, window_id: str, timeout: float = 5.0, interval: float = 0.5
    ) -> bool:
        """Poll session_map.json until an entry for window_id appears.

        Returns True if the entry was found within timeout, False otherwise.
        """
        logger.debug(
            "Waiting for session_map entry: window_id=%s, timeout=%.1f",
            window_id,
            timeout,
        )
        key = f"{config.tmux_session_name}:{window_id}"
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            try:
                if config.session_map_file.exists():
                    async with aiofiles.open(config.session_map_file, "r") as f:
                        content = await f.read()
                    session_map = json.loads(content)
                    info = session_map.get(key, {})
                    if info.get("session_id"):
                        # Found — load into window_states immediately
                        logger.debug(
                            "session_map entry found for window_id %s", window_id
                        )
                        await self.load_session_map()
                        return True
            except (json.JSONDecodeError, OSError):
                pass
            await asyncio.sleep(interval)
        logger.warning(
            "Timed out waiting for session_map entry: window_id=%s", window_id
        )
        return False

    async def load_session_map(self) -> None:
        """Read session_map.json and update window_states with new session associations.

        Keys in session_map are formatted as "tmux_session:window_id" (e.g. "ccbot:@12").
        Only entries matching our tmux_session_name are processed.
        Also cleans up window_states entries not in current session_map.
        Updates window_display_names from the "window_name" field in values.
        """
        if not config.session_map_file.exists():
            return
        try:
            async with aiofiles.open(config.session_map_file, "r") as f:
                content = await f.read()
            session_map = json.loads(content)
        except (json.JSONDecodeError, OSError):
            return

        prefix = f"{config.tmux_session_name}:"

        # Self-heal old-format keys (session:window_name) that an outdated hook
        # may write at runtime: resolve them against live windows and rewrite to
        # @window_id in place, so the delivery loop below can see them. Only
        # lists tmux windows when such keys are actually present (zero cost in
        # steady state). Mirrors the tolerance already in _load_current_session_map.
        if any(
            k.startswith(prefix) and not self._is_window_id(k[len(prefix) :])
            for k in session_map
        ):
            windows = await tmux_manager.list_windows()
            live_by_name = {w.window_name: w.window_id for w in windows}
            if self._migrate_old_format_map(session_map, live_by_name):
                atomic_write_json(config.session_map_file, session_map)
                logger.info("Migrated old-format session_map keys during load")

        valid_wids: set[str] = set()
        changed = False

        for key, info in session_map.items():
            # Only process entries for our tmux session
            if not key.startswith(prefix):
                continue
            window_id = key[len(prefix) :]
            if not self._is_window_id(window_id):
                continue
            valid_wids.add(window_id)
            new_sid = info.get("session_id", "")
            new_cwd = info.get("cwd", "")
            new_wname = info.get("window_name", "")
            if not new_sid:
                continue
            state = self.get_window_state(window_id)
            if state.session_id != new_sid or state.cwd != new_cwd:
                logger.info(
                    "Session map: window_id %s updated sid=%s, cwd=%s",
                    window_id,
                    new_sid,
                    new_cwd,
                )
                state.session_id = new_sid
                state.cwd = new_cwd
                changed = True
            # Update display name. session_map records the name the window had
            # when Claude started — "projects-2" and the like. Once a topic has
            # earned a real name, that stale value must not overwrite it.
            if new_wname and not state.auto_named and state.window_name != new_wname:
                state.window_name = new_wname
                changed = True

        # Clean up window_states entries not in current session_map.
        stale_wids = [w for w in self.window_states if w and w not in valid_wids]
        for wid in stale_wids:
            logger.info("Removing stale window_state: %s", wid)
            del self.window_states[wid]
            changed = True

        if changed:
            self._save_state()

    # --- Window state management ---

    def get_window_state(self, window_id: str) -> WindowState:
        """Get or create window state."""
        if window_id not in self.window_states:
            self.window_states[window_id] = WindowState()
        return self.window_states[window_id]

    def mark_auto_named(self, window_id: str, value: bool) -> None:
        """Set whether this window's topic already carries a real name."""
        state = self.get_window_state(window_id)
        if state.auto_named == value:
            return
        state.auto_named = value
        self._save_state()

    def clear_window_session(self, window_id: str) -> None:
        """Clear session association for a window (e.g., after /clear command)."""
        state = self.get_window_state(window_id)
        state.session_id = ""
        self._save_state()
        logger.info("Cleared session for window_id %s", window_id)

    @staticmethod
    def _encode_cwd(cwd: str) -> str:
        """Encode a cwd path to match Claude Code's project directory naming.

        Replaces all non-alphanumeric characters (except dash) with dashes.
        E.g. /home/user_name/Code/project -> -home-user-name-Code-project
        """
        return re.sub(r"[^a-zA-Z0-9-]", "-", cwd)

    def _build_session_file_path(self, session_id: str, cwd: str) -> Path | None:
        """Build the direct file path for a session from session_id and cwd."""
        if not session_id or not cwd:
            return None
        encoded_cwd = self._encode_cwd(cwd)
        return config.claude_projects_path / encoded_cwd / f"{session_id}.jsonl"

    async def _get_session_direct(
        self, session_id: str, cwd: str
    ) -> ClaudeSession | None:
        """Get a ClaudeSession directly from session_id and cwd (no scanning)."""
        file_path = self._build_session_file_path(session_id, cwd)

        # Fallback: glob search if direct path doesn't exist
        if not file_path or not file_path.exists():
            pattern = f"*/{session_id}.jsonl"
            matches = list(config.claude_projects_path.glob(pattern))
            if matches:
                file_path = matches[0]
                logger.debug("Found session via glob: %s", file_path)
            else:
                return None

        # Single pass: read file once, extract summary + count messages
        summary = ""
        last_user_msg = ""
        message_count = 0
        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                async for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    message_count += 1
                    try:
                        data = json.loads(line)
                        # Check for summary
                        if data.get("type") == "summary":
                            s = data.get("summary", "")
                            if s:
                                summary = s
                        # Track last user message as fallback
                        elif TranscriptParser.is_user_message(data):
                            parsed = TranscriptParser.parse_message(data)
                            if parsed and parsed.text.strip():
                                last_user_msg = parsed.text.strip()
                    except json.JSONDecodeError:
                        continue
        except OSError:
            return None

        if not summary:
            # LOCAL PATCH: the fallback is the last user message, which on a
            # Telegram-bridged session is wrapped in <channel …> markup.
            cleaned = " ".join(re.sub(r"<[^>]+>", " ", last_user_msg).split())
            summary = cleaned[:50] if cleaned else "Untitled"

        return ClaudeSession(
            session_id=session_id,
            summary=summary,
            message_count=message_count,
            file_path=str(file_path),
        )

    # --- Directory session listing ---

    async def list_sessions_for_directory(self, cwd: str) -> list[ClaudeSession]:
        """List existing Claude sessions for a directory.

        Encodes the cwd path to find the project directory under
        ~/.claude/projects/{encoded_cwd}/, globs *.jsonl files, and
        extracts summary info from each.

        Returns a list sorted by mtime (most recent first), capped at 10.
        """
        encoded_cwd = self._encode_cwd(cwd)
        project_dir = config.claude_projects_path / encoded_cwd
        if not project_dir.is_dir():
            return []

        # Collect JSONL files sorted by mtime (newest first)
        jsonl_files = sorted(
            project_dir.glob("*.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        # Skip sessions-index and cap at 10
        sessions: list[ClaudeSession] = []
        for f in jsonl_files:
            if f.stem == "sessions-index":
                continue
            if len(sessions) >= 10:
                break
            session_id = f.stem
            session = await self._get_session_direct(session_id, cwd)
            if session and session.message_count > 0:
                sessions.append(session)
        return sessions

    # --- Window → Session resolution ---

    async def resolve_session_for_window(self, window_id: str) -> ClaudeSession | None:
        """Resolve a tmux window to the best matching Claude session.

        Uses persisted session_id + cwd to construct file path directly.
        Returns None if no session is associated with this window.
        """
        state = self.get_window_state(window_id)

        if not state.session_id or not state.cwd:
            return None

        session = await self._get_session_direct(state.session_id, state.cwd)
        if session:
            return session

        # File no longer exists, clear state
        logger.warning(
            "Session file no longer exists for window_id %s (sid=%s, cwd=%s)",
            window_id,
            state.session_id,
            state.cwd,
        )
        state.session_id = ""
        state.cwd = ""
        self._save_state()
        return None

    # --- Thread binding management ---

    def bind_thread(
        self, user_id: int, thread_id: int, window_id: str, window_name: str = ""
    ) -> None:
        """Bind a Telegram topic thread to a tmux window.

        Args:
            user_id: Telegram user ID
            thread_id: Telegram topic thread ID
            window_id: Tmux window ID (e.g. '@0')
            window_name: Display name for the window (optional)
        """
        for other_id, other in self.window_states.items():
            if (
                other_id != window_id
                and other.thread_id == thread_id
                and other.user_id == user_id
            ):
                other.user_id = None
                other.thread_id = None
        ws = self.get_window_state(window_id)
        ws.user_id = user_id
        ws.thread_id = thread_id
        if window_name:
            ws.window_name = window_name
        self._save_state()
        display = window_name or self.get_display_name(window_id)
        logger.info(
            "Bound thread %d -> window_id %s (%s) for user %d",
            thread_id,
            window_id,
            display,
            user_id,
        )

    def unbind_thread(self, user_id: int, thread_id: int) -> str | None:
        """Remove a thread binding. Returns the previously bound window_id, or None."""
        window_id = self.get_window_for_thread(user_id, thread_id)
        if window_id is None:
            return None
        ws = self.window_states[window_id]
        ws.user_id = None
        ws.thread_id = None
        self._save_state()
        logger.info(
            "Unbound thread %d (was %s) for user %d",
            thread_id,
            window_id,
            user_id,
        )
        return window_id

    def get_window_for_thread(self, user_id: int, thread_id: int) -> str | None:
        """Look up the window_id bound to a thread."""
        for window_id, ws in self.window_states.items():
            if ws.user_id == user_id and ws.thread_id == thread_id:
                return window_id
        return None

    def resolve_window_for_thread(
        self,
        user_id: int,
        thread_id: int | None,
    ) -> str | None:
        """Resolve the tmux window_id for a user's thread.

        Returns None if thread_id is None or the thread is not bound.
        """
        if thread_id is None:
            return None
        return self.get_window_for_thread(user_id, thread_id)

    def iter_thread_bindings(self) -> Iterator[tuple[int, int, str]]:
        """Iterate all thread bindings as (user_id, thread_id, window_id).

        Provides encapsulated access to the bindings without exposing the
        internal data structure directly.
        """
        for window_id, ws in self.window_states.items():
            if ws.user_id is not None and ws.thread_id is not None:
                yield ws.user_id, ws.thread_id, window_id

    async def find_users_for_session(
        self,
        session_id: str,
    ) -> list[tuple[int, str, int]]:
        """Find all users whose thread-bound window maps to the given session_id.

        Returns list of (user_id, window_id, thread_id) tuples.
        """
        result: list[tuple[int, str, int]] = []
        for user_id, thread_id, window_id in self.iter_thread_bindings():
            # In-memory lookup only: window_states carries the authoritative
            # window→session mapping (synced from session_map each poll cycle).
            # Reading the JSONL here (resolve_session_for_window) would be
            # O(bindings × file size) on every incoming message.
            state = self.window_states.get(window_id)
            if state and state.session_id == session_id:
                result.append((user_id, window_id, thread_id))
        return result

    # --- Tmux helpers ---

    async def _clear_blocking_dialog(self, window_id: str, display: str) -> bool:
        """Dismiss any dialog standing between the pane and Claude's prompt.

        Claude Code shows full-screen dialogs its own upgrades introduce (the
        auto-mode onboarding, for one).  They match no known UI pattern, so
        ccbot can neither forward the question nor deliver a message — text
        sent while one is up is read as navigation keys and vanishes.  Escape
        backs out of them.

        Known interactive UIs pass through untouched: those get surfaced to
        the user, whose answer arrives through this same send path.

        Returns True when the prompt is reachable.
        """
        for _ in range(3):
            pane = await tmux_manager.capture_pane(window_id)
            if pane is None:
                return True  # Can't read the pane — don't block the send.
            if not is_blocking_dialog(pane):
                return True
            logger.warning("Dismissing blocking dialog in %s", display)
            await tmux_manager.send_keys(
                window_id, "Escape", enter=False, literal=False
            )
            await asyncio.sleep(0.6)

        pane = await tmux_manager.capture_pane(window_id)
        return pane is None or is_prompt_ready(pane)

    async def send_to_window(self, window_id: str, text: str) -> tuple[bool, str]:
        """Send text to a tmux window by ID."""
        display = self.get_display_name(window_id)
        logger.debug(
            "send_to_window: window_id=%s (%s), text_len=%d",
            window_id,
            display,
            len(text),
        )
        window = await tmux_manager.find_window_by_id(window_id)
        if not window:
            return False, "Window not found (may have been closed)"
        if not await self._clear_blocking_dialog(window.window_id, display):
            return False, (
                f"{display} is stuck on a dialog I couldn't dismiss — "
                f"send /screenshot to see it, or /esc to clear it"
            )
        success = await tmux_manager.send_keys(window.window_id, text)
        if success:
            return True, f"Sent to {display}"
        return False, "Failed to send keys"

    # --- Message history ---

    async def get_recent_messages(
        self,
        window_id: str,
        *,
        start_byte: int = 0,
        end_byte: int | None = None,
    ) -> tuple[list[dict], int]:
        """Get user/assistant messages for a window's session.

        Resolves window → session, then reads the JSONL.
        Supports byte range filtering via start_byte/end_byte.
        Returns (messages, total_count).
        """
        session = await self.resolve_session_for_window(window_id)
        if not session or not session.file_path:
            return [], 0

        file_path = Path(session.file_path)
        if not file_path.exists():
            return [], 0

        # Read JSONL entries (optionally filtered by byte range)
        entries: list[dict] = []
        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                if start_byte > 0:
                    await f.seek(start_byte)

                while True:
                    # Check byte limit before reading
                    if end_byte is not None:
                        current_pos = await f.tell()
                        if current_pos >= end_byte:
                            break

                    line = await f.readline()
                    if not line:
                        break

                    data = TranscriptParser.parse_line(line)
                    if data:
                        entries.append(data)
        except OSError as e:
            logger.error("Error reading session file %s: %s", file_path, e)
            return [], 0

        parsed_entries, _ = TranscriptParser.parse_entries(entries)
        all_messages = [
            {
                "role": e.role,
                "text": e.text,
                "content_type": e.content_type,
                "timestamp": e.timestamp,
            }
            for e in parsed_entries
        ]

        return all_messages, len(all_messages)


session_manager = SessionManager()
