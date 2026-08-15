"""Tests for SessionManager pure dict operations."""

import json

import pytest

import ccbot.session as session_mod
from ccbot.session import SessionManager
from ccbot.tmux_manager import TmuxWindow


@pytest.fixture
def mgr(monkeypatch) -> SessionManager:
    monkeypatch.setattr(SessionManager, "_load_state", lambda self: None)
    monkeypatch.setattr(SessionManager, "_save_state", lambda self: None)
    return SessionManager()


class TestThreadBindings:
    def test_bind_and_get(self, mgr: SessionManager) -> None:
        mgr.bind_thread(100, 1, "@1")
        assert mgr.get_window_for_thread(100, 1) == "@1"

    def test_bind_unbind_get_returns_none(self, mgr: SessionManager) -> None:
        mgr.bind_thread(100, 1, "@1")
        mgr.unbind_thread(100, 1)
        assert mgr.get_window_for_thread(100, 1) is None

    def test_unbind_nonexistent_returns_none(self, mgr: SessionManager) -> None:
        assert mgr.unbind_thread(100, 999) is None

    def test_iter_thread_bindings(self, mgr: SessionManager) -> None:
        mgr.bind_thread(100, 1, "@1")
        mgr.bind_thread(100, 2, "@2")
        mgr.bind_thread(200, 3, "@3")
        result = set(mgr.iter_thread_bindings())
        assert result == {(100, 1, "@1"), (100, 2, "@2"), (200, 3, "@3")}


class TestGroupChatId:
    """Tests for group chat_id routing (supergroup forum topic support).

    IMPORTANT: These tests protect against regression. The group_chat_ids
    mapping is required for Telegram supergroup forum topics — without it,
    all outbound messages fail with "Message thread not found". This was
    erroneously removed once (26cb81f) and restored in PR #23. Do NOT
    delete these tests or the underlying functionality.
    """

    def test_resolve_with_stored_group_id(self, mgr: SessionManager) -> None:
        """resolve_chat_id returns stored group chat_id for known thread."""
        mgr.set_group_chat_id(100, 1, -1001234567890)
        assert mgr.resolve_chat_id(100, 1) == -1001234567890

    def test_resolve_without_group_id_falls_back_to_user_id(
        self, mgr: SessionManager
    ) -> None:
        """resolve_chat_id falls back to user_id when no group_id stored."""
        assert mgr.resolve_chat_id(100, 1) == 100

    def test_resolve_none_thread_id_falls_back_to_user_id(
        self, mgr: SessionManager
    ) -> None:
        """resolve_chat_id returns user_id when thread_id is None (private chat)."""
        mgr.set_group_chat_id(100, 1, -1001234567890)
        assert mgr.resolve_chat_id(100) == 100

    def test_set_group_chat_id_overwrites(self, mgr: SessionManager) -> None:
        """set_group_chat_id updates the stored value on change."""
        mgr.set_group_chat_id(100, 1, -999)
        mgr.set_group_chat_id(100, 1, -888)
        assert mgr.resolve_chat_id(100, 1) == -888

    def test_multiple_threads_independent(self, mgr: SessionManager) -> None:
        """Different threads for the same user store independent group chat_ids."""
        mgr.set_group_chat_id(100, 1, -111)
        mgr.set_group_chat_id(100, 2, -222)
        assert mgr.resolve_chat_id(100, 1) == -111
        assert mgr.resolve_chat_id(100, 2) == -222

    def test_multiple_users_independent(self, mgr: SessionManager) -> None:
        """Different users store independent group chat_ids."""
        mgr.set_group_chat_id(100, 1, -111)
        mgr.set_group_chat_id(200, 1, -222)
        assert mgr.resolve_chat_id(100, 1) == -111
        assert mgr.resolve_chat_id(200, 1) == -222

    def test_set_group_chat_id_with_none_thread(self, mgr: SessionManager) -> None:
        """set_group_chat_id handles None thread_id (mapped to 0)."""
        mgr.set_group_chat_id(100, None, -999)
        # thread_id=None in resolve falls back to user_id (by design)
        assert mgr.resolve_chat_id(100, None) == 100
        # The stored key is "100:0", only accessible with explicit thread_id=0
        assert mgr.group_chat_ids.get("100:0") == -999


class TestWindowState:
    def test_get_creates_new(self, mgr: SessionManager) -> None:
        state = mgr.get_window_state("@0")
        assert state.session_id == ""
        assert state.cwd == ""

    def test_get_returns_existing(self, mgr: SessionManager) -> None:
        state = mgr.get_window_state("@1")
        state.session_id = "abc"
        assert mgr.get_window_state("@1").session_id == "abc"

    def test_clear_window_session(self, mgr: SessionManager) -> None:
        state = mgr.get_window_state("@1")
        state.session_id = "abc"
        mgr.clear_window_session("@1")
        assert mgr.get_window_state("@1").session_id == ""


class TestResolveWindowForThread:
    def test_none_thread_id_returns_none(self, mgr: SessionManager) -> None:
        assert mgr.resolve_window_for_thread(100, None) is None

    def test_unbound_thread_returns_none(self, mgr: SessionManager) -> None:
        assert mgr.resolve_window_for_thread(100, 42) is None

    def test_bound_thread_returns_window(self, mgr: SessionManager) -> None:
        mgr.bind_thread(100, 42, "@3")
        assert mgr.resolve_window_for_thread(100, 42) == "@3"


class TestDisplayNames:
    def test_get_display_name_fallback(self, mgr: SessionManager) -> None:
        """get_display_name returns window_id when no display name is set."""
        assert mgr.get_display_name("@99") == "@99"

    def test_set_and_get_display_name(self, mgr: SessionManager) -> None:
        mgr.bind_thread(100, 1, "@1", window_name="myproject")
        assert mgr.get_display_name("@1") == "myproject"

    def test_set_display_name_update(self, mgr: SessionManager) -> None:
        mgr.bind_thread(100, 1, "@1", window_name="old-name")
        mgr.update_display_name("@1", "new-name")
        assert mgr.get_display_name("@1") == "new-name"

    def test_bind_thread_sets_display_name(self, mgr: SessionManager) -> None:
        mgr.bind_thread(100, 1, "@1", window_name="proj")
        assert mgr.get_display_name("@1") == "proj"

    def test_bind_thread_without_name_no_display(self, mgr: SessionManager) -> None:
        mgr.bind_thread(100, 1, "@1")
        # No display name set, fallback to window_id
        assert mgr.get_display_name("@1") == "@1"


class TestIsWindowId:
    def test_valid_ids(self, mgr: SessionManager) -> None:
        assert mgr._is_window_id("@0") is True
        assert mgr._is_window_id("@12") is True
        assert mgr._is_window_id("@999") is True

    def test_invalid_ids(self, mgr: SessionManager) -> None:
        assert mgr._is_window_id("myproject") is False
        assert mgr._is_window_id("@") is False
        assert mgr._is_window_id("") is False
        assert mgr._is_window_id("@abc") is False


class TestMigrateOldFormatMap:
    """Pure-function tests for _migrate_old_format_map (session:name -> session:@id)."""

    def test_migrates_name_to_window_id(self, mgr: SessionManager) -> None:
        session_map = {
            "ccbot:ccmux": {"session_id": "sid-1", "cwd": "/proj"},
        }
        changed = mgr._migrate_old_format_map(session_map, {"ccmux": "@4"})
        assert changed is True
        assert "ccbot:ccmux" not in session_map
        assert session_map["ccbot:@4"] == {
            "session_id": "sid-1",
            "cwd": "/proj",
            "window_name": "ccmux",  # backfilled
        }

    def test_drops_orphan_without_live_window(self, mgr: SessionManager) -> None:
        session_map = {"ccbot:gone": {"session_id": "sid-2", "cwd": "/x"}}
        changed = mgr._migrate_old_format_map(session_map, {})
        assert changed is True
        assert session_map == {}

    def test_prefers_existing_window_id_key(self, mgr: SessionManager) -> None:
        session_map = {
            "ccbot:ccmux": {"session_id": "old-sid", "cwd": "/proj"},
            "ccbot:@4": {"session_id": "new-sid", "cwd": "/proj"},
        }
        changed = mgr._migrate_old_format_map(session_map, {"ccmux": "@4"})
        assert changed is True
        # @id entry wins; old-format key discarded
        assert session_map == {"ccbot:@4": {"session_id": "new-sid", "cwd": "/proj"}}

    def test_noop_for_window_id_keys(self, mgr: SessionManager) -> None:
        session_map = {"ccbot:@4": {"session_id": "sid", "cwd": "/proj"}}
        changed = mgr._migrate_old_format_map(session_map, {"ccmux": "@4"})
        assert changed is False
        assert session_map == {"ccbot:@4": {"session_id": "sid", "cwd": "/proj"}}

    def test_ignores_other_tmux_sessions(self, mgr: SessionManager) -> None:
        # Keys for other tmux sessions (different prefix) are left untouched.
        session_map = {"0:2.1.76": {"session_id": "sid", "cwd": "/other"}}
        changed = mgr._migrate_old_format_map(session_map, {"2.1.76": "@9"})
        assert changed is False
        assert session_map == {"0:2.1.76": {"session_id": "sid", "cwd": "/other"}}


class TestResolveStaleIds:
    """Regression tests for startup re-resolution after tmux server restart.

    IMPORTANT: window IDs reset when the tmux server restarts; thread
    bindings MUST survive by re-resolving through display names, which are
    the only identifier that outlives the old id.
    """

    def _setup(self, mgr: SessionManager, monkeypatch, tmp_path, live) -> None:
        async def fake_list_windows() -> list[TmuxWindow]:
            return live

        monkeypatch.setattr(session_mod.tmux_manager, "list_windows", fake_list_windows)
        # Point session_map at a nonexistent file so the trailing
        # session_map cleanup steps are no-ops
        monkeypatch.setattr(
            session_mod.config, "session_map_file", tmp_path / "no_map.json"
        )

    @pytest.mark.asyncio
    async def test_tmux_restart_remaps_the_record(
        self, mgr: SessionManager, monkeypatch, tmp_path
    ) -> None:
        """The whole record follows a stale window_id to the new id."""
        state = mgr.get_window_state("@5")
        state.session_id = "sid-1"
        state.cwd = "/proj"
        mgr.bind_thread(100, 42, "@5", window_name="proj")

        # tmux restarted: same window name, new id
        self._setup(mgr, monkeypatch, tmp_path, [TmuxWindow("@1", "proj", "/proj")])
        await mgr.resolve_stale_ids()

        assert mgr.get_window_for_thread(100, 42) == "@1"
        assert mgr.window_states["@1"].session_id == "sid-1"
        assert "@5" not in mgr.window_states
        assert mgr.get_display_name("@1") == "proj"

    @pytest.mark.asyncio
    async def test_binding_without_window_state_still_remaps(
        self, mgr: SessionManager, monkeypatch, tmp_path
    ) -> None:
        """Bindings resolve via the display name after the id goes stale."""
        mgr.bind_thread(100, 9, "@7", window_name="other")

        self._setup(mgr, monkeypatch, tmp_path, [TmuxWindow("@2", "other", "/o")])
        await mgr.resolve_stale_ids()

        assert mgr.get_window_for_thread(100, 9) == "@2"

    @pytest.mark.asyncio
    async def test_unmatched_binding_dropped(
        self, mgr: SessionManager, monkeypatch, tmp_path
    ) -> None:
        """A stale binding with no live window of the same name is dropped."""
        mgr.bind_thread(100, 42, "@5", window_name="gone")

        self._setup(mgr, monkeypatch, tmp_path, [TmuxWindow("@1", "unrelated", "/u")])
        await mgr.resolve_stale_ids()

        assert mgr.get_window_for_thread(100, 42) is None

    @pytest.mark.asyncio
    async def test_stale_record_never_clobbers_a_live_namesake(
        self, mgr: SessionManager, monkeypatch, tmp_path
    ) -> None:
        """tmux reuses a name once the original window dies.

        The dead record must not be re-resolved onto the live window of the
        same name — that would overwrite a live session and drop its topic.
        """
        dead = mgr.get_window_state("@31")
        dead.session_id = "sid-dead"
        dead.window_name = "projects-2"

        mgr.bind_thread(100, 55, "@38", window_name="projects-2")
        mgr.get_window_state("@38").session_id = "sid-live"

        self._setup(mgr, monkeypatch, tmp_path, [TmuxWindow("@38", "projects-2", "/p")])
        await mgr.resolve_stale_ids()

        assert mgr.window_states["@38"].session_id == "sid-live"
        assert mgr.get_window_for_thread(100, 55) == "@38"
        assert "@31" not in mgr.window_states

    @pytest.mark.asyncio
    async def test_live_ids_untouched(
        self, mgr: SessionManager, monkeypatch, tmp_path
    ) -> None:
        """Bindings pointing at still-live window IDs are kept as-is."""
        mgr.bind_thread(100, 7, "@3", window_name="keep")
        state = mgr.get_window_state("@3")
        state.session_id = "sid-keep"

        self._setup(mgr, monkeypatch, tmp_path, [TmuxWindow("@3", "keep", "/k")])
        await mgr.resolve_stale_ids()

        assert mgr.get_window_for_thread(100, 7) == "@3"
        assert mgr.window_states["@3"].session_id == "sid-keep"


class TestLoadSessionMapMigration:
    """load_session_map self-heals old-format keys at runtime and fills window_states."""

    def _mock_windows(self, monkeypatch, windows: list[TmuxWindow]) -> None:
        async def fake_list_windows() -> list[TmuxWindow]:
            return windows

        monkeypatch.setattr(session_mod.tmux_manager, "list_windows", fake_list_windows)

    @pytest.mark.asyncio
    async def test_old_format_key_populates_window_state(
        self, mgr: SessionManager, monkeypatch, tmp_path
    ) -> None:
        map_file = tmp_path / "session_map.json"
        map_file.write_text(
            json.dumps({"ccbot:ccmux": {"session_id": "sid-1", "cwd": "/proj"}})
        )
        monkeypatch.setattr(session_mod.config, "session_map_file", map_file)
        self._mock_windows(monkeypatch, [TmuxWindow("@4", "ccmux", "/proj")])

        await mgr.load_session_map()

        # Delivery path can now resolve @4 -> sid-1
        state = mgr.get_window_state("@4")
        assert state.session_id == "sid-1"
        assert state.cwd == "/proj"
        # File was rewritten to the @window_id form
        rewritten = json.loads(map_file.read_text())
        assert "ccbot:ccmux" not in rewritten
        assert rewritten["ccbot:@4"]["session_id"] == "sid-1"

    @pytest.mark.asyncio
    async def test_no_tmux_lookup_when_no_old_keys(
        self, mgr: SessionManager, monkeypatch, tmp_path
    ) -> None:
        map_file = tmp_path / "session_map.json"
        map_file.write_text(
            json.dumps({"ccbot:@4": {"session_id": "sid-1", "cwd": "/proj"}})
        )
        monkeypatch.setattr(session_mod.config, "session_map_file", map_file)

        async def boom() -> list[TmuxWindow]:
            raise AssertionError("list_windows should not be called without old keys")

        monkeypatch.setattr(session_mod.tmux_manager, "list_windows", boom)

        await mgr.load_session_map()
        assert mgr.get_window_state("@4").session_id == "sid-1"


class TestStateMigration:
    """v1 spread a window across four maps; v2 keeps one record per window."""

    V1 = {
        "window_states": {
            "@3": {"session_id": "sid-a", "cwd": "/proj", "auto_named": True},
        },
        "window_display_names": {"@3": "Real Name", "@9": "Orphan"},
        "thread_bindings": {"100": {"42": "@3"}},
        "user_window_offsets": {"100": {"@3": 999}},
        "group_chat_ids": {"100:42": -1001234},
    }

    def _load(self, monkeypatch, tmp_path, payload) -> SessionManager:
        path = tmp_path / "state.json"
        path.write_text(json.dumps(payload))
        monkeypatch.setattr(session_mod.config, "state_file", path)
        monkeypatch.setattr(SessionManager, "_save_state", lambda self: None)
        return SessionManager()

    def test_v1_folds_into_one_record(self, monkeypatch, tmp_path) -> None:
        mgr = self._load(monkeypatch, tmp_path, self.V1)
        ws = mgr.window_states["@3"]
        assert ws.session_id == "sid-a"
        assert ws.cwd == "/proj"
        assert ws.auto_named is True
        assert ws.window_name == "Real Name"
        assert (ws.user_id, ws.thread_id) == (100, 42)
        assert mgr.get_window_for_thread(100, 42) == "@3"
        assert mgr.get_display_name("@3") == "Real Name"

    def test_v1_display_name_without_window_state_survives(
        self, monkeypatch, tmp_path
    ) -> None:
        """A name with no window_states entry still becomes a record."""
        mgr = self._load(monkeypatch, tmp_path, self.V1)
        assert mgr.get_display_name("@9") == "Orphan"

    def test_group_chat_ids_kept(self, monkeypatch, tmp_path) -> None:
        mgr = self._load(monkeypatch, tmp_path, self.V1)
        assert mgr.resolve_chat_id(100, 42) == -1001234

    def test_dead_offsets_dropped(self, monkeypatch, tmp_path) -> None:
        """monitor_state.json owns byte offsets; v1's copy was never read."""
        mgr = self._load(monkeypatch, tmp_path, self.V1)
        assert not hasattr(mgr, "user_window_offsets")

    def test_v2_round_trips(self, monkeypatch, tmp_path) -> None:
        mgr = self._load(monkeypatch, tmp_path, self.V1)
        v2 = {
            "version": 2,
            "window_states": {k: v.to_dict() for k, v in mgr.window_states.items()},
            "group_chat_ids": mgr.group_chat_ids,
        }
        again = self._load(monkeypatch, tmp_path, v2)
        assert again.get_window_for_thread(100, 42) == "@3"
        assert again.get_display_name("@3") == "Real Name"
        assert again.window_states["@3"].auto_named is True
