"""Tests for terminal_parser — regex-based detection of Claude Code UI elements."""

import pytest

from ccbot.terminal_parser import (
    extract_bash_output,
    extract_command_result,
    extract_interactive_content,
    is_blocking_dialog,
    is_interactive_ui,
    is_keystroke_menu,
    is_post_turn_status,
    is_prompt_ready,
    parse_menu_options,
    parse_status_line,
    strip_pane_chrome,
)

# ── parse_status_line ────────────────────────────────────────────────────


class TestParseStatusLine:
    @pytest.mark.parametrize(
        ("spinner", "rest", "expected"),
        [
            ("·", "Working on task", "Working on task"),
            ("✻", "  Reading file  ", "Reading file"),
            ("✽", "Thinking deeply", "Thinking deeply"),
            ("✶", "Analyzing code", "Analyzing code"),
            ("✳", "Processing input", "Processing input"),
            ("✢", "Building project", "Building project"),
        ],
    )
    def test_spinner_chars(self, spinner: str, rest: str, expected: str, chrome: str):
        pane = f"some output\n{spinner}{rest}\n{chrome}"
        assert parse_status_line(pane) == expected

    @pytest.mark.parametrize(
        "pane",
        [
            pytest.param("just normal text\nno spinners here\n", id="no_spinner"),
            pytest.param("", id="empty"),
        ],
    )
    def test_returns_none(self, pane: str):
        assert parse_status_line(pane) is None

    def test_no_chrome_returns_none(self):
        """Without chrome separator, status can't be determined."""
        pane = "output\n✻ Doing work\nno chrome here\n"
        assert parse_status_line(pane) is None

    def test_blank_line_between_status_and_chrome(self, chrome: str):
        """Status line with blank lines before separator."""
        pane = f"output\n✻ Doing work\n\n{chrome}"
        assert parse_status_line(pane) == "Doing work"

    def test_idle_no_status(self, chrome: str):
        """Idle pane (no status line above chrome) returns None."""
        pane = f"some output\n● Tool result\n{chrome}"
        assert parse_status_line(pane) is None

    def test_false_positive_bullet(self, chrome: str):
        """· in regular output must NOT be detected as status."""
        pane = f"· bullet point one\n· bullet point two\nsome result\n{chrome}"
        assert parse_status_line(pane) is None

    def test_tip_block_between_status_and_chrome(self, chrome: str):
        """Claude Code renders a tip block under the status line."""
        pane = (
            "some output\n"
            "✳ Gitifying… (2m 1s · ↓ 6.5k tokens)\n"
            "  ⎿ Tip: Use /btw to ask a quick side question without interrupting\n"
            "     current work\n"
            "\n"
            f"{chrome}"
        )
        assert parse_status_line(pane) == "Gitifying… (2m 1s · ↓ 6.5k tokens)"

    def test_finished_marker_is_not_status(self, chrome: str):
        """The completed marker carries no ellipsis and must not count."""
        pane = f"some output\n✻ Sautéed for 7s\nsome result\n{chrome}"
        assert parse_status_line(pane) is None

    def test_uses_fixture(self, sample_pane_status_line: str):
        assert parse_status_line(sample_pane_status_line) == "Reading file src/main.py"


# ── is_post_turn_status ──────────────────────────────────────────────────


class TestIsPostTurnStatus:
    @pytest.mark.parametrize(
        "status_line",
        [
            "Propagating… (running stop hook · 8s · ↓ 292 tokens)",
            "Deciphering… (running stop hook · 1m 10s · ↓ 4.1k tokens)",
            "Wrapping up… (running subagent stop hook · 3s)",
            "Idling… (RUNNING STOP HOOK · 2s)",
        ],
    )
    def test_post_turn_hooks(self, status_line: str):
        assert is_post_turn_status(status_line) is True

    @pytest.mark.parametrize(
        "status_line",
        [
            None,
            "",
            "Percolating… (12s · ↓ 1.2k tokens)",
            "Reading file src/main.py",
            # Mid-turn hooks mean a reply is still coming.
            "Musing… (running PreToolUse hook · 1s)",
            "Musing… (running PostToolUse hook · 1s)",
            "Musing… (running UserPromptSubmit hook · 1s)",
        ],
    )
    def test_still_working(self, status_line: str | None):
        assert is_post_turn_status(status_line) is False


# ── extract_interactive_content ──────────────────────────────────────────


class TestExtractInteractiveContent:
    def test_exit_plan_mode(self, sample_pane_exit_plan: str):
        result = extract_interactive_content(sample_pane_exit_plan)
        assert result is not None
        assert result.name == "ExitPlanMode"
        assert "Would you like to proceed?" in result.content
        assert "ctrl-g to edit in" in result.content

    def test_exit_plan_mode_variant(self):
        pane = (
            "  Claude has written up a plan\n  ─────\n  Details here\n  Esc to cancel\n"
        )
        result = extract_interactive_content(pane)
        assert result is not None
        assert result.name == "ExitPlanMode"
        assert "Claude has written up a plan" in result.content

    def test_ask_user_multi_tab(self, sample_pane_ask_user_multi_tab: str):
        result = extract_interactive_content(sample_pane_ask_user_multi_tab)
        assert result is not None
        assert result.name == "AskUserQuestion"
        assert "←" in result.content

    def test_ask_user_single_tab(self, sample_pane_ask_user_single_tab: str):
        result = extract_interactive_content(sample_pane_ask_user_single_tab)
        assert result is not None
        assert result.name == "AskUserQuestion"
        assert "Enter to select" in result.content

    def test_permission_prompt(self, sample_pane_permission: str):
        result = extract_interactive_content(sample_pane_permission)
        assert result is not None
        assert result.name == "PermissionPrompt"
        assert "Do you want to proceed?" in result.content

    def test_permission_prompt_keeps_the_preamble(self):
        """The question sits at the foot of its box — what it asks about is above it."""
        pane = "\n".join(
            [
                "  Ran 3 shell commands",
                "",
                "─" * 60,
                ' Use skill "schedule"?',
                " Claude may use instructions, code, or files from this Skill.",
                "",
                " Do you want to proceed?",
                " ❯ 1. Yes",
                "   2. No",
                "",
                " Esc to cancel · Tab to amend",
            ]
        )
        result = extract_interactive_content(pane)
        assert result is not None
        assert result.name == "PermissionPrompt"
        assert 'Use skill "schedule"?' in result.content
        assert "Ran 3 shell commands" not in result.content

    def test_permission_prompt_preamble_is_capped(self):
        """A preamble with no border in reach is cut short and marked."""
        pane = "\n".join(
            ["  scrollback"] * 60
            + [" Do you want to proceed?", " ❯ 1. Yes", " Esc to cancel"]
        )
        result = extract_interactive_content(pane)
        assert result is not None
        assert result.content.startswith("…\n")
        # "…" + 30 preamble lines + the 3 dialog lines
        assert len(result.content.split("\n")) == 34

    def test_restore_checkpoint(self):
        pane = (
            "  Restore the code to a previous state?\n"
            "  ─────\n"
            "  Some details\n"
            "  Enter to continue\n"
        )
        result = extract_interactive_content(pane)
        assert result is not None
        assert result.name == "RestoreCheckpoint"
        assert "Restore the code" in result.content

    def test_settings(self):
        pane = "  Settings: press tab to cycle\n  ─────\n  Option 1\n  Esc to cancel\n"
        result = extract_interactive_content(pane)
        assert result is not None
        assert result.name == "Settings"
        assert "Settings:" in result.content

    def test_settings_model_picker(self, sample_pane_settings: str):
        result = extract_interactive_content(sample_pane_settings)
        assert result is not None
        assert result.name == "Settings"
        assert "Select model" in result.content
        assert "Sonnet" in result.content
        assert "Enter to confirm" in result.content

    def test_settings_esc_to_cancel_bottom(self):
        pane = (
            "  Settings: press tab to cycle\n"
            "  ─────\n"
            "  Model\n"
            "  ─────\n"
            "  ● claude-sonnet-4-20250514\n"
            "  ○ claude-opus-4-20250514\n"
            "  Esc to cancel\n"
        )
        result = extract_interactive_content(pane)
        assert result is not None
        assert result.name == "Settings"
        assert "Esc to cancel" in result.content

    def test_settings_esc_to_exit_bottom(self):
        pane = (
            "  Settings: press tab to cycle\n"
            "  ─────\n"
            "  Model\n"
            "  ─────\n"
            "  ● Default (Opus 4.6)\n"
            "  ○ claude-sonnet-4-20250514\n"
            "\n"
            "  Enter to confirm · Esc to exit\n"
        )
        result = extract_interactive_content(pane)
        assert result is not None
        assert result.name == "Settings"
        assert "Enter to confirm" in result.content

    @pytest.mark.parametrize(
        "pane",
        [
            pytest.param("$ echo hello\nhello\n$\n", id="no_ui"),
            pytest.param("", id="empty"),
        ],
    )
    def test_returns_none(self, pane: str):
        assert extract_interactive_content(pane) is None

    def test_switch_model_confirmation(self):
        pane = (
            "⏺ Shiz here.\n"
            "\n"
            "▔" * 40 + "\n"
            "   Switch model?\n"
            "   Your next response will be slower and use more tokens\n"
            "\n"
            "   This conversation is cached for the current model. Switching to\n"
            "   Opus 5 means the full history gets re-read on your next message.\n"
            "\n"
            "   ❯ 1. Yes, switch to Opus 5\n"
            "     2. No, go back\n"
        )
        result = extract_interactive_content(pane)
        assert result is not None
        assert result.name == "ConfirmChoice"
        # The reason for the confirmation has to survive, not just the options.
        assert "cached for the current model" in result.content
        assert parse_menu_options(result.content) == [
            ("1", "Yes, switch to Opus 5"),
            ("2", "No, go back"),
        ]

    def test_min_gap_too_small_returns_none(self):
        pane = "  Do you want to proceed?\n  Esc to cancel\n"
        assert extract_interactive_content(pane) is None


# ── is_interactive_ui ────────────────────────────────────────────────────


class TestIsInteractiveUI:
    def test_true_when_ui_present(self, sample_pane_exit_plan: str):
        assert is_interactive_ui(sample_pane_exit_plan) is True

    def test_false_when_no_ui(self, sample_pane_no_ui: str):
        assert is_interactive_ui(sample_pane_no_ui) is False

    def test_settings_is_interactive(self, sample_pane_settings: str):
        assert is_interactive_ui(sample_pane_settings) is True

    def test_false_for_empty_string(self):
        assert is_interactive_ui("") is False


# ── is_keystroke_menu ────────────────────────────────────────────────────


class TestIsKeystrokeMenu:
    def test_true_for_model_picker(self, sample_pane_settings: str):
        assert is_keystroke_menu(sample_pane_settings) is True

    def test_false_for_ui_that_takes_typed_text(self, sample_pane_exit_plan: str):
        assert is_keystroke_menu(sample_pane_exit_plan) is False

    def test_false_when_no_ui(self, sample_pane_no_ui: str):
        assert is_keystroke_menu(sample_pane_no_ui) is False


# ── parse_menu_options ───────────────────────────────────────────────────


MODEL_MENU = (
    "  Select model\n"
    "  Switch between Claude models.\n"
    "\n"
    "    1. Default (recommended)  Sonnet 5 · Efficient for routine tasks\n"
    "    2. Sonnet                 Sonnet 5 · Efficient for routine tasks\n"
    "    3. Fable                  Fable 5 · Most capable for your hardest and\n"
    "                              longest-running tasks\n"
    "  ❯ 4. Opus ✔                 Opus 5 · Best for everyday, complex tasks\n"
    "    5. Haiku                  Haiku 4.5 · Fastest for quick answers\n"
    "\n"
    "  Enter to set as default · s to use this session only · Esc to cancel\n"
)


class TestParseMenuOptions:
    def test_reads_every_row_in_order(self):
        assert parse_menu_options(MODEL_MENU) == [
            ("1", "Default (recommended)"),
            ("2", "Sonnet"),
            ("3", "Fable"),
            ("4", "Opus ✔"),
            ("5", "Haiku"),
        ]

    def test_wrapped_description_is_not_a_row(self):
        assert "longest-running tasks" not in dict(parse_menu_options(MODEL_MENU))

    def test_no_numbered_rows(self):
        assert parse_menu_options("  Settings: press tab to cycle\n  Esc to cancel\n") == []


# ── extract_command_result ───────────────────────────────────────────────


class TestExtractCommandResult:
    def test_reads_the_last_result_line(self):
        pane = (
            "❯ /model\n"
            "  ⎿  Set model to Haiku 4.5 and saved as your default\n"
            "\n"
            "❯ /model\n"
            "  ⎿  Set model to Fable 5 and saved as your default\n"
            "\n"
            "─" * 40 + "\n"
            "❯ \n"
            "─" * 40 + "\n"
            "  ⏵⏵ auto mode on\n"
        )
        assert extract_command_result(pane) == (
            "Set model to Fable 5 and saved as your default"
        )

    def test_none_when_no_result_line(self):
        assert extract_command_result("❯ hello\n\n" + "─" * 40 + "\n❯ \n") is None


# ── strip_pane_chrome ───────────────────────────────────────────────────


class TestStripPaneChrome:
    def test_strips_from_separator(self):
        lines = [
            "some output",
            "more output",
            "─" * 30,
            "❯",
            "─" * 30,
            "  [Opus 4.6] Context: 34%",
        ]
        assert strip_pane_chrome(lines) == ["some output", "more output"]

    def test_no_separator_returns_all(self):
        lines = ["line 1", "line 2", "line 3"]
        assert strip_pane_chrome(lines) == lines

    def test_short_separator_not_triggered(self):
        lines = ["output", "─" * 10, "more output"]
        assert strip_pane_chrome(lines) == lines

    def test_only_searches_last_10_lines(self):
        # Separator at line 0 with 15 lines total — outside the last-10 window
        lines = ["─" * 30] + [f"line {i}" for i in range(14)]
        assert strip_pane_chrome(lines) == lines


# ── extract_bash_output ─────────────────────────────────────────────────


class TestExtractBashOutput:
    def test_extracts_command_output(self):
        pane = "some context\n! echo hello\n⎿ hello\n"
        result = extract_bash_output(pane, "echo hello")
        assert result is not None
        assert "! echo hello" in result
        assert "hello" in result

    def test_command_not_found_returns_none(self):
        pane = "some context\njust normal output\n"
        assert extract_bash_output(pane, "echo hello") is None

    def test_chrome_stripped(self):
        pane = (
            "some context\n"
            "! ls\n"
            "⎿ file.txt\n"
            + "─" * 30
            + "\n"
            + "❯\n"
            + "─" * 30
            + "\n"
            + "  [Opus 4.6] Context: 34%\n"
        )
        result = extract_bash_output(pane, "ls")
        assert result is not None
        assert "file.txt" in result
        assert "Opus" not in result

    def test_prefix_match_long_command(self):
        pane = "! long_comma…\n⎿ output\n"
        result = extract_bash_output(pane, "long_command_that_gets_truncated")
        assert result is not None
        assert "output" in result

    def test_trailing_blank_lines_stripped(self):
        pane = "! echo hi\n⎿ hi\n\n\n"
        result = extract_bash_output(pane, "echo hi")
        assert result is not None
        assert not result.endswith("\n")


# ── is_prompt_ready / is_blocking_dialog ─────────────────────────────────

BAR = "─" * 80
IDLE_PANE = (
    "  Two weeks since my last note. What's on today?\n"
    "\n"
    "✻ Cooked for 19s\n"
    "\n" + "─" * 72 + " shiz ──\n"
    "❯ catch me up on what's open\n" + BAR + "\n"
    "  ⏸ manual mode on · ? for shortcuts · ← for agents\n"
)

# Full-screen onboarding: keeps the box borders, drops the ❯ line and footer.
ONBOARDING_MENU_PANE = (
    "─" * 72 + " shiz ──\n"
    "\n" + BAR + "\n"
    " Make auto mode your default permission mode?\n"
    "\n"
    "   Auto mode lets Claude handle permission prompts automatically.\n"
    "\n"
    "   ❯ 1. Yes, set auto mode as my default permission mode\n"
    "     2. No, keep manual mode\n"
)

ONBOARDING_FORM_PANE = (
    "▔" * 80 + "\n"
    "   Set up auto mode for your environment?\n"
    "\n"
    "   Claude Code reads this project and your recent Claude sessions.\n"
    "\n"
    "     How you use Claude here    ◀ Mixed ▶\n"
    "   ❯ Also scan shell history    [ ]\n"
    "\n"
    "     Continue\n"
    "\n"
    "   ←/→ to change usage · Enter to continue · Esc to cancel\n"
)


class TestIsPromptReady:
    def test_idle_prompt_is_ready(self):
        assert is_prompt_ready(IDLE_PANE) is True

    def test_mid_turn_prompt_is_ready(self):
        pane = IDLE_PANE.replace("? for shortcuts", "esc to interrupt")
        assert is_prompt_ready(pane) is True

    def test_empty_prompt_is_ready(self):
        assert (
            is_prompt_ready(IDLE_PANE.replace("catch me up on what's open", "")) is True
        )

    def test_onboarding_menu_is_not_ready(self):
        assert is_prompt_ready(ONBOARDING_MENU_PANE) is False

    def test_onboarding_form_is_not_ready(self):
        assert is_prompt_ready(ONBOARDING_FORM_PANE) is False

    def test_empty_pane_is_not_ready(self):
        assert is_prompt_ready("") is False

    def test_separator_without_footer_is_not_ready(self):
        assert is_prompt_ready("❯ typing\n" + BAR + "\n") is False


class TestIsBlockingDialog:
    def test_idle_prompt_does_not_block(self):
        assert is_blocking_dialog(IDLE_PANE) is False

    def test_unrecognised_onboarding_blocks(self):
        assert is_blocking_dialog(ONBOARDING_MENU_PANE) is True
        assert is_blocking_dialog(ONBOARDING_FORM_PANE) is True

    def test_known_interactive_ui_does_not_block(self):
        pane = "  Do you want to proceed?\n  ❯ 1. Yes\n    2. No\n  Esc to cancel\n"
        assert is_interactive_ui(pane) is True
        assert is_blocking_dialog(pane) is False

    def test_trust_folder_prompt_does_not_block(self):
        """Claude Code's first-run folder check is a recognised UI."""
        pane = (
            "  Do you want to proceed?\n"
            "  Claude Code'll be able to read, edit, and execute files here.\n"
            "\n"
            "  ❯ 1. Yes, I trust this folder\n"
            "    2. No, exit\n"
            "\n"
            "  Enter to confirm · Esc to cancel\n"
        )
        assert is_blocking_dialog(pane) is False

    @pytest.mark.parametrize(
        "pane",
        [
            pytest.param("", id="empty"),
            pytest.param("\n\n\n", id="blank_lines"),
            pytest.param("Booting…\n\n", id="one_line"),
        ],
    )
    def test_window_still_drawing_does_not_block(self, pane: str):
        """A pane mid-boot buffers keystrokes; it is not a dialog."""
        assert is_blocking_dialog(pane) is False
