"""Terminal output parser — detects Claude Code UI elements in pane text.

Parses captured tmux pane content to detect:
  - Interactive UIs (AskUserQuestion, ExitPlanMode, Permission Prompt,
    RestoreCheckpoint) via regex-based UIPattern matching with top/bottom
    delimiters.
  - Status line (spinner characters + working text) by scanning from bottom up.
  - Prompt readiness — whether the input box is on screen, so a send reaches
    Claude rather than a dialog drawn over it.

All Claude Code text patterns live here. To support a new UI type or
a changed Claude Code version, edit UI_PATTERNS / STATUS_SPINNERS.

Key functions: is_interactive_ui(), extract_interactive_content(),
is_keystroke_menu(), parse_menu_options(), extract_command_result(),
is_prompt_ready(), is_blocking_dialog(), parse_status_line(),
is_post_turn_status(), strip_pane_chrome(), extract_bash_output().
"""

import re
from dataclasses import dataclass


@dataclass
class InteractiveUIContent:
    """Content extracted from an interactive UI."""

    content: str  # The extracted display content
    name: str = ""  # Pattern name that matched (e.g. "AskUserQuestion")


@dataclass(frozen=True)
class UIPattern:
    """A text-marker pair that delimits an interactive UI region.

    Extraction scans lines top-down: the first line matching any `top` pattern
    marks the start, the first subsequent line matching any `bottom` pattern
    marks the end.  Both boundary lines are included in the extracted content.

    ``top`` and ``bottom`` are tuples of compiled regexes — any single match
    is sufficient.  This accommodates wording changes across Claude Code
    versions (e.g. a reworded confirmation prompt).

    ``back_to`` extends the region upward: from the ``top`` line, scan up to
    ``max_back`` lines for the dialog's opening border and start just below
    it.  A confirmation question sits at the *foot* of its box, so a pattern
    anchored on the question alone drops the tool name, command, skill
    description or diff the question is about.  When no ``back_to`` line is
    within reach the region starts ``max_back`` lines up, marked with ``…``.
    """

    name: str  # Descriptive label (not used programmatically)
    top: tuple[re.Pattern[str], ...]
    bottom: tuple[re.Pattern[str], ...]
    min_gap: int = 2  # minimum lines between top and bottom (inclusive)
    back_to: tuple[re.Pattern[str], ...] = ()  # opens the region above `top`
    max_back: int = 30  # how far above `top` to look for a `back_to` line


# Opening border of a Claude Code dialog box, drawn as a rule across the pane
# or as the top-left corner of a rounded box.
_RE_DIALOG_BORDER = re.compile(r"^\s*[╭╰]?─{5,}")


# ── UI pattern definitions (order matters — first match wins) ────────────

UI_PATTERNS: list[UIPattern] = [
    UIPattern(
        name="ExitPlanMode",
        top=(
            re.compile(r"^\s*Would you like to proceed\?"),
            # v2.1.29+: longer prefix that may wrap across lines
            re.compile(r"^\s*Claude has written up a plan"),
        ),
        bottom=(
            re.compile(r"^\s*ctrl-g to edit in "),
            re.compile(r"^\s*Esc to (cancel|exit)"),
        ),
    ),
    UIPattern(
        name="AskUserQuestion",
        top=(re.compile(r"^\s*←\s+[☐✔☒]"),),  # Multi-tab: no bottom needed
        bottom=(),
        min_gap=1,
    ),
    UIPattern(
        name="AskUserQuestion",
        top=(re.compile(r"^\s*[☐✔☒]"),),  # Single-tab: bottom required
        bottom=(re.compile(r"^\s*Enter to select"),),
        min_gap=1,
    ),
    UIPattern(
        name="PermissionPrompt",
        top=(
            re.compile(r"^\s*Do you want to proceed\?"),
            re.compile(r"^\s*Do you want to make this edit"),
            re.compile(r"^\s*Do you want to create \S"),
            re.compile(r"^\s*Do you want to delete \S"),
        ),
        bottom=(re.compile(r"^\s*Esc to cancel"),),
        back_to=(_RE_DIALOG_BORDER,),
    ),
    UIPattern(
        # Second stage of /model: picking a different model from the menu opens
        # this when the conversation is already cached, so the menu closing is
        # not the end of the switch.  Listed before the numbered permission
        # menu, whose top marker also matches the "1. Yes" line, so the
        # explanation above the options survives into the extract.
        name="ConfirmChoice",
        top=(re.compile(r"^\s*Switch model\?"),),
        bottom=(re.compile(r"^\s*2\.\s"),),
    ),
    UIPattern(
        # Permission menu with numbered choices (no "Esc to cancel" line)
        name="PermissionPrompt",
        top=(re.compile(r"^\s*❯\s*1\.\s*Yes"),),
        bottom=(),
        min_gap=2,
        back_to=(_RE_DIALOG_BORDER,),
    ),
    UIPattern(
        # Bash command approval
        name="BashApproval",
        top=(
            re.compile(r"^\s*Bash command\s*$"),
            re.compile(r"^\s*This command requires approval"),
        ),
        bottom=(re.compile(r"^\s*Esc to cancel"),),
    ),
    UIPattern(
        name="RestoreCheckpoint",
        top=(re.compile(r"^\s*Restore the code"),),
        bottom=(re.compile(r"^\s*Enter to continue"),),
    ),
    UIPattern(
        name="Settings",
        top=(
            re.compile(r"^\s*Settings:.*tab to cycle"),
            re.compile(r"^\s*Select model"),
        ),
        bottom=(
            re.compile(r"Esc to cancel"),
            re.compile(r"Esc to exit"),
            re.compile(r"Enter to confirm"),
            re.compile(r"^\s*Type to filter"),
        ),
    ),
]


# ── Post-processing ──────────────────────────────────────────────────────

_RE_LONG_DASH = re.compile(r"^─{5,}$")


def _shorten_separators(text: str) -> str:
    """Replace lines of 5+ ─ characters with exactly ─────."""
    return "\n".join(
        "─────" if _RE_LONG_DASH.match(line) else line for line in text.split("\n")
    )


# ── Core extraction ──────────────────────────────────────────────────────


def _try_extract(lines: list[str], pattern: UIPattern) -> InteractiveUIContent | None:
    """Try to extract content matching a single UI pattern.

    When ``pattern.bottom`` is empty, the region extends from the top marker
    to the last non-empty line (used for multi-tab AskUserQuestion where the
    bottom delimiter varies by tab).
    """
    top_idx: int | None = None
    bottom_idx: int | None = None

    for i, line in enumerate(lines):
        if top_idx is None:
            if any(p.search(line) for p in pattern.top):
                top_idx = i
        elif pattern.bottom and any(p.search(line) for p in pattern.bottom):
            bottom_idx = i
            break

    if top_idx is None:
        return None

    # No bottom patterns → use last non-empty line as boundary
    if not pattern.bottom:
        for i in range(len(lines) - 1, top_idx, -1):
            if lines[i].strip():
                bottom_idx = i
                break

    if bottom_idx is None or bottom_idx - top_idx < pattern.min_gap:
        return None

    start_idx, clipped = _widen_start(lines, top_idx, pattern)
    body = "\n".join(lines[start_idx : bottom_idx + 1]).rstrip()
    content = f"…\n{body}" if clipped else body
    return InteractiveUIContent(content=_shorten_separators(content), name=pattern.name)


def _widen_start(
    lines: list[str], top_idx: int, pattern: UIPattern
) -> tuple[int, bool]:
    """Move the region's start above ``top_idx`` to cover the dialog's preamble.

    Returns the new start index and whether the preamble was cut short of its
    opening border.
    """
    if not pattern.back_to or top_idx == 0:
        return top_idx, False
    floor = max(0, top_idx - pattern.max_back)
    for i in range(top_idx - 1, floor - 1, -1):
        if any(p.search(lines[i]) for p in pattern.back_to):
            return i + 1, False
    return floor, floor > 0


# ── Public API ───────────────────────────────────────────────────────────


def extract_interactive_content(pane_text: str) -> InteractiveUIContent | None:
    """Extract content from an interactive UI in terminal output.

    Tries each UI pattern in declaration order; first match wins.
    Returns None if no recognizable interactive UI is found.
    """
    if not pane_text:
        return None

    lines = pane_text.strip().split("\n")
    for pattern in UI_PATTERNS:
        result = _try_extract(lines, pattern)
        if result:
            return result
    return None


def is_interactive_ui(pane_text: str) -> bool:
    """Check if terminal currently shows an interactive UI."""
    return extract_interactive_content(pane_text) is not None


# UI names whose every keystroke is a hotkey. The other UIs take a typed
# answer — AskUserQuestion has an "Other" field, ExitPlanMode reads free text
# — so text sent into those still lands.
KEYSTROKE_MENUS = frozenset({"Settings", "ConfirmChoice"})


def is_keystroke_menu(pane_text: str) -> bool:
    """Check whether the pane shows a menu that consumes typed text.

    ``/model``, ``/usage`` and their siblings draw a selection menu with no
    text field: letters move or confirm the highlighted row and Enter closes
    the menu.  A message typed into one disappears without reaching Claude,
    and its trailing Enter commits whichever row happened to be selected.
    """
    content = extract_interactive_content(pane_text)
    return content is not None and content.name in KEYSTROKE_MENUS


# A menu row: optional cursor, the number key that picks it, then the label.
# The label ends at the two-space gutter before the description column.
_RE_MENU_ROW = re.compile(r"^\s*(?:❯\s*)?([1-9])\.\s+(\S.*?)(?:\s{2,}|$)")


def parse_menu_options(content: str) -> list[tuple[str, str]]:
    """Read the numbered rows of a selection menu as (key, label) pairs.

    Claude Code's menus take the row number as a single keystroke that both
    selects and confirms, so these keys let a caller pick a row outright
    instead of walking to it with arrows.  Rows are returned in screen order;
    a menu with no numbered rows yields an empty list.
    """
    options: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line in content.split("\n"):
        m = _RE_MENU_ROW.match(line)
        if m and m.group(1) not in seen:
            seen.add(m.group(1))
            options.append((m.group(1), m.group(2).strip()))
    return options


_RE_COMMAND_RESULT = re.compile(r"^\s*⎿\s+(\S.*)$")


def extract_command_result(pane_text: str) -> str | None:
    """Return the last ``⎿`` result line a slash command printed, if any.

    Closing a menu leaves its outcome ("Set model to Fable 5 …") on this line
    and nowhere else — it never reaches the transcript, so a caller that wants
    to report what a menu did has to read it off the pane.
    """
    lines = strip_pane_chrome(pane_text.splitlines())
    for line in reversed(lines):
        m = _RE_COMMAND_RESULT.match(line)
        if m:
            return m.group(1).strip()
    return None


# ── Prompt readiness ─────────────────────────────────────────────────────

_RE_PROMPT_LINE = re.compile(r"^\s*❯")


def is_prompt_ready(pane_text: str) -> bool:
    """Check whether keystrokes sent to the pane will reach Claude's prompt.

    The input box closes with a bare ``─`` separator that has a ``❯`` line
    just above it and the status footer just below::

        ──────────────────────────────────── shiz ──
        ❯ catch me up on what's open
        ────────────────────────────────────────────
          ⏸ manual mode on · ? for shortcuts

    Dialogs draw over that region, keeping the borders but replacing the
    ``❯`` line and dropping the footer, so text typed into them never becomes
    a prompt.  Matching on the closing separator's neighbours tells the two
    apart; a menu's ``❯ 1. Yes`` has no separator under it.

    Returns True while Claude is mid-turn — the box stays visible and accepts
    queued input.
    """
    if not pane_text:
        return False

    tail = pane_text.rstrip().split("\n")[-12:]

    for i in range(len(tail) - 1, -1, -1):
        stripped = tail[i].strip()
        if len(stripped) < 20 or any(c != "─" for c in stripped):
            continue
        # Footer must follow, and the prompt line sits just above.
        if not any(line.strip() for line in tail[i + 1 :]):
            return False
        return any(_RE_PROMPT_LINE.match(line) for line in tail[max(0, i - 5) : i])
    return False


def is_blocking_dialog(pane_text: str) -> bool:
    """Check whether the pane shows a dialog that swallows input silently.

    True when the input box is gone *and* the screen matches no known
    interactive UI — so ccbot can neither forward the question nor deliver a
    message.  Known UIs are excluded: those get surfaced to the user, who
    answers them through the normal send path.

    A near-empty pane is a window still drawing its first frame, not a
    dialog.  Claude Code buffers keystrokes that arrive while it boots, so
    those sends go through untouched.
    """
    if len([line for line in pane_text.split("\n") if line.strip()]) < 3:
        return False
    return not is_prompt_ready(pane_text) and not is_interactive_ui(pane_text)


# ── Status line parsing ─────────────────────────────────────────────────

# Spinner characters Claude Code uses in its status line
STATUS_SPINNERS = frozenset(["·", "✻", "✽", "✶", "✳", "✢"])


def parse_status_line(pane_text: str) -> str | None:
    """Extract the Claude Code status line from terminal output.

    The status line (spinner + working text) appears immediately above
    the chrome separator (a full line of ``─`` characters).  We locate
    the separator first, then check the line just above it — this avoids
    false positives from ``·`` bullets in Claude's regular output.

    Returns the text after the spinner, or None if no status line found.
    """
    if not pane_text:
        return None

    lines = pane_text.split("\n")

    # Find the chrome separator: topmost ──── line in the last 10 lines
    chrome_idx: int | None = None
    search_start = max(0, len(lines) - 10)
    for i in range(search_start, len(lines)):
        stripped = lines[i].strip()
        if len(stripped) >= 20 and all(c == "─" for c in stripped):
            chrome_idx = i
            break

    if chrome_idx is None:
        return None  # No chrome visible — can't determine status

    # Scan upward for the spinner line. Claude Code can render a tip block
    # between the status line and the chrome, so keep looking past non-spinner
    # lines — but once we're past the line directly above the separator, only a
    # live status line counts. The ellipsis is what marks one, and it keeps
    # prose bullets and the finished marker ("✻ Sautéed for 7s") out.
    adjacent = True
    for i in range(chrome_idx - 1, max(chrome_idx - 9, -1), -1):
        line = lines[i].strip()
        if not line:
            continue
        if line[0] in STATUS_SPINNERS and (adjacent or "…" in line):
            return line[1:].strip()
        adjacent = False
    return None


# Claude Code keeps the spinner running while it executes Stop and SubagentStop
# hooks, which fire *after* the turn's final message is written. The optional
# middle word absorbs "subagent".
_RE_POST_TURN_HOOK = re.compile(r"running\s+(?:\w+\s+)?stop\s+hook", re.IGNORECASE)


def is_post_turn_status(status_line: str | None) -> bool:
    """Check whether a status line is a hook running after the turn ended.

    A spinner normally means a reply is still coming. During a post-turn hook
    it means the reply already arrived and Claude is finishing up, so callers
    that ask "is more content coming?" must treat this as no. Liveness checks
    want the opposite and should keep using parse_status_line.
    """
    return bool(status_line and _RE_POST_TURN_HOOK.search(status_line))


# ── Pane chrome stripping & bash output extraction ─────────────────────


def strip_pane_chrome(lines: list[str]) -> list[str]:
    """Strip Claude Code's bottom chrome (prompt area + status bar).

    The bottom of the pane looks like::

        ────────────────────────  (separator)
        ❯                        (prompt)
        ────────────────────────  (separator)
          [Opus 4.6] Context: 34%
          ⏵⏵ bypass permissions…

    This function finds the topmost ``────`` separator in the last 10 lines
    and strips everything from there down.
    """
    search_start = max(0, len(lines) - 10)
    for i in range(search_start, len(lines)):
        stripped = lines[i].strip()
        if len(stripped) >= 20 and all(c == "─" for c in stripped):
            return lines[:i]
    return lines


def extract_bash_output(pane_text: str, command: str) -> str | None:
    """Extract ``!`` command output from a captured tmux pane.

    Searches from the bottom for the ``! <command>`` echo line, then
    returns that line and everything below it (including the ``⎿`` output).
    Returns *None* if the command echo wasn't found.
    """
    lines = strip_pane_chrome(pane_text.splitlines())

    # Find the last "! <command>" echo line (search from bottom).
    # Match on the first 10 chars of the command in case the line is truncated.
    cmd_idx: int | None = None
    match_prefix = command[:10]
    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].strip()
        if stripped.startswith(f"! {match_prefix}") or stripped.startswith(
            f"!{match_prefix}"
        ):
            cmd_idx = i
            break

    if cmd_idx is None:
        return None

    # Include the command echo line and everything after it
    raw_output = lines[cmd_idx:]

    # Strip trailing empty lines
    while raw_output and not raw_output[-1].strip():
        raw_output.pop()

    if not raw_output:
        return None

    return "\n".join(raw_output).strip()


# ── Usage modal parsing ──────────────────────────────────────────────────────────


@dataclass
class UsageInfo:
    """Parsed output from Claude Code's /usage modal."""

    raw_text: str  # Full captured pane text
    parsed_lines: list[str]  # Cleaned content lines from the modal


def parse_usage_output(pane_text: str) -> UsageInfo | None:
    """Extract usage information from Claude Code's /usage settings tab.

    The /usage modal shows a Settings overlay with a "Usage" tab containing
    progress bars and reset times.  This parser looks for the Settings header
    line, then collects all content until "Esc to cancel".

    Returns UsageInfo with cleaned lines, or None if not detected.
    """
    if not pane_text:
        return None

    lines = pane_text.strip().split("\n")

    # Find the Settings header that indicates we're in the usage modal
    start_idx: int | None = None
    end_idx: int | None = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if start_idx is None:
            # The usage tab header line
            if "Settings:" in stripped and "Usage" in stripped:
                start_idx = i + 1  # skip the header itself
        else:
            if stripped.startswith("Esc to"):
                end_idx = i
                break

    if start_idx is None:
        return None
    if end_idx is None:
        end_idx = len(lines)

    # Collect content lines, stripping progress bar characters and whitespace
    cleaned: list[str] = []
    for line in lines[start_idx:end_idx]:
        # Strip the line but preserve meaningful content
        stripped = line.strip()
        if not stripped:
            continue
        # Remove progress bar block characters but keep the rest
        # Progress bars are like: █████▋   38% used
        # Strip leading block chars, keep the percentage
        stripped = re.sub(r"^[\u2580-\u259f\s]+", "", stripped).strip()
        if stripped:
            cleaned.append(stripped)

    if cleaned:
        return UsageInfo(raw_text=pane_text, parsed_lines=cleaned)

    return None
