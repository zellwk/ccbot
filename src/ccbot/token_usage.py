"""Token accounting for a Claude session, read from its transcript on disk.

Claude Code appends one JSONL line per message under ``~/.claude/projects``,
each assistant line carrying a ``usage`` block. The newest block's input side
is the live context window; output tokens accumulate over the session.

Reading the file costs nothing on the Claude side — the session is never woken.
"""

import json
from pathlib import Path

TRANSCRIPT_ROOT = Path.home() / ".claude" / "projects"


def find_transcript(session_id: str) -> Path | None:
    """Locate a session's JSONL transcript across all project slugs."""
    if not session_id:
        return None
    matches = sorted(
        TRANSCRIPT_ROOT.glob(f"*/{session_id}.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def read_usage(session_id: str) -> dict[str, int] | None:
    """Sum token usage for a session. Returns None when no transcript exists."""
    path = find_transcript(session_id)
    if not path:
        return None

    context = 0
    output_total = 0
    turns = 0
    compactions = 0

    with path.open(errors="replace") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if entry.get("isCompactSummary") or entry.get("type") == "compact_boundary":
                compactions += 1
            usage = (entry.get("message") or {}).get("usage")
            if not usage:
                continue
            turns += 1
            output_total += usage.get("output_tokens", 0)
            context = (
                usage.get("input_tokens", 0)
                + usage.get("cache_read_input_tokens", 0)
                + usage.get("cache_creation_input_tokens", 0)
            )

    return {
        "context": context,
        "output_total": output_total,
        "turns": turns,
        "compactions": compactions,
    }
