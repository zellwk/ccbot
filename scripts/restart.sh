#!/usr/bin/env bash
set -euo pipefail

# Restart the supervised ccbot instance.
#
# On the Mac mini, ccbot is owned by launchd (com.zellwk.ccbot), which runs
# ccbot-supervise.sh every 5 minutes.  That script keeps one instance alive in
# tmux session `ccbot-host`, started from the editable install at
# ~/.local/bin/ccbot.  Telegram allows exactly one getUpdates poller per token,
# so a second copy started any other way duplicates every reply and fills the
# log with Conflict tracebacks.  Stop what is running, then hand the start back
# to the supervise script.
#
# Do not kill on the pattern "bin/ccbot" — that also matches tmux command lines
# in session `ccbot`, which hold live Claude sessions.

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SUPERVISE="$HOME/projects/@zellwk/ai-init/context/claude/channels/telegram/ccbot-supervise.sh"
HOST_SESSION="ccbot-host"
TOOL_PYTHON="$HOME/.local/share/uv/tools/ccbot/bin/python"
MAX_WAIT=10  # seconds to wait for a process to exit

# Patterns are anchored to the start of the command line so a shell that merely
# mentions a path (a grep, an ssh one-liner) cannot masquerade as a running bot.
supervised_pids() { pgrep -f "^$TOOL_PYTHON" 2>/dev/null || true; }
stray_pids() { pgrep -f "^uv run ccbot|^$PROJECT_DIR/.venv/bin/python" 2>/dev/null || true; }

# A copy started from the project venv is never the supervised one — it is the
# duplicate poller.  Kill it outright.
strays="$(stray_pids)"
if [ -n "$strays" ]; then
    echo "Killing stray ccbot instance(s): $strays"
    # shellcheck disable=SC2086
    kill $strays 2>/dev/null || true
    sleep 2
    strays="$(stray_pids)"
    if [ -n "$strays" ]; then
        # shellcheck disable=SC2086
        kill -9 $strays 2>/dev/null || true
    fi
fi

# Stop the supervised instance.  It ignores SIGTERM, so interrupt it inside its
# own tmux session and escalate only if that fails.
if [ -n "$(supervised_pids)" ]; then
    echo "Stopping supervised ccbot in tmux session '$HOST_SESSION'..."
    if tmux has-session -t "$HOST_SESSION" 2>/dev/null; then
        tmux send-keys -t "$HOST_SESSION" C-c
    fi

    waited=0
    while [ -n "$(supervised_pids)" ] && [ "$waited" -lt "$MAX_WAIT" ]; do
        sleep 1
        waited=$((waited + 1))
        echo "  Waiting for process to exit... (${waited}s/${MAX_WAIT}s)"
    done

    if [ -n "$(supervised_pids)" ]; then
        echo "Still running after ${MAX_WAIT}s, sending SIGKILL..."
        # shellcheck disable=SC2086
        kill -9 $(supervised_pids) 2>/dev/null || true
        sleep 2
    fi
    echo "Process stopped."
else
    echo "No supervised ccbot process running."
fi

# Start it the same way launchd does, so both paths stay identical.
echo "Starting ccbot..."
if [ -x "$SUPERVISE" ]; then
    "$SUPERVISE"
else
    echo "Supervise script not found at $SUPERVISE, starting directly."
    tmux kill-session -t "$HOST_SESSION" 2>/dev/null || true
    tmux new-session -d -s "$HOST_SESSION" -c "$HOME/projects" \
        'cd ~/projects && ~/.local/bin/ccbot >> /tmp/ccbot.log 2>&1'
fi

sleep 5
if [ -n "$(supervised_pids)" ]; then
    echo "ccbot restarted successfully (pid $(supervised_pids)). Recent logs:"
    echo "----------------------------------------"
    tail -20 /tmp/ccbot.log
    echo "----------------------------------------"
else
    echo "Warning: ccbot did not start. Recent logs:"
    echo "----------------------------------------"
    tail -30 /tmp/ccbot.log
    echo "----------------------------------------"
    exit 1
fi
