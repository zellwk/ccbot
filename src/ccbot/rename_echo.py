"""Topic renames this bot made itself, whose service message is noise.

Telegram posts a "changed the topic name" service message for every
``editForumTopic`` call, including ccbot's own — a liveness probe re-sending
the name a topic already has, and an auto-name landing on a new topic.
Registering the rename here lets the topic-edited handler delete the echo
instead of showing a rename nobody made.
"""

# (chat_id, thread_id, name) of renames awaiting their service message.
_expected: set[tuple[int, int, str]] = set()


def expect_rename_echo(chat_id: int, thread_id: int, name: str) -> None:
    """Mark a rename this bot just made, so its service message gets deleted."""
    _expected.add((chat_id, thread_id, name))


def consume_rename_echo(chat_id: int, thread_id: int, name: str) -> bool:
    """Report whether this rename was the bot's own, and forget it."""
    key = (chat_id, thread_id, name)
    if key not in _expected:
        return False
    _expected.discard(key)
    return True
