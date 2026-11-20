"""Tests for voice_handler — forwarding a transcript, or the voice file when it fails."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccbot.bot import voice_handler


def _make_update(ogg_data: bytes, thread_id: int = 42) -> MagicMock:
    """Build a minimal mock Update carrying a voice note in a forum topic."""
    voice_file = MagicMock()
    voice_file.file_path = "voice/file_7.oga"
    voice_file.download_as_bytearray = AsyncMock(return_value=bytearray(ogg_data))

    update = MagicMock()
    update.effective_user.id = 1
    update.message.message_thread_id = thread_id
    update.message.voice.file_unique_id = "AgADvoice"
    update.message.voice.get_file = AsyncMock(return_value=voice_file)
    update.message.chat.send_action = AsyncMock()
    return update


@pytest.fixture
def bot_env(tmp_path):
    """Patch the bot's session plumbing and point its media dir at tmp_path."""
    with (
        patch("ccbot.bot.is_user_allowed", return_value=True),
        patch("ccbot.bot._get_thread_id", return_value=42),
        patch("ccbot.bot.session_manager") as mock_sm,
        patch("ccbot.bot.tmux_manager") as mock_tmux,
        patch("ccbot.bot.safe_reply", new_callable=AsyncMock) as mock_reply,
        patch("ccbot.bot._MEDIA_DIR", tmp_path),
    ):
        mock_sm.get_window_for_thread.return_value = "@5"
        mock_sm.send_to_window = AsyncMock(return_value=(True, "ok"))
        mock_tmux.find_window_by_id = AsyncMock(return_value=MagicMock())
        yield {"sm": mock_sm, "reply": mock_reply, "media_dir": tmp_path}


class TestVoiceHandler:
    @pytest.mark.asyncio
    async def test_forwards_transcript(self, bot_env):
        with patch("ccbot.bot.transcribe_voice", AsyncMock(return_value="Hello there")):
            await voice_handler(_make_update(b"ogg"), MagicMock())

        bot_env["sm"].send_to_window.assert_called_once_with("@5", "Hello there")
        bot_env["reply"].assert_called_once()
        assert bot_env["reply"].call_args.args[1] == '🎤 "Hello there"'
        assert list(bot_env["media_dir"].iterdir()) == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error",
        [RuntimeError("parakeet-mlx wrote no transcript"), ValueError("Empty")],
    )
    async def test_failure_forwards_saved_voice_file(self, bot_env, error):
        with patch("ccbot.bot.transcribe_voice", AsyncMock(side_effect=error)):
            await voice_handler(_make_update(b"the-ogg-bytes"), MagicMock())

        [saved] = list(bot_env["media_dir"].iterdir())
        assert saved.name.endswith("_AgADvoice.oga")
        assert saved.read_bytes() == b"the-ogg-bytes"

        [(wid, text), _] = bot_env["sm"].send_to_window.call_args
        assert wid == "@5"
        assert str(saved) in text
        assert "transcription failed" in text

        bot_env["reply"].assert_called_once()
        assert "Couldn't transcribe" in bot_env["reply"].call_args.args[1]
