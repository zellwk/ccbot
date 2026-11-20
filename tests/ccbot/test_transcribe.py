"""Unit tests for transcribe — voice-to-text via local parakeet-mlx or OpenAI."""

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from ccbot import transcribe


@pytest.fixture(autouse=True)
def _reset_client():
    """Ensure each test starts with a fresh client."""
    transcribe._client = None
    yield
    transcribe._client = None


@pytest.fixture
def mock_config():
    """Patch config with test values."""
    with patch.object(transcribe, "config") as cfg:
        cfg.openai_api_key = "sk-test-key"
        cfg.openai_base_url = "https://api.openai.com/v1"
        yield cfg


def _mock_response(*, json_data: dict, status_code: int = 200) -> httpx.Response:
    """Build a fake httpx.Response."""
    request = httpx.Request("POST", "https://api.openai.com/v1/audio/transcriptions")
    resp = httpx.Response(status_code=status_code, json=json_data, request=request)
    return resp


class TestTranscribeWithOpenAI:
    @pytest.mark.asyncio
    async def test_success(self, mock_config):
        resp = _mock_response(json_data={"text": "Hello world"})
        with patch.object(
            httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=resp
        ) as mock_post:
            result = await transcribe.transcribe_voice(b"fake-ogg-data")

        assert result == "Hello world"
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert "Bearer sk-test-key" in str(call_kwargs)

    @pytest.mark.asyncio
    async def test_empty_transcription_raises(self, mock_config):
        resp = _mock_response(json_data={"text": ""})
        with patch.object(
            httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=resp
        ):
            with pytest.raises(ValueError, match="Empty transcription"):
                await transcribe.transcribe_voice(b"fake-ogg-data")

    @pytest.mark.asyncio
    async def test_whitespace_only_raises(self, mock_config):
        resp = _mock_response(json_data={"text": "   "})
        with patch.object(
            httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=resp
        ):
            with pytest.raises(ValueError, match="Empty transcription"):
                await transcribe.transcribe_voice(b"fake-ogg-data")

    @pytest.mark.asyncio
    async def test_missing_text_field_raises(self, mock_config):
        resp = _mock_response(json_data={"result": "something"})
        with patch.object(
            httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=resp
        ):
            with pytest.raises(ValueError, match="Empty transcription"):
                await transcribe.transcribe_voice(b"fake-ogg-data")

    @pytest.mark.asyncio
    async def test_api_error_raises(self, mock_config):
        resp = _mock_response(json_data={"error": "Unauthorized"}, status_code=401)
        with patch.object(
            httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=resp
        ):
            with pytest.raises(httpx.HTTPStatusError):
                await transcribe.transcribe_voice(b"fake-ogg-data")

    @pytest.mark.asyncio
    async def test_custom_base_url(self, mock_config):
        mock_config.openai_base_url = "https://proxy.example.com/v1"
        resp = _mock_response(json_data={"text": "Transcribed"})
        with patch.object(
            httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=resp
        ) as mock_post:
            result = await transcribe.transcribe_voice(b"fake-ogg-data")

        assert result == "Transcribed"
        url_arg = mock_post.call_args[0][0]
        assert url_arg == "https://proxy.example.com/v1/audio/transcriptions"

    @pytest.mark.asyncio
    async def test_base_url_trailing_slash_stripped(self, mock_config):
        mock_config.openai_base_url = "https://proxy.example.com/v1/"
        resp = _mock_response(json_data={"text": "OK"})
        with patch.object(
            httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=resp
        ) as mock_post:
            await transcribe.transcribe_voice(b"fake-ogg-data")

        url_arg = mock_post.call_args[0][0]
        assert url_arg == "https://proxy.example.com/v1/audio/transcriptions"


@pytest.fixture
def local_config():
    """Patch config with no OpenAI key, so transcription runs locally."""
    with patch.object(transcribe, "config") as cfg:
        cfg.openai_api_key = ""
        yield cfg


@pytest.fixture
def fake_parakeet(tmp_path, monkeypatch):
    """Point PARAKEET_BIN at a shell script that stands in for parakeet-mlx.

    The script gets parakeet's argv, so "$6" is the output dir. The test
    supplies the script body.
    """

    def _install(body: str) -> None:
        script = tmp_path / "parakeet-mlx"
        script.write_text(f"#!/bin/sh\n{body}\n")
        script.chmod(0o755)
        monkeypatch.setattr(transcribe, "PARAKEET_BIN", script)

    return _install


class TestTranscribeLocally:
    @pytest.mark.asyncio
    async def test_returns_transcript_parakeet_writes(
        self, local_config, fake_parakeet
    ):
        fake_parakeet('printf "  Hello from parakeet\\n" > "$6/voice.txt"')

        result = await transcribe.transcribe_voice(b"fake-ogg-data")

        assert result == "Hello from parakeet"

    @pytest.mark.asyncio
    async def test_passes_audio_bytes_to_parakeet(self, local_config, fake_parakeet):
        fake_parakeet('cp "$7" "$6/voice.txt"')

        result = await transcribe.transcribe_voice(b"the-ogg-bytes")

        assert result == "the-ogg-bytes"

    @pytest.mark.asyncio
    async def test_puts_homebrew_on_path_for_ffmpeg(
        self, local_config, fake_parakeet, monkeypatch
    ):
        monkeypatch.setenv("PATH", "/usr/bin:/bin")
        fake_parakeet('printf "%s" "$PATH" > "$6/voice.txt"')

        result = await transcribe.transcribe_voice(b"fake-ogg-data")

        assert result.split(":")[0] == "/opt/homebrew/bin"

    @pytest.mark.asyncio
    async def test_missing_transcript_raises_with_output(
        self, local_config, fake_parakeet
    ):
        # parakeet-mlx prints its error and still exits 0
        fake_parakeet('echo "Error transcribing file: FFmpeg is not installed"')

        with pytest.raises(RuntimeError, match="FFmpeg is not installed"):
            await transcribe.transcribe_voice(b"fake-ogg-data")

    @pytest.mark.asyncio
    async def test_empty_transcript_raises(self, local_config, fake_parakeet):
        fake_parakeet('printf "  \\n" > "$6/voice.txt"')

        with pytest.raises(ValueError, match="Empty transcription"):
            await transcribe.transcribe_voice(b"fake-ogg-data")

    @pytest.mark.asyncio
    async def test_removes_temp_dir(self, local_config, fake_parakeet):
        fake_parakeet('printf "done" > "$6/voice.txt"')
        temp_root = Path(tempfile.gettempdir())
        before = set(temp_root.glob("ccbot-voice-*"))

        await transcribe.transcribe_voice(b"fake-ogg-data")

        assert set(temp_root.glob("ccbot-voice-*")) == before

    @pytest.mark.asyncio
    async def test_hung_parakeet_is_killed_and_raises(
        self, local_config, fake_parakeet, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(transcribe, "PARAKEET_TIMEOUT_SECONDS", 0.5)
        pid_file = tmp_path / "pid"
        fake_parakeet(f'echo $$ > "{pid_file}"; exec sleep 5')

        with pytest.raises(RuntimeError, match="no transcript within"):
            await transcribe.transcribe_voice(b"fake-ogg-data")

        with pytest.raises(ProcessLookupError):
            os.kill(int(pid_file.read_text()), 0)


class TestCloseClient:
    @pytest.mark.asyncio
    async def test_close_client_when_open(self):
        transcribe._client = httpx.AsyncClient()
        assert transcribe._client is not None
        await transcribe.close_client()
        assert transcribe._client is None

    @pytest.mark.asyncio
    async def test_close_client_when_none(self):
        assert transcribe._client is None
        await transcribe.close_client()
        assert transcribe._client is None
