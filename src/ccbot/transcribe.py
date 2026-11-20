"""Voice-to-text transcription, locally with parakeet-mlx or through OpenAI.

Local transcription is the default. It runs the parakeet-mlx CLI in a
subprocess, which reads the OGG through ffmpeg and writes the transcript
to a temp directory. Setting OPENAI_API_KEY sends voice to OpenAI's
gpt-4o-transcribe instead, over httpx (no OpenAI SDK needed).

Key function: transcribe_voice(ogg_data) -> str
"""

import asyncio
import logging
import os
import tempfile
from pathlib import Path

import httpx

from .config import config

logger = logging.getLogger(__name__)

PARAKEET_BIN = Path.home() / ".local" / "bin" / "parakeet-mlx"
PARAKEET_MODEL = "beshkenadze/parakeet-ultra-mlx-fp16"
PARAKEET_TIMEOUT_SECONDS = 180

# parakeet-mlx finds ffmpeg on PATH, which a launchd-started bot may lack
HOMEBREW_BIN = "/opt/homebrew/bin"

_client: httpx.AsyncClient | None = None


async def transcribe_voice(ogg_data: bytes) -> str:
    """Transcribe OGG voice data to text, through OpenAI when a key is set.

    Raises:
        ValueError: If the transcription comes back empty.
        RuntimeError: If parakeet-mlx writes no transcript or times out.
        httpx.HTTPStatusError: On OpenAI API errors (401, 429, 5xx, etc.)
    """
    if config.openai_api_key:
        text = await _transcribe_with_openai(ogg_data)
    else:
        text = await _transcribe_locally(ogg_data)

    text = text.strip()
    if not text:
        raise ValueError("Empty transcription")
    return text


async def _transcribe_with_openai(ogg_data: bytes) -> str:
    """Return OpenAI's transcript of the OGG data."""
    url = f"{config.openai_base_url.rstrip('/')}/audio/transcriptions"
    client = _get_client()
    response = await client.post(
        url,
        headers={"Authorization": f"Bearer {config.openai_api_key}"},
        files={"file": ("voice.ogg", ogg_data, "audio/ogg")},
        data={"model": "gpt-4o-transcribe"},
    )
    response.raise_for_status()
    return response.json().get("text", "")


async def _transcribe_locally(ogg_data: bytes) -> str:
    """Return parakeet-mlx's transcript of the OGG data."""
    with tempfile.TemporaryDirectory(prefix="ccbot-voice-") as temp_dir:
        audio_path = Path(temp_dir) / "voice.oga"
        audio_path.write_bytes(ogg_data)

        # Runs parakeet-mlx without blocking the event loop
        process = await asyncio.create_subprocess_exec(
            str(PARAKEET_BIN),
            "--model",
            PARAKEET_MODEL,
            "--output-format",
            "txt",
            "--output-dir",
            temp_dir,
            str(audio_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env={**os.environ, "PATH": f"{HOMEBREW_BIN}:{os.environ.get('PATH', '')}"},
        )
        try:
            output, _ = await asyncio.wait_for(
                process.communicate(), timeout=PARAKEET_TIMEOUT_SECONDS
            )
        except TimeoutError:
            process.kill()
            await process.wait()
            raise RuntimeError(
                f"parakeet-mlx gave no transcript within {PARAKEET_TIMEOUT_SECONDS}s"
            ) from None

        # Reads the transcript, which parakeet-mlx names after the input.
        # It exits 0 even when it fails, so a missing file is the failure.
        transcript_path = Path(temp_dir) / "voice.txt"
        if not transcript_path.exists():
            last_output = " ".join(output.decode(errors="replace").split())[-300:]
            raise RuntimeError(f"parakeet-mlx wrote no transcript: {last_output}")
        return transcript_path.read_text()


def _get_client() -> httpx.AsyncClient:
    """Return a lazily-initialized httpx client singleton."""
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=30.0)
    return _client


async def close_client() -> None:
    """Close the httpx client (call on shutdown)."""
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
        _client = None
