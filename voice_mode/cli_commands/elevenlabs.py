"""CLI commands for the ElevenLabs TTS provider.

Exposed under ``voicemode elevenlabs ...`` (and ``voice-mode elevenlabs ...``).
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Optional

import click

from voice_mode.elevenlabs_provider import (
    DEFAULT_MODEL_ID,
    DEFAULT_VOICE_ID,
    elevenlabs_text_to_speech,
    list_elevenlabs_voices,
)

# Resolved lazily so tests can patch the module attribute, and so a user
# can change `ELEVENLABS_API_KEY` between invocations.
ELEVENLABS_API_KEY: Optional[str] = os.getenv("ELEVENLABS_API_KEY")


def _refresh_api_key() -> Optional[str]:
    """Re-read the env var (tests may patch the module attribute directly)."""
    return ELEVENLABS_API_KEY if ELEVENLABS_API_KEY is not None else os.getenv("ELEVENLABS_API_KEY")


@click.group()
def elevenlabs() -> None:
    """ElevenLabs TTS provider — list voices, set credentials, smoke-test."""


@elevenlabs.command("list-voices")
def list_voices_cmd() -> None:
    """List voices available to the configured ELEVENLABS_API_KEY."""
    api_key = _refresh_api_key()
    if not api_key:
        click.echo(
            "ELEVENLABS_API_KEY is not set. Export it or run "
            "`voicemode elevenlabs set-api-key`.",
            err=True,
        )
        sys.exit(2)

    try:
        voices = asyncio.run(list_elevenlabs_voices(api_key))
    except ImportError as exc:
        click.echo(f"ElevenLabs SDK missing: {exc}", err=True)
        click.echo("Install with: pip install voice-mode[elevenlabs]", err=True)
        sys.exit(3)
    except Exception as exc:
        click.echo(f"ElevenLabs request failed: {exc}", err=True)
        sys.exit(1)

    if not voices:
        click.echo("(no voices returned)")
        return
    width = max(len(v.get("voice_id") or "") for v in voices)
    for v in voices:
        click.echo(
            f"{(v.get('voice_id') or '').ljust(width)}  "
            f"{v.get('name') or '<unnamed>'}  "
            f"[{v.get('category') or '?'}]"
        )


@elevenlabs.command("set-api-key")
@click.argument("api_key")
def set_api_key_cmd(api_key: str) -> None:
    """Save ELEVENLABS_API_KEY to the voicemode credential store."""
    from voice_mode.credential_store import get_credential_store

    store = get_credential_store()
    data = store.load() or {}
    data["ELEVENLABS_API_KEY"] = api_key
    store.save(data)
    click.echo(f"Saved ELEVENLABS_API_KEY to {store.name} credential store.")


@elevenlabs.command("test")
@click.argument("text", required=False, default="Hello from ElevenLabs via voicemode.")
@click.option(
    "--voice-id",
    default=None,
    help=f"ElevenLabs voice ID (default: {DEFAULT_VOICE_ID}).",
)
@click.option(
    "--model-id",
    default=None,
    help=f"ElevenLabs model ID (default: {DEFAULT_MODEL_ID}).",
)
def test_cmd(text: str, voice_id: Optional[str], model_id: Optional[str]) -> None:
    """Synthesize TEXT with ElevenLabs and play it through the default output."""
    api_key = _refresh_api_key()
    if not api_key:
        click.echo("ELEVENLABS_API_KEY is not set.", err=True)
        sys.exit(2)

    success, metrics = asyncio.run(
        elevenlabs_text_to_speech(
            text=text,
            voice_id=voice_id,
            model_id=model_id,
            api_key=api_key,
        )
    )
    if not success:
        err = (metrics or {}).get("error", "unknown error")
        click.echo(f"FAILED: {err}", err=True)
        sys.exit(1)
    click.echo(
        f"OK — ttfa={metrics.get('ttfa', 0):.2f}s "
        f"gen={metrics.get('generation', 0):.2f}s "
        f"playback={metrics.get('playback', 0):.2f}s"
    )
