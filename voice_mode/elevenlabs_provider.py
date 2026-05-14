"""ElevenLabs TTS provider for voicemode.

This module is the bridge between voicemode's failover loop
(`voice_mode.simple_failover.simple_tts_failover`) and the official
ElevenLabs Python SDK (`AsyncElevenLabs`).

Why a dedicated module: voicemode's existing TTS path assumes an
OpenAI-compatible HTTP API (Kokoro, mlx-audio, OpenAI itself). ElevenLabs
uses a different request/response shape and auth header, so it cannot be
served by `AsyncOpenAI`.

The `elevenlabs` package is an **optional** dependency; users opt in via
`pip install voice-mode[elevenlabs]`. We lazy-import the SDK so the rest of
voicemode can import this module without paying the dependency cost.
"""

from __future__ import annotations

import io
import logging
import tempfile
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

logger = logging.getLogger("voicemode")


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # George (ElevenLabs SDK example voice)
DEFAULT_MODEL_ID = "eleven_turbo_v2_5"
DEFAULT_OUTPUT_FORMAT = "mp3_44100_128"


# ---------------------------------------------------------------------------
# Voice ID mapping across providers
#
# Best-effort approximations so that a user's preferred voice survives a
# failover from ElevenLabs -> OpenAI / Kokoro and vice versa. Keys are
# ElevenLabs voice IDs (premade voices, stable IDs).
# ---------------------------------------------------------------------------

ELEVENLABS_TO_OPENAI: Dict[str, str] = {
    "JBFqnCBsd6RMkjVDRZzb": "onyx",    # George (male, deep) -> onyx
    "21m00Tcm4TlvDq8ikWAM": "nova",    # Rachel (female) -> nova
    "AZnzlk1XvdvUeBnXmlld": "shimmer", # Domi (female, strong) -> shimmer
    "EXAVITQu4vr4xnSDxMaL": "shimmer", # Bella (female) -> shimmer
    "ErXwobaYiN019PkySvjV": "onyx",    # Antoni (male) -> onyx
    "MF3mGyEYCl7XYWbV9V6O": "alloy",   # Elli (female) -> alloy
    "TxGEqnHWrfWFTfGW9XjX": "echo",    # Josh (male) -> echo
    "VR6AewLTigWG4xSOukaG": "fable",   # Arnold (male) -> fable
    "pNInz6obpgDQGcFmaJgB": "onyx",    # Adam (male) -> onyx
    "yoZ06aMxZJJ28mfd3POQ": "echo",    # Sam (male) -> echo
}

ELEVENLABS_TO_KOKORO: Dict[str, str] = {
    "JBFqnCBsd6RMkjVDRZzb": "bm_george",
    "21m00Tcm4TlvDq8ikWAM": "af_sky",
    "AZnzlk1XvdvUeBnXmlld": "af_bella",
    "EXAVITQu4vr4xnSDxMaL": "af_bella",
    "ErXwobaYiN019PkySvjV": "am_adam",
    "MF3mGyEYCl7XYWbV9V6O": "af_alloy",
    "TxGEqnHWrfWFTfGW9XjX": "am_echo",
    "VR6AewLTigWG4xSOukaG": "bm_fable",
    "pNInz6obpgDQGcFmaJgB": "am_adam",
    "yoZ06aMxZJJ28mfd3POQ": "am_echo",
}

# Reverse maps (built once, used both directions).
OPENAI_TO_ELEVENLABS: Dict[str, str] = {
    "alloy":   "MF3mGyEYCl7XYWbV9V6O",  # Elli
    "echo":    "TxGEqnHWrfWFTfGW9XjX",  # Josh
    "fable":   "VR6AewLTigWG4xSOukaG",  # Arnold
    "nova":    "21m00Tcm4TlvDq8ikWAM",  # Rachel
    "onyx":    "pNInz6obpgDQGcFmaJgB",  # Adam
    "shimmer": "EXAVITQu4vr4xnSDxMaL",  # Bella
}

KOKORO_TO_ELEVENLABS: Dict[str, str] = {
    "af_sky":     "21m00Tcm4TlvDq8ikWAM",
    "af_sarah":   "21m00Tcm4TlvDq8ikWAM",
    "af_alloy":   "MF3mGyEYCl7XYWbV9V6O",
    "af_bella":   "EXAVITQu4vr4xnSDxMaL",
    "am_adam":    "pNInz6obpgDQGcFmaJgB",
    "am_echo":    "TxGEqnHWrfWFTfGW9XjX",
    "am_onyx":    "pNInz6obpgDQGcFmaJgB",
    "bm_george":  "JBFqnCBsd6RMkjVDRZzb",
    "bm_fable":   "VR6AewLTigWG4xSOukaG",
}


def map_voice_to_openai(voice: str) -> str:
    """Return the OpenAI voice closest to `voice` (an ElevenLabs voice ID).

    Falls back to ``alloy`` when no mapping is known.
    """
    return ELEVENLABS_TO_OPENAI.get(voice, "alloy")


def map_voice_to_kokoro(voice: str) -> str:
    """Return the Kokoro voice closest to `voice` (an ElevenLabs voice ID).

    Falls back to ``af_sky`` (the project default) when no mapping is known.
    """
    return ELEVENLABS_TO_KOKORO.get(voice, "af_sky")


def map_voice_to_elevenlabs(voice: str) -> str:
    """Return the ElevenLabs voice ID closest to `voice`.

    Accepts:
    - an existing ElevenLabs voice ID (passes through unchanged),
    - an OpenAI voice name (alloy / echo / fable / nova / onyx / shimmer),
    - a Kokoro voice name (af_sky, am_adam, ...).

    Falls back to ``DEFAULT_VOICE_ID`` (George) when no mapping is known.
    """
    if not voice:
        return DEFAULT_VOICE_ID
    # Already an ElevenLabs ID?
    if voice in ELEVENLABS_TO_OPENAI:
        return voice
    if voice in OPENAI_TO_ELEVENLABS:
        return OPENAI_TO_ELEVENLABS[voice]
    if voice in KOKORO_TO_ELEVENLABS:
        return KOKORO_TO_ELEVENLABS[voice]
    return DEFAULT_VOICE_ID


# ---------------------------------------------------------------------------
# Lazy SDK access (so importing voice_mode.elevenlabs_provider does not
# require the optional `elevenlabs` package to be installed).
# ---------------------------------------------------------------------------


def _import_async_elevenlabs():
    """Import and return ``elevenlabs.AsyncElevenLabs``.

    Isolated into its own function so tests can patch it to simulate the
    SDK being absent.
    """
    from elevenlabs import AsyncElevenLabs  # type: ignore[import-not-found]
    return AsyncElevenLabs


def _make_async_client(api_key: str):
    """Build an ``AsyncElevenLabs`` client. Separated for test patching."""
    AsyncElevenLabs = _import_async_elevenlabs()
    return AsyncElevenLabs(api_key=api_key)


_MISSING_SDK_HINT = (
    "ElevenLabs SDK not installed. Run `pip install voice-mode[elevenlabs]` "
    "(or `uv pip install -e \".[elevenlabs]\"` for editable installs)."
)


# ---------------------------------------------------------------------------
# Audio playback
# ---------------------------------------------------------------------------


async def _play_audio_bytes(audio_bytes: bytes, fmt: str = "mp3") -> bool:  # pragma: no cover - exercises the real audio device; covered by manual smoke test
    """Play `audio_bytes` (full clip) through the default output device.

    Uses pydub for decoding and sounddevice for playback, mirroring the
    buffered branch in `voice_mode.core.text_to_speech`. Returns False if
    playback can't be initialised (no audio device, etc.) instead of
    raising, so callers can treat it as a failover-worthy failure.
    """
    try:
        import numpy as np
        import sounddevice as sd
        from pydub import AudioSegment
    except Exception as exc:
        logger.error(f"Audio stack unavailable: {exc}")
        return False

    try:
        with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            audio = AudioSegment.from_file(tmp.name, format=fmt)

        samples = np.array(audio.get_array_of_samples())
        if audio.channels == 2:
            samples = samples.reshape((-1, 2))
        samples = samples.astype(np.float32) / 32767.0

        sd.default.samplerate = audio.frame_rate
        sd.default.channels = audio.channels
        sd.play(samples, audio.frame_rate)
        sd.wait()
        return True
    except Exception as exc:
        logger.error(f"ElevenLabs audio playback failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def elevenlabs_text_to_speech(
    text: str,
    voice_id: Optional[str] = None,
    model_id: Optional[str] = None,
    api_key: Optional[str] = None,
    output_format: str = DEFAULT_OUTPUT_FORMAT,
    **_unused: Any,
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Synthesize ``text`` with ElevenLabs and play the audio.

    Mirrors ``voice_mode.core.text_to_speech``'s return signature so the
    failover loop can dispatch to either path uniformly.

    Returns:
        (success, metrics) where metrics is a dict with keys
        ``generation``, ``ttfa``, ``playback`` on success, or
        ``{"error": "..."}`` on failure.
    """
    metrics: Dict[str, Any] = {}

    if not api_key:
        return False, {"error": "ELEVENLABS_API_KEY is missing"}

    voice_id = voice_id or DEFAULT_VOICE_ID
    model_id = model_id or DEFAULT_MODEL_ID

    logger.info(
        "ElevenLabs TTS: voice_id=%s model_id=%s text_preview=%r",
        voice_id,
        model_id,
        text[:80],
    )

    try:
        client = _make_async_client(api_key)
    except ImportError as exc:
        logger.error(f"ElevenLabs SDK import failed: {exc}")
        return False, {"error": _MISSING_SDK_HINT}
    except Exception as exc:
        logger.error(f"ElevenLabs client init failed: {exc}")
        return False, {"error": f"ElevenLabs client init failed: {exc}"}

    generation_start = time.perf_counter()
    try:
        stream = client.text_to_speech.stream(
            voice_id=voice_id,
            model_id=model_id,
            output_format=output_format,
            text=text,
        )
        buffer = io.BytesIO()
        ttfa_recorded = False
        ttfa = 0.0
        async for chunk in stream:
            if not ttfa_recorded:
                ttfa = time.perf_counter() - generation_start
                metrics["ttfa"] = ttfa
                ttfa_recorded = True
            buffer.write(chunk)
        audio_bytes = buffer.getvalue()
        metrics["generation"] = time.perf_counter() - generation_start
    except Exception as exc:
        logger.error(f"ElevenLabs TTS request failed: {exc}")
        return False, {"error": str(exc)}

    if not audio_bytes:
        return False, {"error": "ElevenLabs returned no audio"}

    playback_fmt = "mp3" if output_format.startswith("mp3") else "wav"
    playback_start = time.perf_counter()
    played = await _play_audio_bytes(audio_bytes, fmt=playback_fmt)
    metrics["playback"] = time.perf_counter() - playback_start
    metrics["audio_bytes"] = len(audio_bytes)
    return played, metrics


async def list_elevenlabs_voices(api_key: str) -> List[Dict[str, Any]]:
    """Return the list of voices available to ``api_key``.

    Each entry: ``{"voice_id", "name", "category"}``. Raises ImportError
    (via :func:`_import_async_elevenlabs`) if the SDK is missing.
    """
    client = _make_async_client(api_key)
    response = await client.voices.search()
    voices_attr = getattr(response, "voices", response)
    out: List[Dict[str, Any]] = []
    for v in voices_attr:
        out.append(
            {
                "voice_id": getattr(v, "voice_id", None) or getattr(v, "id", None),
                "name": getattr(v, "name", None),
                "category": getattr(v, "category", None),
            }
        )
    return out
