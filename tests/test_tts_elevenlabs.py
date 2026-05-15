"""Tests for ElevenLabs TTS provider integration.

Covers:
- Provider type detection for ElevenLabs URLs.
- Voice ID mapping across providers (ElevenLabs <-> OpenAI/Kokoro).
- `elevenlabs_text_to_speech` happy path (mocked SDK, no network).
- Clear error when the optional `elevenlabs` SDK is missing.
- `simple_tts_failover` dispatches to the ElevenLabs path and falls through
  to OpenAI when ElevenLabs fails.
- CLI `voicemode elevenlabs list-voices` invokes the helper.

All tests mock at function/SDK level; none hit the network.
"""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner


# ---------------------------------------------------------------------------
# 1) Provider detection
# ---------------------------------------------------------------------------


class TestProviderDetection:
    def test_detect_elevenlabs_url(self):
        from voice_mode.provider_discovery import detect_provider_type

        assert detect_provider_type("https://api.elevenlabs.io/v1") == "elevenlabs"

    def test_detect_elevenlabs_url_no_trailing_path(self):
        from voice_mode.provider_discovery import detect_provider_type

        assert detect_provider_type("https://api.elevenlabs.io") == "elevenlabs"

    def test_detect_openai_still_works(self):
        from voice_mode.provider_discovery import detect_provider_type

        assert detect_provider_type("https://api.openai.com/v1") == "openai"


# ---------------------------------------------------------------------------
# 2) Voice mapping
# ---------------------------------------------------------------------------


class TestVoiceMapping:
    def test_known_elevenlabs_voice_maps_to_openai(self):
        from voice_mode.elevenlabs_provider import map_voice_to_openai

        # Rachel -> nova is a documented mapping.
        assert map_voice_to_openai("21m00Tcm4TlvDq8ikWAM") == "nova"

    def test_known_openai_voice_maps_to_elevenlabs(self):
        from voice_mode.elevenlabs_provider import map_voice_to_elevenlabs

        # OpenAI 'nova' should land on a real ElevenLabs voice ID.
        eleven_id = map_voice_to_elevenlabs("nova")
        assert isinstance(eleven_id, str)
        assert len(eleven_id) >= 10  # ElevenLabs voice IDs are 20-char alnum

    def test_unknown_voice_falls_back_to_default(self):
        from voice_mode.elevenlabs_provider import (
            map_voice_to_elevenlabs,
            DEFAULT_VOICE_ID,
        )

        assert map_voice_to_elevenlabs("definitely-not-a-voice") == DEFAULT_VOICE_ID

    def test_unseeded_elevenlabs_id_passes_through(self):
        """An ElevenLabs voice ID not in the static seed list (e.g. a
        professional or cloned voice) must pass through unchanged instead
        of being silently rewritten to George."""
        from voice_mode.elevenlabs_provider import map_voice_to_elevenlabs

        # 20-char alphanumeric IDs taken from ElevenLabs' professional
        # voice catalog (Ava, Charlotte, Shelby) — not seed-listed.
        for unseeded in (
            "gJx1vCzNCD1EQHT212Ls",
            "uhYnkYTBc711oAY590Ea",
            "rfkTsdZrVWEVhDycUYn9",
        ):
            assert map_voice_to_elevenlabs(unseeded) == unseeded

    def test_malformed_id_still_falls_back(self):
        """Strings that look ID-ish but don't match the 20-char alnum shape
        must NOT pass through — they fall back to DEFAULT_VOICE_ID. Protects
        against silently sending obvious garbage to the API."""
        from voice_mode.elevenlabs_provider import (
            map_voice_to_elevenlabs,
            DEFAULT_VOICE_ID,
        )

        for bad in (
            "too-short",                   # has hyphens, wrong length
            "has_underscores_inside",      # underscores are not alnum
            "JBFqnCBsd6RMkjVDRZ",          # 18 chars — too short
            "JBFqnCBsd6RMkjVDRZzbXX",      # 22 chars — too long
            "JBFqn CBsd6RMkjVDRZzb",       # contains a space
        ):
            assert map_voice_to_elevenlabs(bad) == DEFAULT_VOICE_ID

    def test_env_var_overrides_default_voice(self, monkeypatch):
        """VOICEMODE_ELEVENLABS_DEFAULT_VOICE must drive the fallback. We
        patch the module-level constant directly because it is evaluated
        from the env at import time."""
        from voice_mode import elevenlabs_provider

        monkeypatch.setattr(
            elevenlabs_provider, "DEFAULT_VOICE_ID", "OverrideVoiceID2026"
        )
        # Unknown voice -> the patched default, not George.
        assert (
            elevenlabs_provider.map_voice_to_elevenlabs("nope-no-such-voice")
            == "OverrideVoiceID2026"
        )
        # Empty voice -> the patched default too.
        assert elevenlabs_provider.map_voice_to_elevenlabs("") == "OverrideVoiceID2026"


# ---------------------------------------------------------------------------
# 3) elevenlabs_text_to_speech happy path
# ---------------------------------------------------------------------------


class TestElevenLabsTextToSpeech:
    @pytest.mark.asyncio
    async def test_streams_audio_and_returns_metrics(self):
        """elevenlabs_text_to_speech should call AsyncElevenLabs.text_to_speech.stream,
        write audio bytes to the player path, and return (True, metrics)."""
        from voice_mode import elevenlabs_provider

        async def _fake_stream(**kwargs):
            for chunk in (b"\x00\x01", b"\x02\x03"):
                yield chunk

        fake_client = MagicMock()
        fake_client.text_to_speech.stream = MagicMock(side_effect=_fake_stream)

        with patch.object(
            elevenlabs_provider, "_make_async_client", return_value=fake_client
        ), patch.object(
            elevenlabs_provider, "_play_audio_bytes", new=AsyncMock(return_value=True)
        ):
            success, metrics = await elevenlabs_provider.elevenlabs_text_to_speech(
                text="hello world",
                voice_id="JBFqnCBsd6RMkjVDRZzb",
                model_id="eleven_turbo_v2_5",
                api_key="test-key",
            )

        assert success is True
        assert metrics is not None
        assert "generation" in metrics
        assert metrics["generation"] >= 0
        fake_client.text_to_speech.stream.assert_called_once()
        call_kwargs = fake_client.text_to_speech.stream.call_args.kwargs
        assert call_kwargs.get("voice_id") == "JBFqnCBsd6RMkjVDRZzb"
        assert call_kwargs.get("model_id") == "eleven_turbo_v2_5"
        assert call_kwargs.get("text") == "hello world"

    @pytest.mark.asyncio
    async def test_missing_api_key_returns_failure(self):
        from voice_mode import elevenlabs_provider

        success, metrics = await elevenlabs_provider.elevenlabs_text_to_speech(
            text="hi", voice_id="x", model_id="y", api_key=None,
        )
        assert success is False
        assert metrics is not None
        assert "missing" in (metrics.get("error") or "").lower()


# ---------------------------------------------------------------------------
# 4) Missing SDK -> clear error
# ---------------------------------------------------------------------------


class TestMissingSDK:
    @pytest.mark.asyncio
    async def test_clear_error_when_sdk_not_installed(self):
        """When the optional `elevenlabs` package is absent, calling
        elevenlabs_text_to_speech must raise (or return) a message that
        directs the user to `pip install voice-mode[elevenlabs]`."""
        from voice_mode import elevenlabs_provider

        # Simulate ImportError on AsyncElevenLabs resolution.
        with patch.object(
            elevenlabs_provider,
            "_import_async_elevenlabs",
            side_effect=ImportError("No module named 'elevenlabs'"),
        ):
            success, metrics = await elevenlabs_provider.elevenlabs_text_to_speech(
                text="hi",
                voice_id="x",
                model_id="y",
                api_key="test-key",
            )

        assert success is False
        assert metrics is not None
        msg = (metrics.get("error") or "").lower()
        assert "pip install" in msg
        assert "voice-mode[elevenlabs]" in msg or "voice_mode[elevenlabs]" in msg


# ---------------------------------------------------------------------------
# 5) simple_tts_failover dispatch
# ---------------------------------------------------------------------------


class TestFailoverDispatch:
    @pytest.mark.asyncio
    async def test_elevenlabs_failure_falls_through_to_openai(self):
        """First endpoint is ElevenLabs (raises); second is OpenAI (succeeds).
        simple_tts_failover must reach OpenAI and report it as the winning
        provider."""
        from voice_mode.simple_failover import simple_tts_failover

        eleven_url = "https://api.elevenlabs.io/v1"
        openai_url = "https://api.openai.com/v1"

        eleven_call = AsyncMock(side_effect=Exception("elevenlabs down"))
        openai_call = AsyncMock(return_value=(True, {"generation": 0.1, "ttfa": 0.2}))

        with patch(
            "voice_mode.simple_failover.TTS_BASE_URLS",
            [eleven_url, openai_url],
        ), patch(
            "voice_mode.simple_failover.OPENAI_API_KEY", "test-key"
        ), patch(
            "voice_mode.elevenlabs_provider.elevenlabs_text_to_speech", new=eleven_call
        ), patch(
            "voice_mode.core.text_to_speech", new=openai_call
        ):
            success, metrics, config = await simple_tts_failover(
                text="hello world",
                voice="nova",
                model="tts-1",
            )

        assert success is True, f"failover should succeed via openai, got {config}"
        assert config["provider"] == "openai"
        assert config["base_url"] == openai_url
        eleven_call.assert_awaited_once()
        openai_call.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_elevenlabs_success_short_circuits_openai(self):
        from voice_mode.simple_failover import simple_tts_failover

        eleven_url = "https://api.elevenlabs.io/v1"
        openai_url = "https://api.openai.com/v1"

        eleven_call = AsyncMock(return_value=(True, {"generation": 0.05, "ttfa": 0.1}))
        openai_call = AsyncMock(return_value=(True, {"generation": 0.3, "ttfa": 0.4}))

        with patch(
            "voice_mode.simple_failover.TTS_BASE_URLS",
            [eleven_url, openai_url],
        ), patch(
            "voice_mode.simple_failover.OPENAI_API_KEY", "test-key"
        ), patch(
            "voice_mode.elevenlabs_provider.elevenlabs_text_to_speech", new=eleven_call
        ), patch(
            "voice_mode.core.text_to_speech", new=openai_call
        ):
            success, metrics, config = await simple_tts_failover(
                text="hello world",
                voice="nova",
                model="tts-1",
            )

        assert success is True
        assert config["provider"] == "elevenlabs"
        assert config["base_url"] == eleven_url
        eleven_call.assert_awaited_once()
        openai_call.assert_not_awaited()


# ---------------------------------------------------------------------------
# 6) CLI: voicemode elevenlabs list-voices
# ---------------------------------------------------------------------------


class TestCliListVoices:
    def test_list_voices_invokes_helper(self):
        """`voicemode elevenlabs list-voices` should call list_elevenlabs_voices
        and print id + name."""
        from voice_mode.cli_commands.elevenlabs import elevenlabs as elevenlabs_group

        fake_voices = [
            {"voice_id": "JBFqnCBsd6RMkjVDRZzb", "name": "George", "category": "premade"},
            {"voice_id": "21m00Tcm4TlvDq8ikWAM", "name": "Rachel", "category": "premade"},
        ]

        with patch(
            "voice_mode.cli_commands.elevenlabs.list_elevenlabs_voices",
            new=AsyncMock(return_value=fake_voices),
        ), patch(
            "voice_mode.cli_commands.elevenlabs.ELEVENLABS_API_KEY", "test-key"
        ):
            runner = CliRunner()
            result = runner.invoke(elevenlabs_group, ["list-voices"])

        assert result.exit_code == 0, result.output
        assert "George" in result.output
        assert "Rachel" in result.output
        assert "JBFqnCBsd6RMkjVDRZzb" in result.output

    def test_list_voices_without_api_key_fails_clean(self):
        from voice_mode.cli_commands.elevenlabs import elevenlabs as elevenlabs_group

        with patch(
            "voice_mode.cli_commands.elevenlabs.ELEVENLABS_API_KEY", None
        ), patch.dict("os.environ", {}, clear=False) as _:
            import os
            os.environ.pop("ELEVENLABS_API_KEY", None)
            runner = CliRunner()
            result = runner.invoke(elevenlabs_group, ["list-voices"])

        assert result.exit_code != 0
        assert "ELEVENLABS_API_KEY" in result.output

    def test_list_voices_handles_sdk_missing(self):
        from voice_mode.cli_commands.elevenlabs import elevenlabs as elevenlabs_group

        with patch(
            "voice_mode.cli_commands.elevenlabs.list_elevenlabs_voices",
            new=AsyncMock(side_effect=ImportError("No module named 'elevenlabs'")),
        ), patch(
            "voice_mode.cli_commands.elevenlabs.ELEVENLABS_API_KEY", "test-key"
        ):
            runner = CliRunner()
            result = runner.invoke(elevenlabs_group, ["list-voices"])

        assert result.exit_code != 0
        assert "voice-mode[elevenlabs]" in result.output

    def test_list_voices_handles_empty_result(self):
        from voice_mode.cli_commands.elevenlabs import elevenlabs as elevenlabs_group

        with patch(
            "voice_mode.cli_commands.elevenlabs.list_elevenlabs_voices",
            new=AsyncMock(return_value=[]),
        ), patch(
            "voice_mode.cli_commands.elevenlabs.ELEVENLABS_API_KEY", "test-key"
        ):
            runner = CliRunner()
            result = runner.invoke(elevenlabs_group, ["list-voices"])

        assert result.exit_code == 0
        assert "no voices" in result.output.lower()

    def test_list_voices_handles_generic_error(self):
        from voice_mode.cli_commands.elevenlabs import elevenlabs as elevenlabs_group

        with patch(
            "voice_mode.cli_commands.elevenlabs.list_elevenlabs_voices",
            new=AsyncMock(side_effect=RuntimeError("upstream 500")),
        ), patch(
            "voice_mode.cli_commands.elevenlabs.ELEVENLABS_API_KEY", "test-key"
        ):
            runner = CliRunner()
            result = runner.invoke(elevenlabs_group, ["list-voices"])

        assert result.exit_code != 0
        assert "upstream 500" in result.output


# ---------------------------------------------------------------------------
# 7) Additional unit coverage
# ---------------------------------------------------------------------------


class TestListVoicesHelper:
    """Direct unit coverage for voice_mode.elevenlabs_provider.list_elevenlabs_voices."""

    @pytest.mark.asyncio
    async def test_list_voices_normalises_sdk_response(self):
        from voice_mode import elevenlabs_provider

        # SDK responses use either `voice_id` or `id`, and may wrap voices in
        # a container object — the helper must handle both.
        # (MagicMock reserves `name=` for the mock's repr, so we set it after.)
        voice_a = MagicMock()
        voice_a.voice_id = "abc123"
        voice_a.name = "Aria"
        voice_a.category = "premade"

        voice_b = MagicMock()
        # Force fallback to `id` for voice_b.
        del voice_b.voice_id
        voice_b.id = "def456"
        voice_b.name = "Brave"
        voice_b.category = "cloned"

        fake_response = MagicMock()
        fake_response.voices = [voice_a, voice_b]

        fake_client = MagicMock()
        fake_client.voices.search = AsyncMock(return_value=fake_response)

        with patch.object(
            elevenlabs_provider, "_make_async_client", return_value=fake_client
        ):
            voices = await elevenlabs_provider.list_elevenlabs_voices("test-key")

        assert voices == [
            {"voice_id": "abc123", "name": "Aria", "category": "premade"},
            {"voice_id": "def456", "name": "Brave", "category": "cloned"},
        ]


class TestSetApiKey:
    def test_set_api_key_writes_to_credential_store(self):
        from voice_mode.cli_commands.elevenlabs import elevenlabs as elevenlabs_group

        fake_store = MagicMock()
        fake_store.name = "plaintext"
        fake_store.load.return_value = {"OTHER": "x"}

        with patch(
            "voice_mode.credential_store.get_credential_store",
            return_value=fake_store,
        ):
            runner = CliRunner()
            result = runner.invoke(elevenlabs_group, ["set-api-key", "sk-test"])

        assert result.exit_code == 0, result.output
        # Ensure save() got a dict containing the new key alongside existing keys.
        save_arg = fake_store.save.call_args.args[0]
        assert save_arg["ELEVENLABS_API_KEY"] == "sk-test"
        assert save_arg["OTHER"] == "x"
        assert "Saved" in result.output


class TestTestCommand:
    def test_test_command_invokes_tts(self):
        from voice_mode.cli_commands.elevenlabs import elevenlabs as elevenlabs_group

        with patch(
            "voice_mode.cli_commands.elevenlabs.elevenlabs_text_to_speech",
            new=AsyncMock(return_value=(True, {"ttfa": 0.5, "generation": 1.0, "playback": 2.0})),
        ), patch(
            "voice_mode.cli_commands.elevenlabs.ELEVENLABS_API_KEY", "test-key"
        ):
            runner = CliRunner()
            result = runner.invoke(elevenlabs_group, ["test", "hello"])

        assert result.exit_code == 0, result.output
        assert "OK" in result.output

    def test_test_command_reports_failure(self):
        from voice_mode.cli_commands.elevenlabs import elevenlabs as elevenlabs_group

        with patch(
            "voice_mode.cli_commands.elevenlabs.elevenlabs_text_to_speech",
            new=AsyncMock(return_value=(False, {"error": "quota exceeded"})),
        ), patch(
            "voice_mode.cli_commands.elevenlabs.ELEVENLABS_API_KEY", "test-key"
        ):
            runner = CliRunner()
            result = runner.invoke(elevenlabs_group, ["test", "hello"])

        assert result.exit_code != 0
        assert "quota exceeded" in result.output

    def test_test_command_without_api_key(self):
        from voice_mode.cli_commands.elevenlabs import elevenlabs as elevenlabs_group

        with patch(
            "voice_mode.cli_commands.elevenlabs.ELEVENLABS_API_KEY", None
        ):
            import os
            os.environ.pop("ELEVENLABS_API_KEY", None)
            runner = CliRunner()
            result = runner.invoke(elevenlabs_group, ["test", "hello"])

        assert result.exit_code != 0
        assert "ELEVENLABS_API_KEY" in result.output


class TestTextToSpeechEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_audio_response_reports_failure(self):
        from voice_mode import elevenlabs_provider

        async def _empty_stream(**kwargs):
            if False:  # produce an empty async iterator
                yield b""

        fake_client = MagicMock()
        fake_client.text_to_speech.stream = MagicMock(side_effect=_empty_stream)

        with patch.object(
            elevenlabs_provider, "_make_async_client", return_value=fake_client
        ):
            success, metrics = await elevenlabs_provider.elevenlabs_text_to_speech(
                text="hi",
                voice_id="JBFqnCBsd6RMkjVDRZzb",
                model_id="eleven_turbo_v2_5",
                api_key="test-key",
            )

        assert success is False
        assert metrics is not None
        assert "no audio" in metrics["error"].lower()

    @pytest.mark.asyncio
    async def test_stream_exception_reports_failure(self):
        from voice_mode import elevenlabs_provider

        fake_client = MagicMock()
        fake_client.text_to_speech.stream = MagicMock(
            side_effect=RuntimeError("network down")
        )

        with patch.object(
            elevenlabs_provider, "_make_async_client", return_value=fake_client
        ):
            success, metrics = await elevenlabs_provider.elevenlabs_text_to_speech(
                text="hi",
                voice_id="x",
                model_id="y",
                api_key="test-key",
            )

        assert success is False
        assert "network down" in metrics["error"]

    @pytest.mark.asyncio
    async def test_client_init_exception_reports_failure(self):
        from voice_mode import elevenlabs_provider

        with patch.object(
            elevenlabs_provider,
            "_make_async_client",
            side_effect=ValueError("bad key format"),
        ):
            success, metrics = await elevenlabs_provider.elevenlabs_text_to_speech(
                text="hi",
                voice_id="x",
                model_id="y",
                api_key="test-key",
            )

        assert success is False
        assert "bad key format" in metrics["error"]


class TestMapVoiceEdgeCases:
    def test_empty_string_returns_default(self):
        from voice_mode.elevenlabs_provider import (
            map_voice_to_elevenlabs,
            DEFAULT_VOICE_ID,
        )

        assert map_voice_to_elevenlabs("") == DEFAULT_VOICE_ID

    def test_elevenlabs_id_passes_through(self):
        from voice_mode.elevenlabs_provider import (
            map_voice_to_elevenlabs,
            DEFAULT_VOICE_ID,
        )

        assert map_voice_to_elevenlabs(DEFAULT_VOICE_ID) == DEFAULT_VOICE_ID

    def test_kokoro_voice_maps(self):
        from voice_mode.elevenlabs_provider import map_voice_to_elevenlabs

        # af_sky should resolve to Rachel (21m00Tcm4TlvDq8ikWAM)
        assert map_voice_to_elevenlabs("af_sky") == "21m00Tcm4TlvDq8ikWAM"

    def test_unknown_voice_maps_to_kokoro_default(self):
        from voice_mode.elevenlabs_provider import map_voice_to_kokoro

        # Unknown ElevenLabs voice falls back to af_sky.
        assert map_voice_to_kokoro("zzzNotAVoice") == "af_sky"
