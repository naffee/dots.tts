from __future__ import annotations

import base64
import binascii
import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve

import runpod
import soundfile as sf

from apps.gradio.service import (
    DEFAULT_PROMPT_NONE,
    discover_prompt_presets,
    resolve_prompt_selection,
)
from dots_tts.runtime import DotsTtsRuntime
from dots_tts.utils.logging import configure_logging
from dots_tts.utils.util import seed_everything


DEFAULT_MODEL = os.environ.get("DOTS_TTS_MODEL", "rednote-hilab/dots.tts-soar")
DEFAULT_PRECISION = os.environ.get("DOTS_TTS_PRECISION", "bfloat16")
DEFAULT_OPTIMIZE = os.environ.get("DOTS_TTS_OPTIMIZE", "0") == "1"
DEFAULT_MAX_GENERATE_LENGTH = int(
    os.environ.get("DOTS_TTS_MAX_GENERATE_LENGTH", "500")
)
DEFAULT_HF_CACHE = os.environ.get("HF_HOME") or os.environ.get(
    "HUGGINGFACE_HUB_CACHE"
)

configure_logging()

_RUNTIME_CACHE: dict[tuple[str, str, bool, int], DotsTtsRuntime] = {}
_PROMPT_PRESETS = discover_prompt_presets()


def _get_runtime(
    model_name_or_path: str,
    precision: str,
    optimize: bool,
    max_generate_length: int,
) -> DotsTtsRuntime:
    cache_key = (model_name_or_path, precision, optimize, max_generate_length)
    runtime = _RUNTIME_CACHE.get(cache_key)
    if runtime is None:
        runtime = DotsTtsRuntime.from_pretrained(
            model_name_or_path,
            cache_dir=DEFAULT_HF_CACHE,
            precision=precision,
            optimize=optimize,
            max_generate_length=max_generate_length,
        )
        _RUNTIME_CACHE[cache_key] = runtime
    return runtime


def _looks_like_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"}


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _decode_prompt_audio(prompt_audio: str | None) -> tuple[str | None, list[str]]:
    if not prompt_audio:
        return None, []
    if Path(prompt_audio).exists():
        return prompt_audio, []
    if _looks_like_url(prompt_audio):
        temp_dir = tempfile.mkdtemp(prefix="dots-tts-prompt-")
        suffix = Path(urlparse(prompt_audio).path).suffix or ".wav"
        output_path = Path(temp_dir) / f"prompt_audio{suffix}"
        urlretrieve(prompt_audio, output_path)
        return str(output_path), [temp_dir]

    try:
        payload = prompt_audio.split(",", 1)[-1]
        audio_bytes = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(
            "prompt_audio must be a local path, http(s) URL, or base64-encoded audio."
        ) from exc

    temp_dir = tempfile.mkdtemp(prefix="dots-tts-prompt-")
    output_path = Path(temp_dir) / "prompt_audio.wav"
    output_path.write_bytes(audio_bytes)
    return str(output_path), [temp_dir]


def _resolve_prompt_inputs(job_input: dict) -> tuple[str | None, str | None, list[str]]:
    preset_name = str(job_input.get("preset_name", "") or "").strip()
    prompt_audio = job_input.get("prompt_audio")
    prompt_text = job_input.get("prompt_text")

    if preset_name and preset_name != DEFAULT_PROMPT_NONE:
        preset_audio_path, preset_prompt_text = resolve_prompt_selection(
            preset_name,
            _PROMPT_PRESETS,
        )
        if preset_audio_path is None:
            available_presets = [preset.name for preset in _PROMPT_PRESETS]
            raise ValueError(
                f"Unknown preset_name={preset_name!r}. "
                f"Available presets: {available_presets or '[]'}."
            )
        if not prompt_audio:
            prompt_audio = preset_audio_path
        if not prompt_text:
            prompt_text = preset_prompt_text or None

    prompt_audio_url = job_input.get("prompt_audio_url")
    prompt_audio_base64 = job_input.get("prompt_audio_base64")
    if prompt_audio is None:
        prompt_audio = prompt_audio_url or prompt_audio_base64

    prompt_audio_path, cleanup_dirs = _decode_prompt_audio(prompt_audio)
    normalized_prompt_text = (prompt_text or "").strip() or None
    return prompt_audio_path, normalized_prompt_text, cleanup_dirs


def _extract_job_input(job: dict) -> dict:
    if not isinstance(job, dict):
        return {}

    raw_input = job.get("input")
    if isinstance(raw_input, dict):
        return raw_input

    if isinstance(raw_input, str):
        return {"text": raw_input}

    if isinstance(job.get("body"), dict):
        return job["body"]

    if isinstance(job.get("body"), str):
        return {"text": job["body"]}

    return job


def _resolve_text(job_input: dict) -> str | None:
    for key in ("text", "input", "prompt", "message"):
        value = job_input.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def handler(job: dict) -> dict:
    job_input = _extract_job_input(job)

    text = _resolve_text(job_input)
    if not text:
        return {
            "error": "Missing synthesis text.",
            "expected_input": {
                "text": "Hello from dots tts.",
                "preset_name": "<optional preset name if prompt assets exist>",
                "prompt_audio": "<optional local path, URL, or base64 audio>",
                "prompt_text": "<optional transcript matching prompt_audio>",
                "model_name_or_path": DEFAULT_MODEL,
                "num_steps": 10,
                "guidance_scale": 1.2,
                "seed": 42,
            },
            "accepted_text_fields": ["input.text", "input.prompt", "input.message"],
            "available_presets": [preset.name for preset in _PROMPT_PRESETS],
            "received_keys": sorted(job_input.keys()),
        }

    model_name_or_path = job_input.get("model_name_or_path", DEFAULT_MODEL)
    precision = job_input.get("precision", DEFAULT_PRECISION)
    optimize = _as_bool(job_input.get("optimize", DEFAULT_OPTIMIZE))
    max_generate_length = int(
        job_input.get("max_generate_length", DEFAULT_MAX_GENERATE_LENGTH)
    )
    seed = int(job_input.get("seed", 42))

    runtime = _get_runtime(
        model_name_or_path=model_name_or_path,
        precision=precision,
        optimize=optimize,
        max_generate_length=max_generate_length,
    )

    prompt_audio_path = None
    prompt_text = None
    cleanup_dirs: list[str] = []
    try:
        prompt_audio_path, prompt_text, cleanup_dirs = _resolve_prompt_inputs(
            job_input
        )
        seed_everything(seed)
        result = runtime.generate(
            text=text,
            prompt_audio_path=prompt_audio_path,
            prompt_text=prompt_text,
            template_name=job_input.get("template_name"),
            language=job_input.get("language"),
            speaker_scale=float(job_input.get("speaker_scale", 1.5)),
            ode_method=job_input.get("ode_method", "euler"),
            num_steps=int(job_input.get("num_steps", 10)),
            guidance_scale=float(job_input.get("guidance_scale", 1.2)),
            normalize_text=_as_bool(job_input.get("normalize_text", False)),
            profile_inference=_as_bool(job_input.get("profile_inference", False)),
        )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_wav_path = Path(temp_wav.name)

        try:
            sf.write(
                temp_wav_path,
                result["audio"].float().cpu().squeeze().numpy(),
                result["sample_rate"],
                format="WAV",
            )
            audio_b64 = base64.b64encode(temp_wav_path.read_bytes()).decode("utf-8")
        finally:
            temp_wav_path.unlink(missing_ok=True)
    finally:
        for cleanup_dir in cleanup_dirs:
            temp_path = Path(cleanup_dir)
            for child in temp_path.glob("*"):
                child.unlink(missing_ok=True)
            temp_path.rmdir()

    return {
        "audio_base64": audio_b64,
        "content_type": "audio/wav",
        "sample_rate": result["sample_rate"],
        "request_id": result["fid"],
        "time_used": result["time_used"],
        "rtf": result["rtf"],
        "model_name_or_path": model_name_or_path,
        "precision": precision,
        "seed": seed,
        "preset_name": str(job_input.get("preset_name", "") or "") or None,
        "profiling": result["profiling"],
    }


runpod.serverless.start({"handler": handler})
