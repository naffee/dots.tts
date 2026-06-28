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


def handler(job: dict) -> dict:
    job_input = job.get("input") or {}

    text = job_input.get("text")
    if not text:
        raise ValueError("input.text is required.")

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
    cleanup_dirs: list[str] = []
    try:
        prompt_audio_path, cleanup_dirs = _decode_prompt_audio(
            job_input.get("prompt_audio")
        )
        seed_everything(seed)
        result = runtime.generate(
            text=text,
            prompt_audio_path=prompt_audio_path,
            prompt_text=job_input.get("prompt_text"),
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
        "profiling": result["profiling"],
    }


runpod.serverless.start({"handler": handler})
