# dots.tts Runpod Serverless API

## Request

Send jobs in the normal Runpod shape:

```json
{
  "input": {
    "text": "Hello from dots tts.",
    "model_name_or_path": "rednote-hilab/dots.tts-soar",
    "num_steps": 10,
    "guidance_scale": 1.2,
    "seed": 42
  }
}
```

## Accepted text fields

The handler accepts text from any of these fields:

- `input.text`
- `input.prompt`
- `input.message`
- `input` as a raw string

## Voice cloning inputs

Optional prompt inputs:

- `prompt_audio`: local path, `http(s)` URL, or base64 audio
- `prompt_audio_url`: explicit URL variant
- `prompt_audio_base64`: explicit base64 variant
- `prompt_text`: transcript matching the prompt audio
- `preset_name`: name of a bundled prompt preset if prompt assets exist in `apps/gradio/default_prompts/`

Valid combinations:

- `text` only
- `text` + `prompt_audio`
- `text` + `prompt_audio` + `prompt_text`
- `text` + `preset_name`
- `text` + `preset_name` + `prompt_text`

Invalid combination:

- `text` + `prompt_text` only

## Other supported fields

- `model_name_or_path`
- `precision`
- `optimize`
- `max_generate_length`
- `seed`
- `template_name`
- `language`
- `speaker_scale`
- `ode_method`
- `num_steps`
- `guidance_scale`
- `normalize_text`
- `profile_inference`

## Response

Successful responses return:

```json
{
  "audio_base64": "<wav bytes as base64>",
  "content_type": "audio/wav",
  "sample_rate": 48000,
  "request_id": "abc123",
  "time_used": 12.34,
  "rtf": 0.56,
  "model_name_or_path": "rednote-hilab/dots.tts-soar",
  "precision": "bfloat16",
  "seed": 42,
  "preset_name": null,
  "profiling": null
}
```

## Notes

- This is a serverless inference API, not the Gradio web UI.
- The current repo checkout does not include actual default prompt audio assets, so `preset_name` will usually be unavailable unless you add them yourself.
