---
name: artist
description: Generate or edit images locally through llamaswap's diffusion models (flux-schnell, qwen-image-2.1-uc). Use when the user asks to draw, create, generate, illustrate, render, design, or edit a picture, image, photo, or artwork.
---

# Artist

Use llamaswap's local image models at `http://localhost:9090/v1`
(override with the `LLAMASWAP_BASE_URL` env var). These run on the local GPU:
a request unloads the chat LLM and the audio/music roles, so the next chat
turn may take longer while the chat model reloads.

## Workflow

1. Turn the request into a concrete visual prompt: subject, style, lighting,
   composition, colors. Keep the user's intent; do not add watermarks/text
   unless asked.
2. Choose a model:
   - `flux-schnell` - fast (4-step distilled FLUX), text-to-image only.
   - `qwen-image-2.1-uc` - higher quality, and the only model that supports
     image editing (`--image`).
3. Run the helper from the project root and save under `outputs/` unless the
   user asked for a specific path.
4. Verify the file exists and is a real image (`file <path>`). Only then report
   the path. If the command fails, show the error instead of claiming success.

## Text-to-image

```bash
SCRIPT="${OPENCODE_CONFIG_DIR:-$HOME/.config/opencode}/skills/artist/generate.py"
[ -f "$SCRIPT" ] || SCRIPT=".opencode/skills/artist/generate.py"
python3 "$SCRIPT" \
  --prompt "a cat piloting a tiny spaceship, cinematic lighting, 35mm" \
  --model flux-schnell \
  --size 1024x1024 \
  --output outputs/cat-spaceship.png
```

Useful flags: `--n 2` (multiple images), `--size 512x512`,
`--timeout 1800` (first request includes model load).

## Image edit

Edits require `qwen-image-2.1-uc`:

```bash
SCRIPT="${OPENCODE_CONFIG_DIR:-$HOME/.config/opencode}/skills/artist/generate.py"
[ -f "$SCRIPT" ] || SCRIPT=".opencode/skills/artist/generate.py"
python3 "$SCRIPT" \
  --model qwen-image-2.1-uc \
  --image path/to/input.png \
  --prompt "turn the sky into a sunset, keep the subject unchanged" \
  --output outputs/edited.png
```

Add `--mask path/to/mask.png` for an inpaint mask.

## Raw API

The helper posts to the OpenAI-compatible endpoints and decodes the base64
result (`data[0].b64_json`):

- `POST /v1/images/generations` with JSON
  `{"model": ..., "prompt": ..., "size": ..., "n": ...}`
- `POST /v1/images/edits` as multipart with `model`, `prompt`, `image`, `mask`

Both return `{"created": ..., "data": [{"b64_json": ...}]}`. Generation can
take minutes on the first request (model download is not involved; loading is).

## Rules

- Only use the image models above; never invent image URLs or pretend an image
  was generated.
- Keep prompts safe-for-work unless the user explicitly requests otherwise.
- Report the saved file path as `file_path`, not a fabricated link.
