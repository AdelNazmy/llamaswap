---
name: music-composer
description: Compose and generate music or full songs locally through llamaswap's MiniMax Music 3 model, including lyrics and style control. Use when the user asks to make a song, music, melody, beat, track, jingle, or to set lyrics to music.
---

# Music Composer

Use llamaswap's local music model at `http://localhost:9090/v1` (override with
the `LLAMASWAP_BASE_URL` env var). Music generation runs on the local GPU and
unloads the chat LLM, image, TTS/ASR, and embedding servers, so the next chat
turn may be slower while the chat model reloads.

## Workflow

1. Build a structured caption. MiniMax drifts to Chinese vocals with loose
   prompts, so include genre, BPM, key, mood, vocal language, and arrangement
   in this shape:

   ```text
   Genre: pop rock. BPM: 128. Key: E major. Bright, energetic, polished studio production. Vocals: clear male lead, English vocal language. Arrangement: clean drums, crisp rhythm guitars, wide stereo choruses.
   ```

2. Lyrics: if the user supplied lyrics, use them. Otherwise write short lyrics
   with section tags capitalized on their own lines (`[Verse]`, `[Chorus]`,
   `[Bridge]`). Keep the vocal language explicit in the caption.
3. Choose a duration budget in seconds (typically 20-60; maximum 360) and an
   output path under `outputs/` unless the user asked otherwise. The budget is
   an upper bound: the model can finish earlier when the lyrics end.
4. Run the helper from the project root, then verify the file exists and is
   audio (`file <path>`). Only report success after verification.

## Command

```bash
SCRIPT="${OPENCODE_CONFIG_DIR:-$HOME/.config/opencode}/skills/music-composer/compose.py"
[ -f "$SCRIPT" ] || SCRIPT=".opencode/skills/music-composer/compose.py"
python3 "$SCRIPT" \
  --caption "Genre: pop rock. BPM: 128. Key: E major. Bright, energetic, polished studio production. Vocals: clear male lead, English vocal language. Arrangement: clean drums, crisp rhythm guitars, wide stereo choruses." \
  --lyrics $'[Verse]\nCity lights are shining low\n[Chorus]\nTurn it up and let it fly' \
  --duration 30 \
  --output outputs/city-lights.wav
```

Useful flags: `--seed 7` for repeatable output, `--format mp3` (transcoded
server-side), `--lyrics-file lyrics.txt`, and `--extra guidance_scale=1.2` for
backend options passed through to audio.cpp.

## Raw API

The helper posts JSON to the custom music endpoint and writes the binary
response:

```http
POST /v1/audio/music
Content-Type: application/json

{"model": "minimax-music3", "input": "<structured caption>", "lyrics": "[Verse] ...", "duration": 30, "response_format": "wav"}
```

The response is binary audio (`audio/wav` by default). Unknown JSON fields are
forwarded as backend request options, so seed, steps, and guidance options work
without proxy changes. `duration` is a generation budget; the returned audio
can be shorter if the model reaches its end token first.

## Rules

- Only use `minimax-music3` for music; do not pretend audio was generated.
- Do not include copyrighted lyrics unless the user supplied them.
- Report the saved file path as `file_path`.
