#!/usr/bin/env python3
"""Create a small LLM response and replay it through TTS.

Example:
    python scripts/llm_tts_replay.py --llm-model qwen3.5-9b --tts-model supertonic-3 \
        --prompt "Say hello in one sentence." --output test_output/llm_replay.wav
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import httpx


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ask a small LLM, then replay its reply with TTS.")
    parser.add_argument("--base-url", default=os.environ.get("LLAMASWAP_BASE_URL", "http://127.0.0.1:11434"))
    parser.add_argument("--llm-model", default=os.environ.get("LLM_MODEL", "qwen3.5-9b"))
    parser.add_argument("--tts-model", default=os.environ.get("TTS_MODEL", "supertonic-3"))
    parser.add_argument("--prompt", default=os.environ.get("LLM_PROMPT", "Say hello in one sentence."))
    parser.add_argument("--output", default=os.environ.get("LLM_TTS_OUTPUT", "test_output/llm_replay.wav"))
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=600.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=args.timeout) as client:
        llm_resp = client.post(
            f"{args.base_url}/v1/chat/completions",
            json={
                "model": args.llm_model,
                "messages": [{"role": "user", "content": args.prompt}],
                "max_tokens": args.max_tokens,
                "temperature": 0.0,
            },
        )
        llm_resp.raise_for_status()
        llm_payload = llm_resp.json()
        reply = llm_payload["choices"][0]["message"]["content"].strip()
        if not reply:
            raise RuntimeError("LLM returned an empty response")

        print(f"LLM reply: {reply}")

        tts_resp = client.post(
            f"{args.base_url}/v1/audio/speech",
            json={
                "model": args.tts_model,
                "input": reply,
            },
        )
        tts_resp.raise_for_status()
        output_path.write_bytes(tts_resp.content)

    print(f"Saved TTS replay to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
