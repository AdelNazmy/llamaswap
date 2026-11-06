#!/usr/bin/env python3
"""Generate a song through the local llamaswap music endpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

DEFAULT_BASE_URL = os.environ.get("LLAMASWAP_BASE_URL", "http://localhost:9090/v1")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compose a song with llamaswap's MiniMax Music 3 model."
    )
    parser.add_argument(
        "--caption",
        "--input",
        dest="caption",
        required=True,
        help="structured music description (genre/BPM/key/vocals/arrangement)",
    )
    parser.add_argument("--lyrics", help="lyrics with [Verse]/[Chorus] tags on their own lines")
    parser.add_argument("--lyrics-file", help="read lyrics from a file")
    parser.add_argument("--model", default="minimax-music3")
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="duration budget in seconds; the song may end earlier",
    )
    parser.add_argument("--seed", type=int, help="optional random seed")
    parser.add_argument(
        "--format",
        default="wav",
        choices=["wav", "mp3", "flac", "ogg", "opus", "pcm"],
    )
    parser.add_argument("--output", default="song.wav")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--extra", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument(
        "--timeout",
        type=float,
        default=1800,
        help="seconds to wait for model load + generation (default 1800)",
    )
    args = parser.parse_args()

    lyrics = args.lyrics
    if args.lyrics_file:
        with open(args.lyrics_file, encoding="utf-8") as fh:
            lyrics = fh.read()

    payload = {
        "model": args.model,
        "input": args.caption,
        "duration": args.duration,
        "response_format": args.format,
    }
    if lyrics:
        payload["lyrics"] = lyrics
    if args.seed is not None:
        payload["seed"] = args.seed
    for item in args.extra:
        key, sep, value = item.partition("=")
        if not sep:
            parser.error(f"--extra expects KEY=VALUE, got {item!r}")
        payload[key] = value

    url = args.base_url.rstrip("/") + "/audio/music"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            status = resp.status
            ctype = resp.headers.get_content_type()
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        sys.stderr.write(f"llamaswap error {exc.code}: {exc.read()[:2000]!r}\n")
        return 1
    except urllib.error.URLError as exc:
        sys.stderr.write(f"cannot reach {url}: {exc.reason}\n")
        return 1

    if status != 200:
        sys.stderr.write(f"llamaswap error {status}: {raw[:2000]!r}\n")
        return 1

    if ctype == "application/json":
        sys.stderr.write(f"llamaswap did not return audio: {raw[:2000]!r}\n")
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "wb") as fh:
        fh.write(raw)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
