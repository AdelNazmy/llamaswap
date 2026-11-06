#!/usr/bin/env python3
"""Generate or edit images through the local llamaswap server."""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request
import uuid

DEFAULT_BASE_URL = os.environ.get("LLAMASWAP_BASE_URL", "http://localhost:9090/v1")


def request_json(url: str, payload: dict, timeout: float):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.headers.get_content_type(), resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers.get_content_type(), exc.read()
    except urllib.error.URLError as exc:
        return 0, "text/plain", f"cannot reach {url}: {exc.reason}".encode()


def request_multipart(url: str, fields: dict, files: dict, timeout: float):
    boundary = "----llamaswap" + uuid.uuid4().hex
    body = bytearray()
    for name, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        )
        body.extend(str(value).encode())
        body.extend(b"\r\n")
    for name, (filename, content) in files.items():
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(
            f'Content-Disposition: form-data; name="{name}"; '
            f'filename="{filename}"\r\n'.encode()
        )
        body.extend(b"Content-Type: application/octet-stream\r\n\r\n")
        body.extend(content)
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())

    req = urllib.request.Request(
        url,
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.headers.get_content_type(), resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers.get_content_type(), exc.read()
    except urllib.error.URLError as exc:
        return 0, "text/plain", f"cannot reach {url}: {exc.reason}".encode()


def download(url: str, timeout: float) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate or edit an image with llamaswap."
    )
    parser.add_argument("--prompt", required=True, help="image description / edit instruction")
    parser.add_argument(
        "--model",
        default="flux-schnell",
        choices=["flux-schnell", "qwen-image-2.1-uc"],
    )
    parser.add_argument("--size", default="1024x1024", help="WxH, e.g. 1024x1024")
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--image", help="input image file (edits; qwen-image-2.1-uc only)")
    parser.add_argument("--mask", help="optional mask image file for edits")
    parser.add_argument("--output", default="image.png")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--timeout",
        type=float,
        default=1800,
        help="seconds to wait for model load + generation (default 1800)",
    )
    args = parser.parse_args()

    if args.image:
        if args.model != "qwen-image-2.1-uc":
            parser.error("image edits require --model qwen-image-2.1-uc")
        fields = {
            "model": args.model,
            "prompt": args.prompt,
            "size": args.size,
            "n": str(args.n),
        }
        files = {"image": (os.path.basename(args.image), open(args.image, "rb").read())}
        if args.mask:
            files["mask"] = (os.path.basename(args.mask), open(args.mask, "rb").read())
        status, _ctype, raw = request_multipart(
            args.base_url.rstrip("/") + "/images/edits", fields, files, args.timeout
        )
    else:
        payload = {
            "model": args.model,
            "prompt": args.prompt,
            "size": args.size,
            "n": args.n,
        }
        status, _ctype, raw = request_json(
            args.base_url.rstrip("/") + "/images/generations", payload, args.timeout
        )

    if status != 200:
        sys.stderr.write(f"llamaswap error {status}: {raw[:2000]!r}\n")
        return 1

    try:
        obj = json.loads(raw)
    except ValueError:
        sys.stderr.write("llamaswap returned non-JSON image response\n")
        return 1

    items = obj.get("data") or []
    if not items:
        sys.stderr.write(f"no image data in response: {raw[:2000]!r}\n")
        return 1

    stem, ext = os.path.splitext(args.output)
    ext = ext or ".png"
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    written = []
    for index, item in enumerate(items, start=1):
        if item.get("b64_json"):
            content = base64.b64decode(item["b64_json"])
        elif item.get("url"):
            content = download(item["url"], args.timeout)
        else:
            continue
        path = args.output if len(items) == 1 else f"{stem}-{index}{ext}"
        with open(path, "wb") as fh:
            fh.write(content)
        written.append(path)

    if not written:
        sys.stderr.write("image response contained neither b64_json nor url\n")
        return 1

    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
