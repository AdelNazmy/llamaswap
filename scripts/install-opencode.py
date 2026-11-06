#!/usr/bin/env python3
"""Install llamaswap's opencode provider config and skills.

Merges the bundle shipped in ``opencode/`` (provider fragment + skills) into
the user's default opencode config directory (``~/.config/opencode`` by
default). Idempotent: existing providers, models and user settings are kept;
the llamaswap provider/models are added or refreshed.

The deploy script calls this on the host, and the Hub image entrypoint calls
it when an opencode config directory is mounted at
LLAMASWAP_OPENCODE_CONFIG_DIR (see docker-compose.hub.yml).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = Path(
    os.environ.get("LLAMASWAP_OPENCODE_SOURCE") or (REPO_ROOT / "opencode")
)
DEFAULT_CONFIG_DIR = Path(
    os.environ.get("OPENCODE_CONFIG_DIR") or (Path.home() / ".config" / "opencode")
)


def strip_jsonc(text: str) -> str:
    """Remove // and /* */ comments plus trailing commas from JSONC."""
    out: list[str] = []
    i = 0
    length = len(text)
    in_string = False
    escaped = False
    while i < length:
        ch = text[i]
        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < length:
            nxt = text[i + 1]
            if nxt == "/":
                i += 2
                while i < length and text[i] not in "\r\n":
                    i += 1
                continue
            if nxt == "*":
                i += 2
                while i + 1 < length and not (text[i] == "*" and text[i + 1] == "/"):
                    i += 1
                i += 2
                continue
        if ch == ",":
            j = i + 1
            while j < length and text[j] in " \t\r\n":
                j += 1
            if j < length and text[j] in "]}":
                i += 1
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def load_config(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = json.loads(strip_jsonc(text))
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


def deep_merge(base: dict, extra: dict) -> dict:
    """Merge `extra` into `base` (dicts recurse, everything else overwrites)."""
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def apply_overrides(bundle: dict, base_url: str | None, api_key: str | None) -> None:
    provider = bundle.get("provider", {}).get("llamaswap")
    if not isinstance(provider, dict):
        return
    options = provider.setdefault("options", {})
    if base_url:
        options["baseURL"] = base_url
    if api_key:
        options["apiKey"] = api_key


def pick_target(config_dir: Path) -> Path:
    preferred = config_dir / "opencode.json"
    if preferred.exists():
        return preferred
    jsonc = config_dir / "opencode.jsonc"
    if jsonc.exists():
        return jsonc
    return preferred


def install_skills(source: Path, dest: Path) -> list[str]:
    skills_src = source / "skills"
    if not skills_src.is_dir():
        return []
    dest.mkdir(parents=True, exist_ok=True)
    installed: list[str] = []
    for child in sorted(skills_src.iterdir()):
        if not child.is_dir():
            continue
        target = dest / child.name
        shutil.copytree(
            child,
            target,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )
        installed.append(child.name)
    return installed


def chown_paths(paths: list[Path], uid: int, gid: int) -> None:
    if os.geteuid() != 0:
        return
    for path in paths:
        try:
            os.chown(path, uid, gid)
        except OSError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge llamaswap's opencode provider + skills into a config dir."
    )
    parser.add_argument(
        "--config-dir",
        default=str(DEFAULT_CONFIG_DIR),
        help="opencode config directory (default: ~/.config/opencode)",
    )
    parser.add_argument(
        "--source",
        default=str(DEFAULT_SOURCE),
        help="bundle directory containing opencode.json and skills/",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("LLAMASWAP_OPENCODE_BASE_URL"),
        help="override the provider base URL (default: keep bundle value)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("LLAMASWAP_API_KEY"),
        help="optional API key for LLAMASWAP_API_KEY-protected servers",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source = Path(args.source).expanduser()
    bundle_path = source / "opencode.json"
    if not bundle_path.is_file():
        sys.stderr.write(f"install-opencode: bundle not found: {bundle_path}\n")
        return 1

    try:
        bundle = load_config(bundle_path)
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"install-opencode: cannot read {bundle_path}: {exc}\n")
        return 1
    apply_overrides(bundle, args.base_url, args.api_key)

    config_dir = Path(args.config_dir).expanduser()
    target = pick_target(config_dir)
    existing: dict = {}
    if target.exists():
        try:
            existing = load_config(target)
        except (OSError, ValueError) as exc:
            sys.stderr.write(f"install-opencode: cannot parse {target}: {exc}\n")
            return 1

    merged = deep_merge(existing, bundle)
    merged.setdefault("$schema", "https://opencode.ai/config.json")
    rendered = json.dumps(merged, indent=2) + "\n"

    try:
        current = target.read_text(encoding="utf-8") if target.exists() else None
    except OSError:
        current = None

    if current == rendered:
        print(f"install-opencode: up to date {target}")
    elif args.dry_run:
        print(f"install-opencode: would write {target}")
    else:
        config_dir.mkdir(parents=True, exist_ok=True)
        if target.exists():
            backup = target.with_name(
                target.name + ".bak." + time.strftime("%Y%m%d%H%M%S")
            )
            shutil.copy2(target, backup)
            print(f"install-opencode: backup {backup}")
        target.write_text(rendered, encoding="utf-8")
        print(f"install-opencode: wrote {target}")

    installed = install_skills(source, config_dir / "skills")
    if installed:
        action = "would install" if args.dry_run else "installed"
        print(f"install-opencode: {action} skills: {', '.join(installed)}")

    if not args.dry_run and os.geteuid() == 0:
        owner = config_dir.stat()
        touched = [target, config_dir / "skills"]
        touched.extend((config_dir / "skills" / name) for name in installed)
        chown_paths(touched, owner.st_uid, owner.st_gid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
