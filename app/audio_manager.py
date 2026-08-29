"""Per-role on-demand audio servers (TTS / ASR).

One AudioManager exists per audio role (``tts``, ``asr``). Like the
chat-LLM swap path — and unlike the persistent embedding server — the
audio servers are **on-demand**: nothing boots with the proxy. The
first request for a role launches the first configured model; a request
naming a *different* audio model of the same role transparently stops
the running server and launches the requested one. After
``idle_unload_seconds`` with no requests the running server is stopped
to free VRAM (same idle-unload policy as the chat LLM; 0 disables it).

Health checking is generic — every supported backend (llama-server,
audio.cpp audiocpp_server, whisper.cpp whisper-server) exposes
``GET /health``; the path is configurable per model via
``meta.health_path``.
"""

import asyncio
import json
import logging
import os
import signal
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import httpx

from .registry import ModelConfig, UnknownModelError

logger = logging.getLogger("llamaswap.audio_manager")


class AudioState(str, Enum):
    LOADING = "loading"
    READY = "ready"
    STOPPED = "stopped"
    FAILED = "failed"


class AudioLoadError(Exception):
    """Raised when an audio backend could not be brought up."""


@dataclass
class _RunningAudio:
    config: ModelConfig
    process: asyncio.subprocess.Process
    state: AudioState
    detail: str = ""
    config_file: Optional[str] = None


class AudioManager:
    """Starts, monitors and swaps the persistent server for one audio role."""

    def __init__(self, role: str, startup_timeout: float,
                 stop_timeout: float, health_interval: float,
                 config_dir: str | Path, idle_unload_seconds: float = 0.0):
        self.role = role
        self._startup_timeout = startup_timeout
        self._stop_timeout = stop_timeout
        self._health_interval = health_interval
        self._idle_unload_seconds = idle_unload_seconds
        self._lock = asyncio.Lock()
        self._config_dir = Path(config_dir)
        self._current: Optional[_RunningAudio] = None
        self._configs: dict[str, ModelConfig] = {}
        self._last_activity: float = 0.0
        self._idle_task: Optional[asyncio.Task] = None
        if idle_unload_seconds > 0:
            self._idle_task = asyncio.get_running_loop().create_task(
                self._idle_watcher()
            )

    @property
    def name(self) -> Optional[str]:
        cur = self._current
        return cur.config.name if cur and cur.state is AudioState.READY else None

    @property
    def is_running(self) -> bool:
        cur = self._current
        return cur is not None and cur.state is AudioState.READY

    @property
    def current_config(self) -> Optional[ModelConfig]:
        cur = self._current
        return cur.config if cur is not None else None

    @property
    def configured_names(self) -> list[str]:
        return sorted(self._configs)

    def configure(self, configs: list[ModelConfig]) -> None:
        """Refresh the set of models this role can serve (registry reload)."""
        self._configs = {cfg.name: cfg for cfg in configs}
        if self._configs:
            logger.info(
                "%s role configured with %d model(s): %s",
                self.role, len(self._configs), ", ".join(self._configs),
            )

    def config(self, name: str) -> Optional[ModelConfig]:
        return self._configs.get(name)

    def first_config_name(self) -> Optional[str]:
        """Name of the first configured model (boot default), if any."""
        return next(iter(sorted(self._configs)), None)

    def status(self) -> dict[str, Any]:
        cur = self._current
        if cur is None:
            return {"role": self.role, "state": "stopped", "model": None,
                    "detail": ""}
        info: dict[str, Any] = {
            "role": self.role,
            "state": cur.state.value,
            "model": cur.config.name if cur.state in (
                AudioState.READY, AudioState.LOADING, AudioState.FAILED
            ) else None,
            "port": cur.config.port,
            "detail": cur.detail,
        }
        if cur.state is AudioState.READY and self._idle_unload_seconds > 0:
            elapsed = asyncio.get_running_loop().time() - self._last_activity
            info["idle_seconds"] = max(0.0, round(elapsed, 1))
            info["idle_unload_seconds"] = self._idle_unload_seconds
        return info

    def _health_url(self, cfg: ModelConfig) -> str:
        return cfg.health_url()

    def _stderr_line(self, name: str, chunk: bytes) -> None:
        for line in chunk.decode(errors="replace").splitlines():
            line = line.strip()
            if line:
                logger.info("%s-server[%s]: %s", self.role, name, line)

    async def _wait_healthy(self, cfg: ModelConfig,
                            proc: asyncio.subprocess.Process) -> None:
        deadline = asyncio.get_running_loop().time() + self._startup_timeout
        async with httpx.AsyncClient(timeout=5.0) as client:
            while True:
                if proc.returncode is not None:
                    raise AudioLoadError(
                        f"{self.role} server for '{cfg.name}' exited (code "
                        f"{proc.returncode}) during startup; see logs above"
                    )
                try:
                    r = await client.get(self._health_url(cfg))
                    if r.status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                if asyncio.get_running_loop().time() > deadline:
                    raise AudioLoadError(
                        f"{self.role} server for '{cfg.name}' did not become "
                        f"healthy within {self._startup_timeout:.0f}s"
                    )
                await asyncio.sleep(self._health_interval)

    async def _stop_current(self) -> None:
        cur = self._current
        self._current = None
        # Tear down the rendered config file (if any) after the process ends.
        config_file = cur.config_file if cur else None
        if cur is None or cur.process.returncode is not None:
            _cleanup_config(config_file)
            return
        proc = cur.process
        logger.info(
            "stopping %s server for '%s' (pid %s)",
            self.role, cur.config.name, proc.pid,
        )
        try:
            proc.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            _cleanup_config(config_file)
            return
        try:
            await asyncio.wait_for(proc.wait(), timeout=self._stop_timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "%s server for '%s' did not exit, killing",
                self.role, cur.config.name,
            )
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()
        logger.info("%s server for '%s' stopped", self.role, cur.config.name)
        _cleanup_config(config_file)

    async def ensure_model(self, name: str,
                           registry: "Registry") -> ModelConfig:
        """Make sure `name` (an audio model of this role) is the loaded one.

        Returns the model's config; raises UnknownModelError if the model is
        not a member of this role, or AudioLoadError if it fails to load.
        """
        cfg = registry.get(name)  # raises UnknownModelError
        if cfg.role != self.role:
            raise UnknownModelError(
                name,
                [c.name for c in self._configs.values()],
            )
        async with self._lock:
            self._last_activity = asyncio.get_running_loop().time()
            cur = self._current
            if cur is not None and cur.config.name == name:
                # Already this model: wait out an in-flight load if needed.
                if cur.state is AudioState.READY:
                    return cfg
                if cur.state is AudioState.LOADING:
                    await self._wait_loading_settled(cur)
                    if self.is_running:
                        return cfg
                    # The in-flight load failed; fall through to relaunch.
            elif cur is not None and cur.state is AudioState.LOADING:
                # A swap for another model is in flight; wait for it to
                # settle, then swap again (bounded, mirrors ProcessManager).
                await self._wait_loading_settled(cur)
            cur = self._current
            if cur is not None:
                await self._stop_current()
            await self._launch(cfg)
            return cfg

    async def _wait_loading_settled(self, prev: _RunningAudio) -> None:
        deadline = asyncio.get_running_loop().time() + self._startup_timeout + 30.0
        while self._current is prev and prev.state is AudioState.LOADING:
            if asyncio.get_running_loop().time() > deadline:
                break
            await asyncio.sleep(0.25)

    def _render_config(self, cfg: ModelConfig) -> Optional[str]:
        """Write a per-model config JSON file (audio.cpp style), if defined."""
        if not cfg.command.config_json:
            return None
        self._config_dir.mkdir(parents=True, exist_ok=True)
        path = self._config_dir / f"{self.role}-{cfg.name}.json"
        payload = cfg.render_config_json(str(path))
        path.write_text(json.dumps(payload, indent=2))
        return str(path)

    async def _launch(self, cfg: ModelConfig) -> None:
        config_file = self._render_config(cfg)
        argv = cfg.build_argv(config_path=config_file)
        logger.info(
            "launching %s server for '%s': %s",
            self.role, cfg.name, " ".join(argv),
        )
        env = {**os.environ, **cfg.command.env}
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                env=env,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            _cleanup_config(config_file)
            raise AudioLoadError(f"failed to spawn {self.role} server: {exc}") from exc
        entry = _RunningAudio(config=cfg, process=proc, state=AudioState.LOADING,
                              config_file=config_file)
        self._current = entry
        asyncio.get_running_loop().create_task(self._tail_stderr(entry))
        try:
            await self._wait_healthy(cfg, proc)
        except BaseException:
            await self._stop_current()
            raise
        entry.state = AudioState.READY
        entry.detail = ""
        logger.info("%s model '%s' ready on port %d",
                    self.role, cfg.name, cfg.port)

    async def _tail_stderr(self, entry: _RunningAudio) -> None:
        assert entry.process.stderr is not None
        try:
            async for chunk in entry.process.stderr:
                if chunk:
                    self._stderr_line(entry.config.name, chunk)
        except asyncio.CancelledError:
            raise
        rc = await entry.process.wait()
        if entry.state is AudioState.READY and rc not in (0, -signal.SIGTERM):
            entry.state = AudioState.FAILED
            entry.detail = f"{self.role} server exited with code {rc}"
            logger.error(
                "%s server for '%s' exited unexpectedly (code %s)",
                self.role, entry.config.name, rc,
            )

    async def _idle_watcher(self) -> None:
        """Stop the loaded audio server once it has been idle past the timeout."""
        try:
            while True:
                await asyncio.sleep(self._health_interval)
                if self._current is None:
                    continue
                elapsed = asyncio.get_running_loop().time() - self._last_activity
                if elapsed < self._idle_unload_seconds:
                    continue
                async with self._lock:
                    cur = self._current
                    if cur is None or cur.state is not AudioState.READY:
                        continue
                    elapsed = asyncio.get_running_loop().time() - \
                        self._last_activity
                    if elapsed < self._idle_unload_seconds:
                        continue
                    logger.info(
                        "unloading %s model '%s' after %.0fs idle",
                        self.role, cur.config.name, elapsed,
                    )
                    await self._stop_current()
        except asyncio.CancelledError:
            raise

    async def shutdown(self) -> None:
        if self._idle_task is not None:
            self._idle_task.cancel()
            try:
                await self._idle_task
            except asyncio.CancelledError:
                pass
        async with self._lock:
            await self._stop_current()


def _cleanup_config(path: Optional[str]) -> None:
    if not path:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass
