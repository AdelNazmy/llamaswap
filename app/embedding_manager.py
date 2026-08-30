"""Lives/monitors the persistent embedding llama-server subprocess.

Started at app boot and kept running for the lifetime of the proxy so
that embedding requests never have to wait on the LLM swap path. The
one sanctioned shutdown is stop_for_resources(), used to free VRAM
when a chat LLM or the image server needs the GPU; the caller is
expected to best-effort relaunch afterwards via ensure_running().
"""

import asyncio
import logging
import os
import signal
from enum import Enum
from typing import Any, Optional

import httpx

from .registry import ModelConfig

logger = logging.getLogger("llamaswap.embedding_manager")


class EmbedState(str, Enum):
    LOADING = "loading"
    READY = "ready"
    STOPPED = "stopped"
    FAILED = "failed"


class EmbeddingLoadError(Exception):
    """Raised when the embedding server could not be brought up."""


class EmbeddingManager:
    """Starts, monitors and (re)launches the persistent embedding server."""

    def __init__(self, config: ModelConfig, startup_timeout: float,
                 stop_timeout: float, health_interval: float):
        self.config = config
        self.name = config.name
        self._startup_timeout = startup_timeout
        self._stop_timeout = stop_timeout
        self._health_interval = health_interval
        self._lock = asyncio.Lock()
        self._process: Optional[asyncio.subprocess.Process] = None
        self._stderr_task: Optional[asyncio.Task] = None
        self._state = EmbedState.STOPPED
        self._detail = ""
        self._stopped_for_resources = False

    @property
    def is_running(self) -> bool:
        return self._state is EmbedState.READY

    @property
    def is_loading(self) -> bool:
        return self._state is EmbedState.LOADING

    def status(self) -> dict[str, Any]:
        return {
            "state": self._state.value,
            "model": self.name,
            "port": self.config.port,
            "stopped_for_resources": self._stopped_for_resources,
            "detail": self._detail,
        }

    def _health_url(self) -> str:
        return f"http://{self.config.host}:{self.config.port}/health"

    def _stderr_line(self, chunk: bytes) -> None:
        for line in chunk.decode(errors="replace").splitlines():
            line = line.strip()
            if line:
                logger.info("llama-server[embed %s]: %s", self.name, line)

    async def _wait_healthy(self, proc: asyncio.subprocess.Process) -> None:
        deadline = asyncio.get_running_loop().time() + self._startup_timeout
        async with httpx.AsyncClient(timeout=5.0) as client:
            while True:
                if proc.returncode is not None:
                    raise EmbeddingLoadError(
                        f"embedding server '{self.name}' exited (code "
                        f"{proc.returncode}) during startup; see logs above"
                    )
                try:
                    r = await client.get(self._health_url())
                    if r.status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                if asyncio.get_running_loop().time() > deadline:
                    raise EmbeddingLoadError(
                        f"embedding server '{self.name}' did not become "
                        f"healthy within {self._startup_timeout:.0f}s"
                    )
                await asyncio.sleep(self._health_interval)

    async def _stop_locked(self) -> None:
        """Stop the subprocess; caller must hold self._lock."""
        proc = self._process
        self._process = None
        if self._stderr_task is not None:
            self._stderr_task.cancel()
            self._stderr_task = None
        if proc is None or proc.returncode is not None:
            return
        logger.info("stopping embedding server '%s' (pid %s)", self.name, proc.pid)
        try:
            proc.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(proc.wait(), timeout=self._stop_timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "embedding server '%s' did not exit, killing", self.name
            )
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()
        logger.info("embedding server '%s' stopped", self.name)

    async def start(self) -> None:
        """Launch the embedding server and wait until it is healthy."""
        async with self._lock:
            # A startup from another coroutine may have finished while we
            # waited for the lock; don't clobber a healthy server.
            if self._state is EmbedState.READY and self._process is not None \
                    and self._process.returncode is None:
                return
            await self._stop_locked()
            argv = self.config.build_argv()
            logger.info(
                "launching embedding server '%s': %s", self.name, " ".join(argv)
            )
            env = {**os.environ, **self.config.command.env}
            try:
                proc = await asyncio.create_subprocess_exec(
                    *argv,
                    env=env,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.PIPE,
                )
            except OSError as exc:
                self._state = EmbedState.FAILED
                self._detail = f"failed to spawn llama-server: {exc}"
                raise EmbeddingLoadError(self._detail) from exc
            self._process = proc
            self._state = EmbedState.LOADING
            self._detail = ""
            # Drain stderr immediately so the pipe never fills up during
            # the (potentially long) model load.
            self._stderr_task = asyncio.get_running_loop().create_task(
                self._tail_stderr(proc)
            )
            try:
                await self._wait_healthy(proc)
            except BaseException:
                await self._stop_locked()
                if self._state is not EmbedState.FAILED:
                    self._state = EmbedState.FAILED
                raise
            self._state = EmbedState.READY
            self._detail = ""
            self._stopped_for_resources = False
            logger.info(
                "embedding server '%s' ready on port %d", self.name, self.config.port
            )

    async def stop(self) -> None:
        """Stop the embedding server (normal, e.g. app shutdown)."""
        async with self._lock:
            was_running = self._state in (EmbedState.READY, EmbedState.LOADING)
            await self._stop_locked()
            if was_running:
                self._state = EmbedState.STOPPED
                self._detail = ""

    async def stop_for_resources(self, reason: str = "an LLM") -> None:
        """Stop the embedding server to free VRAM for another backend.

        ``reason`` is surfaced in the /health ``detail`` field (e.g.
        "an LLM" or "the image model").
        """
        async with self._lock:
            was_running = self._state in (EmbedState.READY, EmbedState.LOADING)
            await self._stop_locked()
            if was_running:
                self._state = EmbedState.STOPPED
                self._detail = f"stopped to make room for {reason}"
            self._stopped_for_resources = True
            if was_running:
                logger.info(
                    "embedding server '%s' stopped to free resources for %s",
                    self.name, reason,
                )

    async def _wait_loading_settled(self) -> bool:
        """Wait for an in-flight load to finish; return True if it became ready."""
        # Bounded wait: the loader itself enforces self._startup_timeout and
        # will move the state to READY/FAILED, so this is a safety net only.
        deadline = asyncio.get_running_loop().time() + self._startup_timeout + 30.0
        while self._state is EmbedState.LOADING:
            if asyncio.get_running_loop().time() > deadline:
                logger.warning(
                    "embedding server '%s' still loading after %.0fs; "
                    "restarting", self.name, self._startup_timeout,
                )
                return False
            await asyncio.sleep(0.25)
        return self.is_running

    async def ensure_running(self, *, restart_loading: bool = False) -> bool:
        """Best-effort: make sure the embedding server is up.

        If a load is already in flight it is left alone (unless
        ``restart_loading`` is set) and we simply wait for it to settle.
        Returns True if the server is (now) ready, False otherwise.
        Never raises (except on cancellation).
        """
        if self.is_running:
            return True
        if self.is_loading and not restart_loading:
            if await self._wait_loading_settled():
                return True
        try:
            await self.start()
            return True
        except EmbeddingLoadError as exc:
            logger.warning(
                "embedding server '%s' not running (%s)", self.name, exc
            )
            return False
        except asyncio.CancelledError:
            raise

    async def _tail_stderr(self, proc: asyncio.subprocess.Process) -> None:
        assert proc.stderr is not None
        try:
            async for chunk in proc.stderr:
                if chunk:
                    self._stderr_line(chunk)
        except asyncio.CancelledError:
            raise
        rc = await proc.wait()
        if self._state is EmbedState.READY and rc not in (0, -signal.SIGTERM):
            self._state = EmbedState.FAILED
            self._detail = f"llama-server exited with code {rc}"
            logger.error(
                "embedding server '%s' exited unexpectedly (code %s)",
                self.name, rc,
            )

    async def shutdown(self) -> None:
        await self.stop()
