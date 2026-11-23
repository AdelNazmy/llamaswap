"""Process manager: fleet alias switching and the fleet readiness probe.

A fleet engine (Radiance) serves N registered names from ONE process,
selected by the client's model field. The process manager must treat
same-argv registry entries as aliases of the running process (no
SIGTERM/restart on an alias switch), and must wait for the named model's
residency when the engine declares a ready probe (meta.ready_path).
"""

import asyncio
import json
import logging
import signal
from pathlib import Path

from app.process_manager import ModelState, ProcessManager, RunningModel
from app.registry import ModelConfig, Registry


class FakeRegistry:
    def __init__(self, cfgs):
        self._cfgs = cfgs

    def get(self, name):
        if name not in self._cfgs:
            raise KeyError(name)
        return self._cfgs[name]


class FakeProc:
    """Records the signals it receives; exits immediately on SIGTERM."""

    def __init__(self):
        self.returncode = None
        self.pid = 1234
        self.signals = []
        self.killed = False

    def send_signal(self, sig):
        self.signals.append(sig)
        self.returncode = 0

    def kill(self):
        self.killed = True
        self.returncode = -signal.SIGKILL

    async def wait(self):
        return self.returncode


def make_cfg(name, binary="/opt/rad/start.sh", args=None, port=8000):
    return ModelConfig(
        name=name,
        command={"binary": binary, "args": args or []},
        host="127.0.0.1",
        port=port,
        meta={"health_path": "/v1/health", "vram_gb": 30.5,
              "idle_unload_seconds": 0},
    )


def make_manager(cfgs, running):
    """A ProcessManager with `running` (name, cfg, proc) already READY, or
    None. The real _launch is replaced so no subprocess is ever spawned;
    it records the names it was asked to launch. MUST be called from
    within a running event loop (the constructor starts the idle
    watcher)."""
    manager = ProcessManager(startup_timeout=600, stop_timeout=5,
                             health_interval=1.0)
    launched = []

    async def fake_launch(cfg):
        launched.append(cfg.name)
        manager._current = RunningModel(
            name=cfg.name, config=cfg, process=FakeProc(),
            state=ModelState.READY,
        )

    manager._launch = fake_launch
    if running is not None:
        name, cfg, proc = running
        manager._current = RunningModel(name=name, config=cfg, process=proc,
                                        state=ModelState.READY)
    return manager, launched


def test_same_name_no_restart():
    async def case():
        a = make_cfg("qwen3.8-27b-iq4r-v4")
        proc = FakeProc()
        manager, launched = make_manager({"a": a}, running=("a", a, proc))
        name, port = await manager.ensure_model("a", FakeRegistry({"a": a}))
        assert (name, port) == ("a", 8000)
        assert proc.signals == []
        assert launched == []

    asyncio.run(case())


def test_alias_same_argv_no_restart():
    """Switching to another registered name of the SAME fleet (identical
    launch argv) must not SIGTERM/restart the running process."""
    async def case():
        a = make_cfg("qwen3.8-27b-iq4r-v4")
        b = make_cfg("qwen3.6-35b-a3b")  # same binary+args -> same argv
        proc = FakeProc()
        manager, launched = make_manager(
            {"a": a, "b": b}, running=("a", a, proc))
        name, port = await manager.ensure_model("b", FakeRegistry(
            {"a": a, "b": b}))
        assert (name, port) == ("b", 8000)
        assert proc.signals == []          # no SIGTERM
        assert not proc.killed
        assert launched == []              # no restart
        # The bookkeeping follows the alias; the process is untouched.
        assert manager._current is not None
        assert manager._current.name == "b"
        assert manager._current.process is proc

    asyncio.run(case())


def test_alias_switch_back_and_forth():
    async def case():
        a = make_cfg("qwen3.8-27b-iq4r-v4")
        b = make_cfg("qwen3.6-35b-a3b")
        proc = FakeProc()
        manager, launched = make_manager(
            {"a": a, "b": b}, running=("a", a, proc))
        reg = FakeRegistry({"a": a, "b": b})
        await manager.ensure_model("b", reg)
        await manager.ensure_model("a", reg)
        assert manager._current.name == "a"
        assert manager._current.process is proc
        assert proc.signals == [] and launched == []

    asyncio.run(case())


def test_different_argv_restarts():
    """A genuinely different backend (different argv) still swaps the
    process the old way: SIGTERM the current, launch the new one."""
    async def case():
        a = make_cfg("qwen3.8-27b-iq4r-v4")
        c = make_cfg(
            "qwen3.8-27b-ridge",
            binary="/opt/llama.cpp/build/bin/llama-server",
            args=["--model", "/opt/models/x.gguf"], port=8101,
        )
        proc = FakeProc()
        manager, launched = make_manager(
            {"a": a, "c": c}, running=("a", a, proc))
        name, port = await manager.ensure_model("c", FakeRegistry(
            {"a": a, "c": c}))
        assert (name, port) == ("c", 8101)
        assert proc.signals == [signal.SIGTERM]  # the old process stopped
        assert launched == ["qwen3.8-27b-ridge"]  # the new one launched

    asyncio.run(case())


def test_loading_current_ignores_alias():
    """An alias check only applies to a READY process: a loading one is
    still stopped and replaced as before."""
    async def case():
        a = make_cfg("qwen3.8-27b-iq4r-v4")
        b = make_cfg("qwen3.6-35b-a3b")
        proc = FakeProc()
        manager, launched = make_manager(
            {"a": a, "b": b}, running=("a", a, proc))
        manager._current.state = ModelState.LOADING
        await manager.ensure_model("b", FakeRegistry({"a": a, "b": b}))
        assert proc.signals == [signal.SIGTERM]
        assert launched == ["qwen3.6-35b-a3b"]

    asyncio.run(case())


def test_fleet_resident():
    body = json.dumps({"models": [
        {"id": "a", "state": "loading"},
        {"id": "b", "state": "resident"},
    ]}).encode()
    assert ProcessManager._fleet_resident(body, "b")
    assert not ProcessManager._fleet_resident(body, "a")
    assert not ProcessManager._fleet_resident(body, "c")
    assert not ProcessManager._fleet_resident(b"not json", "b")
    assert not ProcessManager._fleet_resident(b"", "b")
    assert not ProcessManager._fleet_resident(
        json.dumps({"models": []}).encode(), "b")
    assert not ProcessManager._fleet_resident(
        json.dumps({"models": [{"id": "b", "state": "parked"}]}).encode(),
        "b")
    assert not ProcessManager._fleet_resident(
        json.dumps({"cards": [{"id": "b", "state": "resident"}]}).encode(),
        "b")


def test_log_lines_dedups_unchanged_heartbeats(caplog):
    """A heartbeat backend (Radiance's live view) emits one key=value
    line per ~250 ms with unchanged values while idle: consecutive
    duplicates must be suppressed, while each change — including a
    return to an earlier value — reaches the log."""
    caplog.set_level(logging.INFO)

    async def case():
        a = make_cfg("qwen3.8-27b-iq4r-v4")
        manager, _ = make_manager({"a": a}, running=("a", a, FakeProc()))
        entry = manager._current
        idle = b"live step=113 run=0 wait=0 kv=0.00 prefix_hit=0.00\n"
        manager._log_lines(entry, idle)
        manager._log_lines(entry, idle)
        manager._log_lines(entry, idle)
        manager._log_lines(entry, b"live step=114 run=1 wait=0 kv=0.42\n")
        # Idle again: the line reappears (only CONSECUTIVE duplicates are
        # suppressed), and a trailing blank line does not affect it.
        manager._log_lines(entry, idle + b"\n")
        return [
            r.getMessage()
            for r in caplog.records
            if r.name == "llamaswap.process_manager"
            and r.levelno == logging.INFO
        ]

    lines = asyncio.run(case())
    assert lines == [
        "llama-server[a]: live step=113 run=0 wait=0 "
        "kv=0.00 prefix_hit=0.00",
        "llama-server[a]: live step=114 run=1 wait=0 "
        "kv=0.42",
        "llama-server[a]: live step=113 run=0 wait=0 "
        "kv=0.00 prefix_hit=0.00",
    ]


def test_radiance_registry_entry():
    """The real backend directory must load the radiance entry intact."""
    reg = Registry(Path(__file__).resolve().parent.parent / "backend")
    cfg = reg.get("qwen3.8-27b-iq4r-v4")
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 8000
    assert cfg.meta.health_path == "/v1/health"
    assert cfg.meta.ready_path == "/admin/fleet"
    assert cfg.meta.vram_gb == 30.5
    assert cfg.meta.idle_unload_seconds == 0
    # start.sh never forwards argv, so the appended host/port flags only
    # need to be harmless extras for it — and must match qwen.yaml's port.
    assert cfg.build_argv() == [
        "/home/adnathan/radiance/start.sh",
        "--host", "127.0.0.1",
        "--port", "8000",
    ]
