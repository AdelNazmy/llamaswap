"""Model registry: loads per-model YAML launch configs from the backend dir."""

import logging
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger("llamaswap.registry")


class CommandSpec(BaseModel):
    """A backend server launch command (binary + argv + optional env).

    ``config_json`` optionally holds a JSON object that is rendered with
    ``str.format()`` placeholders and written to a temp file before launch;
    the rendered file path is substituted for ``{config_path}`` in ``args``.
    This is how structured-config backends (e.g. audio.cpp's server) get a
    per-model config file whose host/port match the enforced YAML values.
    """

    binary: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    config_json: dict[str, Any] | None = None
    # Flag spellings for the listen address. Most backends use --host/--port;
    # stable-diffusion.cpp's sd-server uses --listen-ip/--listen-port.
    host_arg: str = "--host"
    port_arg: str = "--port"


class ModelMeta(BaseModel):
    """Free-form metadata surfaced to clients."""

    context_length: int = 0
    family: str = ""
    capabilities: list[str] = Field(default_factory=list)
    # How the proxy should talk to this backend over the OpenAI audio API:
    #   "json"       — JSON request bodies, binary/JSON responses (llama-server, audio.cpp)
    #   "multipart"  — multipart/form-data uploads (whisper.cpp whisper-server)
    #   "json_path"  — JSON body whose audio is a server-side file path; the
    #                  proxy saves the uploaded file and rewrites the request
    #                  (audio.cpp /v1/audio/transcriptions)
    request_format: str = "json"
    # Health endpoint checked by the process manager (default: /health).
    health_path: str = "/health"
    # Declared VRAM footprint (GiB) used by the VRAM scheduler/guards. When
    # unset, the registry falls back to the on-disk weights-file size.
    vram_gb: Optional[float] = None
    # Pin this backend to a specific GPU: overrides CUDA_VISIBLE_DEVICES at
    # launch, so chat/embed/audio/image can run concurrently on different
    # cards of a multi-GPU box.
    gpu: Optional[int] = None
    model_config = {"extra": "allow"}


ROLES = {"llm", "embedding", "tts", "asr", "image"}
# Request dialects the proxy knows how to translate to/from.
REQUEST_FORMATS = {"json", "multipart", "json_path"}


class ModelConfig(BaseModel):
    """Full definition of a hostable model, read from backend/<name>.yaml.

    role:
      "llm"       — swapped in/out by the ProcessManager
      "embedding" — persistent embedding server (EmbeddingManager)
      "tts"       — on-demand TTS server (RoleServerManager)
      "asr"       — on-demand ASR server (RoleServerManager)
      "image"     — on-demand image-generation server (RoleServerManager)
    """

    name: str
    description: str = ""
    role: str = "llm"
    command: CommandSpec
    host: str = "127.0.0.1"
    port: int = 8101
    meta: ModelMeta = Field(default_factory=ModelMeta)

    @field_validator("role")
    @classmethod
    def _validate_role(cls, v: str) -> str:
        v = v.strip()
        if v not in ROLES:
            raise ValueError(
                f"unknown role '{v}'; expected one of {sorted(ROLES)}"
            )
        return v

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        v = v.strip()
        if not v or " " in v:
            raise ValueError("model name must be a non-empty token without spaces")
        return v

    def health_url(self) -> str:
        return f"http://{self.host}:{self.port}{self.meta.health_path}"

    def launch_env(self, base: dict[str, str]) -> dict[str, str]:
        """Environment for the backend subprocess.

        Merges the per-model ``command.env`` over ``base`` (usually
        ``os.environ``), then applies the ``meta.gpu`` pin (if any) as a
        ``CUDA_VISIBLE_DEVICES`` override — the one universal per-backend
        device pin across llama-server / audiocpp_server / whisper-server /
        sd-server.
        """
        env = {**base, **self.command.env}
        if self.meta.gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.meta.gpu)
        return env

    def build_argv(
        self, config_path: str | None = None,
    ) -> list[str]:
        """argv for the backend subprocess, with host/port enforced.

        ``config_path`` (a rendered config JSON file, if any) is substituted
        for a ``{config_path}`` placeholder in args. The host/port flag
        spellings are configurable per command via ``command.host_arg`` /
        ``command.port_arg`` (default ``--host``/``--port``); some backends
        spell them differently (e.g. sd-server's ``--listen-ip`` /
        ``--listen-port``).
        """
        host_arg = self.command.host_arg
        port_arg = self.command.port_arg
        argv = [self.command.binary]
        args = list(self.command.args)
        # Strip any user-provided host/port flag (value or = form) so the
        # manager's host/port always win.
        i = 0
        while i < len(args):
            a = args[i]
            if a in (host_arg, port_arg) and i + 1 < len(args):
                i += 2
                continue
            if a.startswith(f"{host_arg}=") or a.startswith(f"{port_arg}="):
                i += 1
                continue
            argv.append(a.replace("{config_path}", config_path or ""))
            i += 1
        argv += [host_arg, self.host, port_arg, str(self.port)]
        return argv

    def render_config_json(self, config_path: str) -> dict[str, Any]:
        """Render ``command.config_json`` with host/port baked in.

        Placeholders inside the JSON structure, values, and keys are
        substituted: ``{host}``, ``{port}``, ``{name}``, ``{config_path}``.
        """
        raw = self.command.config_json or {}

        def _sub(value: Any) -> Any:
            if isinstance(value, str):
                return value.format(
                    host=self.host, port=self.port, name=self.name,
                    config_path=config_path,
                )
            if isinstance(value, dict):
                return {_sub(k): _sub(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_sub(v) for v in value]
            return value

        cfg = _sub(raw)
        # The backend's own host/port always follow the enforced YAML values.
        cfg["host"] = self.host
        cfg["port"] = self.port
        return _coerce_num_strings(cfg)


def _coerce_num_strings(value: Any) -> Any:
    """Turn numeric strings (e.g. from `{port}` rendering) back into ints."""
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            try:
                f = float(value)
            except ValueError:
                return value
            return int(f) if f.is_integer() else f
    if isinstance(value, dict):
        return {k: _coerce_num_strings(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_coerce_num_strings(v) for v in value]
    return value


class RegistryError(Exception):
    """Raised when the registry cannot be loaded."""


class UnknownModelError(Exception):
    """Raised when a request references a model not in the registry."""

    def __init__(self, name: str, known: list[str]):
        self.name = name
        self.known = known
        super().__init__(f"unknown model '{name}'; known models: {', '.join(known)}")


class Registry:
    """In-memory view of all YAML model configs in the backend directory."""

    def __init__(self, backend_dir: str | Path):
        self.backend_dir = Path(backend_dir)
        self.models: dict[str, ModelConfig] = {}
        self._llm_vram: dict[str, float] = {}
        self.warnings: list[str] = []
        self.degraded: list[str] = []
        self.reload()

    @staticmethod
    def _model_path(cfg: ModelConfig) -> Optional[str]:
        """Weights file path from the config's --model arg, if present."""
        args = cfg.command.args
        for i, a in enumerate(args):
            if a == "--model" and i + 1 < len(args):
                return args[i + 1]
            if a.startswith("--model="):
                return a.split("=", 1)[1]
        return None

    def reload(self) -> None:
        if not self.backend_dir.is_dir():
            raise RegistryError(f"backend directory not found: {self.backend_dir}")
        models: dict[str, ModelConfig] = {}
        warnings: list[str] = []
        seen_names: dict[str, str] = {}
        # (host, port) -> (name, role): the same listen address is only
        # allowed within ONE role (chat LLMs all reuse 8101 legitimately
        # because they are swapped by a single manager; two *different*
        # roles on one port is always a collision).
        seen_ports: dict[tuple[str, int], tuple[str, str]] = {}
        for path in sorted(self.backend_dir.glob("*.y*ml")):
            try:
                raw: Any = yaml.safe_load(path.read_text()) or {}
                cfg = ModelConfig.model_validate(raw)
            except Exception as exc:  # noqa: BLE001
                raise RegistryError(f"invalid model config {path.name}: {exc}") from exc
            if cfg.name in seen_names:
                raise RegistryError(
                    f"duplicate model name '{cfg.name}' in {path.name} "
                    f"(already defined in {seen_names[cfg.name]})"
                )
            if cfg.meta.request_format not in REQUEST_FORMATS:
                raise RegistryError(
                    f"{path.name}: unknown request_format "
                    f"'{cfg.meta.request_format}'; expected one of "
                    f"{sorted(REQUEST_FORMATS)}"
                )
            key = (cfg.host, cfg.port)
            if key in seen_ports and seen_ports[key][1] != cfg.role:
                other_name, other_role = seen_ports[key]
                raise RegistryError(
                    f"{path.name}: listen address {cfg.host}:{cfg.port} of "
                    f"role '{cfg.role}' collides with '{other_name}' "
                    f"(role '{other_role}') on the same port"
                )
            if not Path(cfg.command.binary).exists():
                warnings.append(
                    f"{path.name}: binary not found: {cfg.command.binary}"
                )
            seen_names[cfg.name] = path.name
            seen_ports[key] = (cfg.name, cfg.role)
            models[cfg.name] = cfg
        self.models = models
        self.warnings = warnings
        self.degraded = []
        # VRAM ranking for the guards: declared meta.vram_gb when present,
        # otherwise the on-disk weights-file size (MiB) as an estimate.
        llm_vram: dict[str, float] = {}
        for name, cfg in models.items():
            if cfg.role != "llm":
                continue
            if cfg.meta.vram_gb is not None:
                llm_vram[name] = cfg.meta.vram_gb * 1024.0  # GiB -> MiB
                continue
            path = self._model_path(cfg)
            if path:
                try:
                    llm_vram[name] = Path(path).stat().st_size / (1024 * 1024)
                except OSError:
                    logger.warning(
                        "cannot stat model file for '%s': %s", name, path
                    )
        self._llm_vram = llm_vram
        llms = [c.name for c in models.values() if c.role == "llm"]
        if llms and not llm_vram:
            msg = (
                "no chat LLM has a known VRAM estimate (weights missing, "
                "unstat-able, or no meta.vram_gb); the audio/image VRAM "
                "guards are disabled"
            )
            self.degraded.append(msg)
            logger.warning(msg)
        if warnings:
            logger.warning(
                "registry %d warning(s): %s", len(warnings), "; ".join(warnings)
            )
        logger.info(
            "registry loaded %d model(s) from %s: %s",
            len(models),
            self.backend_dir,
            ", ".join(models) or "(none)",
        )

    def smallest_llm(self) -> Optional[str]:
        """Name of the chat (role: llm) model with the smallest VRAM
        footprint (declared ``meta.vram_gb``, falling back to weights-file
        size), or None when no chat LLM has a known size — in which case
        the audio/image VRAM guards are deliberately disabled and the
        degraded state is surfaced in ``registry.degraded``."""
        if not self._llm_vram:
            return None
        return min(self._llm_vram, key=lambda name: self._llm_vram[name])

    def llm_vram_mb(self) -> dict[str, float]:
        """VRAM ranking (MiB) for every size-known chat LLM."""
        return dict(self._llm_vram)

    def vram_mb(self, name: str) -> Optional[float]:
        """VRAM estimate (MiB) for any registry model, when known."""
        if name in self._llm_vram:
            return self._llm_vram[name]
        cfg = self.models.get(name)
        if cfg is None:
            return None
        if cfg.meta.vram_gb is not None:
            return cfg.meta.vram_gb * 1024.0
        return None

    def get(self, name: str) -> ModelConfig:
        cfg = self.models.get(name)
        if cfg is None:
            raise UnknownModelError(name, list(self.models))
        return cfg

    def names(self) -> list[str]:
        return list(self.models)

    def model_path(self, name: str) -> Optional[str]:
        """On-disk weights file path for ``name``, if its command names one."""
        cfg = self.models.get(name)
        if cfg is None:
            return None
        return self._model_path(cfg)

    def embedding_config(self) -> Optional[ModelConfig]:
        """The dedicated embedding model config, or None if not defined."""
        return self._first_role("embedding")

    def role_configs(self, role: str) -> list[ModelConfig]:
        """All configs for a given role, sorted by name."""
        return sorted(
            (cfg for cfg in self.models.values() if cfg.role == role),
            key=lambda cfg: cfg.name,
        )

    def audio_roles(self) -> list[str]:
        """The audio roles (tts/asr) that have at least one config."""
        return [r for r in ("tts", "asr") if self.role_configs(r)]

    def _first_role(self, role: str) -> Optional[ModelConfig]:
        for cfg in sorted(self.models.values(), key=lambda c: c.name):
            if cfg.role == role:
                return cfg
        return None

    def list_openai(self) -> list[dict[str, Any]]:
        return [self._to_openai(cfg) for cfg in self.models.values()]

    def to_openai(self, name: str) -> dict[str, Any]:
        return self._to_openai(self.get(name))

    @staticmethod
    def _to_openai(cfg: ModelConfig) -> dict[str, Any]:
        meta = cfg.meta
        return {
            "id": cfg.name,
            "object": "model",
            "created": 0,
            "owned_by": "llamaswap",
            "description": cfg.description,
            "role": cfg.role,
            "context_length": meta.context_length,
            "family": meta.family,
            "capabilities": meta.capabilities,
            "vram_gb": meta.vram_gb,
            "gpu": meta.gpu,
        }
