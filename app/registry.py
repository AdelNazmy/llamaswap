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
    model_config = {"extra": "allow"}


ROLES = {"llm", "embedding", "tts", "asr"}
# Roles served by the ProcessManager (one process, swapped on request).
SWAP_ROLES = {"llm"}
# Roles served by dedicated managers. Only "embedding" is persistent
# (boots with the proxy); "tts"/"asr" are on-demand and idle-unloaded
# like the chat LLM, so nothing else runs without a request.
PERSISTENT_ROLES = {"embedding", "tts", "asr"}


class ModelConfig(BaseModel):
    """Full definition of a hostable model, read from backend/<name>.yaml.

    role:
      "llm"       — swapped in/out by the ProcessManager
      "embedding" — persistent embedding server (EmbeddingManager)
      "tts"       — persistent TTS server (AudioManager)
      "asr"       — persistent ASR server (AudioManager)
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

    def build_argv(
        self, config_path: str | None = None,
    ) -> list[str]:
        """argv for the backend subprocess, with host/port enforced.

        ``config_path`` (a rendered config JSON file, if any) is substituted
        for a ``{config_path}`` placeholder in args.
        """
        argv = [self.command.binary]
        args = list(self.command.args)
        # Strip any user-provided --host/--port (value or = form) so the
        # manager's host/port always win.
        i = 0
        while i < len(args):
            a = args[i]
            if a in ("--host", "--port") and i + 1 < len(args):
                i += 2
                continue
            if a.startswith("--host=") or a.startswith("--port="):
                i += 1
                continue
            argv.append(a.replace("{config_path}", config_path or ""))
            i += 1
        argv += ["--host", self.host, "--port", str(self.port)]
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
        self._llm_sizes: dict[str, int] = {}
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
        for path in sorted(self.backend_dir.glob("*.y*ml")):
            try:
                raw: Any = yaml.safe_load(path.read_text()) or {}
                cfg = ModelConfig.model_validate(raw)
            except Exception as exc:  # noqa: BLE001
                raise RegistryError(f"invalid model config {path.name}: {exc}") from exc
            models[cfg.name] = cfg
        self.models = models
        # Size (bytes) of each chat LLM's weights file, used by
        # smallest_llm() as the proxy for VRAM footprint.
        llm_sizes: dict[str, int] = {}
        for name, cfg in models.items():
            if cfg.role != "llm":
                continue
            path = self._model_path(cfg)
            if path:
                try:
                    llm_sizes[name] = Path(path).stat().st_size
                except OSError:
                    logger.warning(
                        "cannot stat model file for '%s': %s", name, path
                    )
        self._llm_sizes = llm_sizes
        logger.info(
            "registry loaded %d model(s) from %s: %s",
            len(models),
            self.backend_dir,
            ", ".join(models) or "(none)",
        )

    def smallest_llm(self) -> Optional[str]:
        """Name of the chat (role: llm) model with the smallest weights
        file on disk — the cheapest VRAM footprint — or None if there are
        no size-ranked chat models. Ties resolve to the first in sorted
        registry order."""
        if not self._llm_sizes:
            return None
        return min(self._llm_sizes, key=lambda name: self._llm_sizes[name])

    def get(self, name: str) -> ModelConfig:
        cfg = self.models.get(name)
        if cfg is None:
            raise UnknownModelError(name, list(self.models))
        return cfg

    def names(self) -> list[str]:
        return list(self.models)

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
        }
