"""Model registry: loads per-model YAML launch configs from the backend dir."""

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger("llamaswap.registry")


class CommandSpec(BaseModel):
    """A llama-server launch command (binary + argv + optional env)."""

    binary: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)


class ModelMeta(BaseModel):
    """Free-form metadata surfaced to clients."""

    context_length: int = 0
    family: str = ""
    capabilities: list[str] = Field(default_factory=list)
    model_config = {"extra": "allow"}


class ModelConfig(BaseModel):
    """Full definition of a hostable model, read from backend/<name>.yaml."""

    name: str
    description: str = ""
    command: CommandSpec
    host: str = "127.0.0.1"
    port: int = 8101
    meta: ModelMeta = Field(default_factory=ModelMeta)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        v = v.strip()
        if not v or " " in v:
            raise ValueError("model name must be a non-empty token without spaces")
        return v

    def build_argv(self) -> list[str]:
        """argv for the llama-server subprocess, with host/port enforced."""
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
            argv.append(a)
            i += 1
        argv += ["--host", self.host, "--port", str(self.port)]
        return argv


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
        self.reload()

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
        logger.info(
            "registry loaded %d model(s) from %s: %s",
            len(models),
            self.backend_dir,
            ", ".join(models) or "(none)",
        )

    def get(self, name: str) -> ModelConfig:
        cfg = self.models.get(name)
        if cfg is None:
            raise UnknownModelError(name, list(self.models))
        return cfg

    def names(self) -> list[str]:
        return list(self.models)

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
            "context_length": meta.context_length,
            "family": meta.family,
            "capabilities": meta.capabilities,
        }
