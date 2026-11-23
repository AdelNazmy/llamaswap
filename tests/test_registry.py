"""Registry rendering: placeholders, unset env vars, typed values."""

from app.registry import ModelConfig


def _cfg(config_json: dict) -> ModelConfig:
    return ModelConfig(
        name="probe",
        role="llm",
        command={
            "binary": "/bin/true",
            "args": ["serve", "--profile", "{config_path}"],
            "config_json": config_json,
        },
        host="127.0.0.1",
        port=8123,
    )


def test_unresolved_env_placeholder_is_left_verbatim():
    # ``${VAR}`` for an unset variable stays as-is after _expand_env; the
    # render step must not turn it into a KeyError (HTTP 500).
    cfg = _cfg({"exe": "component:engine", "args": ["${UNSET-VAR}/bin/x"]})
    payload = cfg.render_config_json("/tmp/profile.json")
    assert payload["args"][0] == "${UNSET-VAR}/bin/x"


def test_host_port_and_config_path_placeholders_render():
    cfg = _cfg(
        {
            "port": "{port}",
            "host": "{host}",
            "model": "{name}",
            "log": "{config_path}.log",
            "spec": "4",
        }
    )
    payload = cfg.render_config_json("/tmp/profile.json")
    assert payload["port"] == 8123
    assert payload["host"] == "127.0.0.1"
    assert payload["model"] == "probe"
    assert payload["log"] == "/tmp/profile.json.log"
    # Numeric-looking strings that were not wholly a placeholder stay strings.
    assert payload["spec"] == "4"
