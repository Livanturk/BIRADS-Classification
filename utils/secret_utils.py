"""Helpers for resolving secrets without logging them."""

import os
from typing import Iterable, Optional


SENSITIVE_CONFIG_KEYS = {
    "access_token",
    "api_key",
    "api_token",
    "dagshub_token",
    "password",
    "private_key",
    "secret",
    "token",
}
SENSITIVE_CONFIG_SUFFIXES = (
    "_api_key",
    "_password",
    "_private_key",
    "_secret",
    "_token",
)


def get_env_or_config(
    config_value: Optional[str],
    env_names: Iterable[str],
    default: str = "",
) -> str:
    """Resolve a value from env first, then config, expanding ${ENV_NAME}."""
    for env_name in env_names:
        env_value = os.getenv(env_name)
        if env_value:
            return env_value

    if not config_value:
        return default

    if isinstance(config_value, str):
        value = config_value.strip()
        if value.startswith("${") and value.endswith("}"):
            return os.getenv(value[2:-1], default)
        return value

    return str(config_value)


def is_sensitive_config_key(key: str) -> bool:
    """Return True when a flattened config key should never be logged."""
    leaf_key = key.rsplit(".", 1)[-1].lower()
    return (
        leaf_key in SENSITIVE_CONFIG_KEYS
        or leaf_key.endswith(SENSITIVE_CONFIG_SUFFIXES)
    )
