from __future__ import annotations

from backend import config

CONFIG_RUNTIME_RULES: dict[str, dict] = {}
CONFIG_RESTART_REQUIRED: set[str] = set()


def iter_config_keys() -> list[str]:
    return sorted(key for key, value in vars(config).items() if key.isupper() and not key.startswith('_') and not callable(value))


def build_config_payload():
    values = {}
    meta = {}
    for key in iter_config_keys():
        value = getattr(config, key)
        values[key] = value
        meta[key] = {'key': key, 'label': key, 'group': 'general', 'type': type(value).__name__, 'editable': False, 'restartRequired': False, 'description': '', 'reason': 'readonly'}
    return {'success': True, 'values': values, 'meta': meta, 'groups': ['general'], 'editableKeys': [], 'readonlyKeys': sorted(values.keys())}


def validate_runtime_config_updates(updates: dict):
    return {}, {k: 'readonly' for k in updates}


def apply_runtime_config_updates(_updates: dict, _session_registry):
    return None


def apply_runtime_config_command(_command_line: str, _session_registry) -> dict:
    return {'success': False, 'error': 'readonly'}
