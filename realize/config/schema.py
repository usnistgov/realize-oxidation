"""Schema validation for input.yaml.

Validates a parsed input dict against the PPOConfig dataclasses (config/default_config.py)
before any value is applied, so a typo or unknown key fails loudly rather than being
silently dropped.

Rules (fail-fast):
  - Every top-level section must be a known PPOConfig section (or the reserved 'meta'
    section consumed by the loader).
  - Every key within a section must be a field of that section's dataclass.
  - Each value must be type-compatible with the field annotation (int accepted where
    float is expected; bool is NOT treated as int).
Range / cross-field checks are left to PPOConfig.validate().
"""
from dataclasses import fields, is_dataclass
from typing import Any, Dict, Union, get_args, get_origin
import typing

from realize.config.default_config import PPOConfig

# Reserved section consumed directly by the loader (device string, seed controls).
META_SECTION = "meta"
# seed_base: per-run base offset (a SLURM id is folded on top -> different every run).
# seed:      optional explicit pin -> used verbatim, the same seed across runs (repro).
META_KEYS = {"device", "seed_base", "seed"}


def _config_sections() -> Dict[str, type]:
    """Map of section name -> dataclass type for every PPOConfig sub-config."""
    sections = {}
    for f in fields(PPOConfig):
        if is_dataclass(f.type):
            sections[f.name] = f.type
    return sections


def _type_ok(value: Any, annotation: Any) -> bool:
    """Best-effort structural type check against a dataclass field annotation."""
    origin = get_origin(annotation)

    # typing.Optional[X] / Union[...]
    if origin is Union:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if value is None:
            return type(None) in get_args(annotation)
        return any(_type_ok(value, a) for a in args)

    # typing.Literal['a', 'b', ...]
    if origin is typing.Literal:
        return value in get_args(annotation)

    # typing.List[...] / list
    if origin in (list, typing.List):
        if not isinstance(value, list):
            return False
        elem_args = get_args(annotation)
        if not elem_args:
            return True
        return all(_type_ok(v, elem_args[0]) for v in value)

    # Plain scalar annotations.
    if annotation is float:
        # Accept int where float is expected, but never bool.
        return isinstance(value, float) or (isinstance(value, int) and not isinstance(value, bool))
    if annotation is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if annotation is bool:
        return isinstance(value, bool)
    if annotation is str:
        return isinstance(value, str)

    # Unknown / un-annotated: do not block (PPOConfig.validate handles semantics).
    return True


def validate_input(raw: Dict[str, Any]) -> None:
    """Validate a parsed input.yaml dict. Raises ValueError on the first problem."""
    if not isinstance(raw, dict):
        raise ValueError(f"input.yaml must parse to a mapping, got {type(raw).__name__}")

    sections = _config_sections()
    known_top = set(sections) | {META_SECTION}

    for top_key, block in raw.items():
        if top_key not in known_top:
            raise ValueError(
                f"Unknown top-level section '{top_key}' in input.yaml. "
                f"Allowed sections: {sorted(known_top)}"
            )
        if block is None:
            continue
        if not isinstance(block, dict):
            raise ValueError(
                f"Section '{top_key}' must be a mapping of key: value, "
                f"got {type(block).__name__}"
            )

        if top_key == META_SECTION:
            for k in block:
                if k not in META_KEYS:
                    raise ValueError(
                        f"Unknown key 'meta.{k}'. Allowed meta keys: {sorted(META_KEYS)}"
                    )
            continue

        dc = sections[top_key]
        field_types = {f.name: f.type for f in fields(dc)}
        for k, v in block.items():
            if k not in field_types:
                raise ValueError(
                    f"Unknown key '{top_key}.{k}'. Valid keys for '{top_key}': "
                    f"{sorted(field_types)}"
                )
            if not _type_ok(v, field_types[k]):
                raise ValueError(
                    f"Type mismatch for '{top_key}.{k}': expected "
                    f"{field_types[k]}, got value {v!r} ({type(v).__name__})"
                )
