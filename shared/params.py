"""Declarative parameter schema for all pedals.

A pedal's parameter contract is defined as a list of ParamDef objects.
ParamSchema wraps the list and derives the legacy dicts (PARAM_RANGES,
PARAM_SECTIONS, CHOICE_RANGES, default_params, bypass_params) so existing
code continues to work without modification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class ParamType(Enum):
    FLOAT = "float"
    INT = "int"
    CHOICE = "choice"
    BOOL = "bool"
    FLOAT_ARRAY = "float_array"
    INT_ARRAY = "int_array"


@dataclass
class ParamDef:
    key: str
    type: ParamType
    default: Any
    section: str
    label: str = ""
    bypass: Any = None          # if None, uses default
    range: tuple | None = None  # (min, max) for continuous params
    choices: list[str] | None = None  # display names for CHOICE type
    array_size: int = 0         # for ARRAY types
    hidden: bool = False        # not shown in default GUI
    randomize_skip: bool = False


class ParamSchema:
    """Derives all legacy param structures from a declarative param list."""

    def __init__(self, params: list[ParamDef]):
        self._params = params
        self._by_key: dict[str, ParamDef] = {p.key: p for p in params}

    def default_params(self) -> dict:
        return {p.key: _copy(p.default) for p in self._params}

    def bypass_params(self) -> dict:
        result = {}
        for p in self._params:
            if p.bypass is not None:
                result[p.key] = _copy(p.bypass)
            else:
                result[p.key] = _copy(p.default)
        return result

    def param_ranges(self) -> dict[str, tuple]:
        """PARAM_RANGES — continuous params only (float/int with range)."""
        result = {}
        for p in self._params:
            if p.range is not None and p.type not in (ParamType.CHOICE, ParamType.BOOL):
                result[p.key] = p.range
        return result

    def param_sections(self) -> dict[str, list[str]]:
        """PARAM_SECTIONS — section name -> list of param keys."""
        sections: dict[str, list[str]] = {}
        for p in self._params:
            sections.setdefault(p.section, []).append(p.key)
        return sections

    def choice_ranges(self) -> dict[str, int]:
        """CHOICE_RANGES — choice/bool param -> number of options."""
        result = {}
        for p in self._params:
            if p.type == ParamType.CHOICE:
                if p.choices:
                    result[p.key] = len(p.choices)
                elif p.range:
                    result[p.key] = p.range[1] - p.range[0] + 1
            elif p.type == ParamType.BOOL:
                result[p.key] = 2
        return result

    def randomize_skip(self) -> set[str]:
        return {p.key for p in self._params if p.randomize_skip}

    def validate_and_clamp(self, raw: dict) -> dict:
        """Validate and clamp a raw params dict (e.g. from LLM output).

        Unknown keys are dropped. Values are type-cast and clamped to range.
        """
        defaults = self.default_params()
        result = {}
        for key, value in raw.items():
            if key not in self._by_key:
                continue
            p = self._by_key[key]
            default_val = defaults[key]

            if isinstance(default_val, list):
                if not isinstance(value, list):
                    continue
                expected_len = len(default_val)
                if len(value) < expected_len:
                    value = value + default_val[len(value):]
                elif len(value) > expected_len:
                    value = value[:expected_len]
                clamped = []
                for v in value:
                    if isinstance(default_val[0], int):
                        try:
                            v = int(round(v))
                        except (TypeError, ValueError):
                            v = default_val[0]
                    else:
                        try:
                            v = float(v)
                        except (TypeError, ValueError):
                            v = default_val[0]
                    if p.range:
                        lo, hi = p.range
                        v = max(lo, min(hi, v))
                    clamped.append(v)
                result[key] = clamped

            elif isinstance(default_val, int) and not isinstance(default_val, bool):
                try:
                    v = int(round(value))
                except (TypeError, ValueError):
                    continue
                if p.range:
                    lo, hi = p.range
                    v = max(lo, min(hi, v))
                result[key] = v

            elif isinstance(default_val, float):
                try:
                    v = float(value)
                except (TypeError, ValueError):
                    continue
                if p.range:
                    lo, hi = p.range
                    v = max(lo, min(hi, v))
                result[key] = v

            elif isinstance(default_val, str):
                result[key] = str(value)

            else:
                result[key] = value

        return result

    def get(self, key: str) -> ParamDef | None:
        return self._by_key.get(key)

    def __iter__(self):
        return iter(self._params)

    def __len__(self):
        return len(self._params)


def _copy(val):
    """Shallow copy lists to prevent mutation of defaults."""
    if isinstance(val, list):
        return list(val)
    return val
