"""PedalConfig — everything a pedal needs to plug into the shared GUI base."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class PedalConfig:
    name: str                          # "Reverb", "Lossy", "Fractal"
    preset_dir: str
    preset_categories: list[str]
    window_title: str
    window_geometry: str
    default_params: Callable[[], dict]
    bypass_params: Callable[[], dict]
    param_ranges: dict
    param_sections: dict
    choice_ranges: dict
    render: Callable                   # (audio, params) -> audio
    render_stereo: Callable | None     # (left, right, params) -> (left, right)
    guide_text: str
    param_descriptions: dict
    sample_rate: int = 44100
    default_source: str = ""
    icon_path: str = ""
    extra_tabs: list[tuple[str, Callable]] = field(default_factory=list)
    randomize_skip: set = field(default_factory=set)
    randomize_clamp: dict = field(default_factory=dict)
    migrate_preset: Callable | None = None
    tail_param: str | None = "tail_length"  # param key for tail seconds, None if N/A
    schema: Any = None                      # ParamSchema instance (for LLM validation)
