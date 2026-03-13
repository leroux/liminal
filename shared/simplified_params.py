"""Macro controls for the FDN reverb.

Maps ~14 intuitive macro parameters to the full 30+ FdnParams.
Python implementation matching the Rust `simplified.rs` exactly.

Used by the LLM tuner system prompt and GUI.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict

SR = 44100.0
N = 8

# Size mapping constants
MIN_DELAY_MS = 2.0
MAX_DELAY_MS = 300.0
DELAY_RATIO = MAX_DELAY_MS / MIN_DELAY_MS  # 150.0

DELAY_RATIOS = [1.0, 1.249, 1.391, 1.613, 1.788, 1.997, 2.279, 2.461]
DIFFUSION_RATIOS = [1.0, 1.491, 2.208, 3.038]
DIFFUSION_SCALE = 0.1785  # 5.3 / 29.7

MAX_DAMPING = 0.95

DEFAULT_PANS = [-1.0, -0.714, -0.429, -0.143, 0.143, 0.429, 0.714, 1.0]

# Modulation mapping constants
MOD_DELAY_MAX = 50.0
MOD_DELAY_MIN = 5.0
MOD_DAMP_MAX = 0.5
MOD_OUTPUT_MAX = 1.0
MOD_MATRIX_MAX = 0.8

SPREAD_TEMPLATE = [0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 3.0]

VALID_MATRIX_TYPES = [
    "householder", "hadamard", "diagonal",
    "random_orthogonal", "circulant", "stautner_puckette",
]

# Ranges for LLM tuner validation
SIMPLIFIED_RANGES = {
    "size": (0.0, 1.0),
    "decay": (0.0, 1.15),
    "brightness": (0.0, 1.0),
    "diffusion": (0.0, 0.7),
    "mix": (0.0, 1.0),
    "saturation": (0.0, 1.0),
    "pre_delay_ms": (0.0, 250.0),
    "stereo_width": (0.0, 1.0),
    "mod_rate": (0.0, 20.0),
    "mod_depth": (0.0, 1.0),
    "mod_character": (0.0, 1.0),
    "mod_spread": (0.0, 1.0),
    "mod_waveform": (0, 2),
}

# Descriptions for LLM tuner system prompt
SIMPLIFIED_DESCRIPTIONS = {
    "size": "Room size (0=tiny resonator ~2ms, 0.5=medium room ~25ms, 1.0=huge space ~300ms)",
    "decay": "Decay time / feedback (0=very short, 0.85=default, >1.0=infinite/growing)",
    "brightness": "Tonal brightness (0=very dark/heavy damping, 1.0=bright/no damping)",
    "diffusion": "Diffusion amount (0=discrete echoes, 0.7=maximum smearing)",
    "mix": "Wet/dry mix (0=fully dry, 1.0=fully wet)",
    "saturation": "Saturation in feedback loop (0=clean, 1.0=heavy saturation)",
    "pre_delay_ms": "Pre-delay in milliseconds (0-250ms)",
    "stereo_width": "Stereo spread (0=mono, 1.0=full stereo)",
    "matrix_type": f"Feedback matrix type: {', '.join(VALID_MATRIX_TYPES)}",
    "mod_rate": "Modulation rate in Hz (0=no modulation, up to 20Hz)",
    "mod_depth": "Overall modulation intensity (0-1, distributed by mod_character)",
    "mod_character": "Modulation style (0=chorus/delay-only, 0.5=balanced, 1.0=glitchy/matrix-heavy)",
    "mod_spread": "Modulation decorrelation between nodes (0=sync, 1.0=fully spread)",
    "mod_waveform": "Modulation waveform (0=sine, 1=triangle, 2=sample-and-hold)",
}


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


@dataclass
class ReverbParams:
    """Simplified macro controls for the reverb."""

    size: float = 0.5
    decay: float = 0.85
    brightness: float = 0.7
    diffusion: float = 0.5
    mix: float = 0.5
    saturation: float = 0.0
    pre_delay_ms: float = 10.0
    stereo_width: float = 1.0
    matrix_type: str = "householder"
    mod_rate: float = 0.0
    mod_depth: float = 0.0
    mod_character: float = 0.0
    mod_spread: float = 0.0
    mod_waveform: int = 0

    def to_fdn_params(self) -> dict:
        """Convert to full FdnParams dict for DSP processing."""
        # Size → delay_times + diffusion_delays
        base_ms = MIN_DELAY_MS * (DELAY_RATIO ** _clamp(self.size, 0.0, 1.0))
        delay_times = [int(base_ms * r / 1000.0 * SR) for r in DELAY_RATIOS]

        diff_base_ms = base_ms * DIFFUSION_SCALE
        diffusion_delays = [max(1, int(diff_base_ms * r / 1000.0 * SR)) for r in DIFFUSION_RATIOS]

        # Brightness → damping
        damping = MAX_DAMPING * (1.0 - _clamp(self.brightness, 0.0, 1.0))

        # Stereo width → node pans
        w = _clamp(self.stereo_width, 0.0, 1.0)
        node_pans = [p * w for p in DEFAULT_PANS]

        # Matrix type validation
        mt = self.matrix_type if self.matrix_type in VALID_MATRIX_TYPES else "householder"

        params = {
            "delay_times": delay_times,
            "damping_coeffs": [damping] * N,
            "input_gains": [1.0 / N] * N,
            "output_gains": [1.0] * N,
            "node_pans": node_pans,
            "feedback_gain": _clamp(self.decay, 0.0, 1.15),
            "wet_dry": _clamp(self.mix, 0.0, 1.0),
            "diffusion": _clamp(self.diffusion, 0.0, 0.7),
            "diffusion_stages": 4,
            "diffusion_delays": diffusion_delays,
            "saturation": _clamp(self.saturation, 0.0, 1.0),
            "pre_delay": int(_clamp(self.pre_delay_ms, 0.0, 250.0) / 1000.0 * SR),
            "stereo_width": w,
            "matrix_type": mt,
            "matrix_seed": 42,
        }

        # Modulation
        params["mod_master_rate"] = max(0.0, self.mod_rate)
        params["mod_waveform"] = _clamp(self.mod_waveform, 0, 2)

        if self.mod_rate > 0.0 and self.mod_depth > 0.0:
            d = _clamp(self.mod_depth, 0.0, 1.0)
            c = _clamp(self.mod_character, 0.0, 1.0)

            delay_depth = d * _lerp(MOD_DELAY_MAX, MOD_DELAY_MIN, c)
            damp_depth = d * _lerp(0.0, MOD_DAMP_MAX, c)
            output_depth = d * _lerp(0.0, MOD_OUTPUT_MAX, c)
            matrix_depth = d * _lerp(0.0, MOD_MATRIX_MAX, c)

            s = _clamp(self.mod_spread, 0.0, 1.0)

            params["mod_depth_delay"] = [delay_depth] * N
            params["mod_depth_damping"] = [damp_depth] * N
            params["mod_depth_output"] = [output_depth] * N
            params["mod_depth_matrix"] = matrix_depth
            params["mod_correlation"] = 1.0 - s
            params["mod_node_rate_mult"] = [1.0 + s * t for t in SPREAD_TEMPLATE]
        else:
            params["mod_depth_delay"] = [0.0] * N
            params["mod_depth_damping"] = [0.0] * N
            params["mod_depth_output"] = [0.0] * N
            params["mod_depth_matrix"] = 0.0
            params["mod_correlation"] = 1.0
            params["mod_node_rate_mult"] = [1.0] * N

        return params

    @classmethod
    def from_fdn_params(cls, p: dict) -> "ReverbParams":
        """Best-fit reverse mapping from full FdnParams (lossy)."""
        defaults = _get_defaults()

        # Size: invert exponential from mean delay time
        dt = p.get("delay_times", defaults["delay_times"])
        mean_delay_ms = sum(s / SR * 1000.0 for s in dt) / max(len(dt), 1)
        mean_ratio = sum(DELAY_RATIOS) / N
        base_ms = max(mean_delay_ms / mean_ratio, MIN_DELAY_MS)
        size = math.log(base_ms / MIN_DELAY_MS) / math.log(DELAY_RATIO)

        # Decay
        decay = p.get("feedback_gain", defaults["feedback_gain"])

        # Brightness from mean damping
        dc = p.get("damping_coeffs", defaults["damping_coeffs"])
        mean_damping = sum(dc) / max(len(dc), 1)
        brightness = 1.0 - (mean_damping / MAX_DAMPING)

        # Direct scalars
        diffusion = p.get("diffusion", defaults["diffusion"])
        mix = p.get("wet_dry", defaults["wet_dry"])
        saturation = p.get("saturation", defaults["saturation"])
        pre_delay_ms = p.get("pre_delay", defaults["pre_delay"]) / SR * 1000.0
        stereo_width = p.get("stereo_width", defaults["stereo_width"])

        # Matrix type
        mt = p.get("matrix_type", defaults["matrix_type"])
        matrix_type = mt if mt in VALID_MATRIX_TYPES else "householder"

        # Modulation reverse mapping
        mod_rate = p.get("mod_master_rate", 0.0)
        mod_waveform = p.get("mod_waveform", 0)
        mod_depth = 0.0
        mod_character = 0.0
        mod_spread = 0.0

        if mod_rate > 0.0:
            mdd = p.get("mod_depth_delay", [0.0] * N)
            mdo = p.get("mod_depth_output", [0.0] * N)

            mean_delay_d = sum(mdd) / max(len(mdd), 1)
            mean_out_d = sum(mdo) / max(len(mdo), 1)
            delay_range = MOD_DELAY_MAX - MOD_DELAY_MIN

            # Solve forward equations:
            #   delay_depth = d * (MOD_DELAY_MAX - delay_range * c)
            #   output_depth = d * MOD_OUTPUT_MAX * c
            if mean_out_d > 1e-6:
                d = (mean_delay_d + delay_range * mean_out_d / MOD_OUTPUT_MAX) / MOD_DELAY_MAX
                c = _clamp(mean_out_d / (d * MOD_OUTPUT_MAX), 0.0, 1.0) if d > 1e-6 else 0.0
                mod_depth = _clamp(d, 0.0, 1.0)
                mod_character = c
            elif mean_delay_d > 1e-6:
                mod_depth = _clamp(mean_delay_d / MOD_DELAY_MAX, 0.0, 1.0)
                mod_character = 0.0
            # else: mod_depth and mod_character stay 0.0

            mod_spread = 1.0 - _clamp(p.get("mod_correlation", 1.0), 0.0, 1.0)

        return cls(
            size=_clamp(size, 0.0, 1.0),
            decay=_clamp(decay, 0.0, 1.15),
            brightness=_clamp(brightness, 0.0, 1.0),
            diffusion=diffusion,
            mix=mix,
            saturation=saturation,
            pre_delay_ms=pre_delay_ms,
            stereo_width=stereo_width,
            matrix_type=matrix_type,
            mod_rate=mod_rate,
            mod_depth=mod_depth,
            mod_character=mod_character,
            mod_spread=mod_spread,
            mod_waveform=mod_waveform,
        )

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ReverbParams":
        """Create from dict, using defaults for missing keys."""
        defaults = cls()
        return cls(**{k: d.get(k, getattr(defaults, k)) for k in defaults.__dataclass_fields__})

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "ReverbParams":
        return cls.from_dict(json.loads(s))


def _get_defaults() -> dict:
    """Default FdnParams values matching Rust."""
    delay_ms = [29.7, 37.1, 41.3, 47.9, 53.1, 59.3, 67.7, 73.1]
    diffusion_ms = [5.3, 7.9, 11.7, 16.1]
    return {
        "delay_times": [int(ms / 1000.0 * SR) for ms in delay_ms],
        "damping_coeffs": [0.3] * N,
        "input_gains": [1.0 / N] * N,
        "output_gains": [1.0] * N,
        "feedback_gain": 0.85,
        "wet_dry": 0.5,
        "diffusion": 0.5,
        "diffusion_stages": 4,
        "diffusion_delays": [int(ms / 1000.0 * SR) for ms in diffusion_ms],
        "saturation": 0.0,
        "pre_delay": int(10.0 / 1000.0 * SR),
        "stereo_width": 1.0,
        "matrix_type": "householder",
        "matrix_seed": 42,
        "mod_master_rate": 0.0,
    }
