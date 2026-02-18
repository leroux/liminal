"""Parameter schema for the Fractal audio fractalization effect.

All callers (GUI, CLI, presets) use the same params dict contract.
Defined declaratively using ParamDef — legacy dicts are derived automatically.
"""

from shared.params import ParamType as T, ParamDef, ParamSchema

SR = 44100

# Named constants for choice params
INTERP_NAMES = ["Nearest", "Linear"]
FILTER_NAMES = ["Bypass", "Lowpass", "Highpass", "Bandpass"]
POST_FILTER_NAMES = ["Bypass", "Lowpass", "Highpass"]

# Bounce target parameter names (subset of params that can be modulated)
BOUNCE_TARGETS = ["scale_ratio", "amplitude_decay", "num_scales", "saturation",
                  "filter_freq", "crush", "spectral"]
BOUNCE_TARGET_NAMES = ["Ratio", "Decay", "Scales", "Saturate",
                       "Filter", "Crush", "Spectral"]

# ── Schema ────────────────────────────────────────────────────────────

_PARAMS = [
    # --- Core fractal ---
    ParamDef("num_scales", T.INT, section="fractal",
             default=3, bypass=2,
             range=(2, 8)),

    ParamDef("scale_ratio", T.FLOAT, section="fractal",
             default=0.5,
             range=(0.1, 0.9)),

    ParamDef("amplitude_decay", T.FLOAT, section="fractal",
             default=0.707, bypass=0.0,
             range=(0.1, 1.0)),

    ParamDef("interp", T.CHOICE, section="fractal",
             default=0,
             choices=INTERP_NAMES),

    ParamDef("reverse_scales", T.BOOL, section="fractal",
             default=0),

    ParamDef("scale_offset", T.FLOAT, section="fractal",
             default=0.0,
             range=(0.0, 1.0)),

    # --- Iteration / feedback ---
    ParamDef("iterations", T.INT, section="iteration",
             default=1,
             range=(1, 4)),

    ParamDef("iter_decay", T.FLOAT, section="iteration",
             default=0.8,
             range=(0.3, 1.0)),

    ParamDef("saturation", T.FLOAT, section="iteration",
             default=0.0,
             range=(0.0, 1.0)),

    ParamDef("feedback", T.FLOAT, section="iteration",
             default=0.0,
             range=(0.0, 0.95)),

    # --- Spectral fractal ---
    ParamDef("spectral", T.FLOAT, section="spectral",
             default=0.0,
             range=(0.0, 1.0)),

    ParamDef("window_size", T.INT, section="spectral",
             default=2048,
             range=(256, 8192)),

    # --- Pre-filter ---
    ParamDef("filter_type", T.CHOICE, section="filter",
             default=0,
             choices=FILTER_NAMES),

    ParamDef("filter_freq", T.FLOAT, section="filter",
             default=2000.0,
             range=(20.0, 20000.0)),

    ParamDef("filter_q", T.FLOAT, section="filter",
             default=0.707,
             range=(0.1, 10.0)),

    ParamDef("post_filter_type", T.CHOICE, section="filter",
             default=0,
             choices=POST_FILTER_NAMES),

    ParamDef("post_filter_freq", T.FLOAT, section="filter",
             default=8000.0,
             range=(20.0, 20000.0)),

    # --- Effects ---
    ParamDef("gate", T.FLOAT, section="effects",
             default=0.0,
             range=(0.0, 1.0)),

    ParamDef("crush", T.FLOAT, section="effects",
             default=0.0,
             range=(0.0, 1.0)),

    ParamDef("decimate", T.FLOAT, section="effects",
             default=0.0,
             range=(0.0, 1.0)),

    # --- Layers ---
    ParamDef("layer_gain_1", T.FLOAT, section="layers",
             default=1.0,
             range=(0.0, 2.0)),

    ParamDef("layer_gain_2", T.FLOAT, section="layers",
             default=1.0,
             range=(0.0, 2.0)),

    ParamDef("layer_gain_3", T.FLOAT, section="layers",
             default=1.0,
             range=(0.0, 2.0)),

    ParamDef("layer_gain_4", T.FLOAT, section="layers",
             default=1.0,
             range=(0.0, 2.0)),

    ParamDef("layer_gain_5", T.FLOAT, section="layers",
             default=1.0,
             range=(0.0, 2.0)),

    ParamDef("layer_gain_6", T.FLOAT, section="layers",
             default=1.0,
             range=(0.0, 2.0)),

    ParamDef("layer_gain_7", T.FLOAT, section="layers",
             default=1.0,
             range=(0.0, 2.0)),

    ParamDef("fractal_only_wet", T.BOOL, section="layers",
             default=0),

    ParamDef("layer_spread", T.FLOAT, section="layers",
             default=0.0,
             range=(0.0, 1.0)),

    ParamDef("layer_detune", T.FLOAT, section="layers",
             default=0.0,
             range=(0.0, 1.0)),

    ParamDef("layer_delay", T.FLOAT, section="layers",
             default=0.0,
             range=(0.0, 1.0)),

    ParamDef("layer_tilt", T.FLOAT, section="layers",
             default=0.0,
             range=(-1.0, 1.0)),

    # --- Bounce ---
    ParamDef("bounce", T.BOOL, section="bounce",
             default=0),

    ParamDef("bounce_target", T.CHOICE, section="bounce",
             default=0,
             choices=BOUNCE_TARGET_NAMES),

    ParamDef("bounce_rate", T.FLOAT, section="bounce",
             default=0.3,
             range=(0.0, 1.0)),

    ParamDef("bounce_lfo_min", T.FLOAT, section="bounce",
             default=0.1,
             range=(0.01, 50.0)),

    ParamDef("bounce_lfo_max", T.FLOAT, section="bounce",
             default=5.0,
             range=(0.01, 50.0)),

    # --- Output ---
    ParamDef("wet_dry", T.FLOAT, section="output",
             default=1.0,
             range=(0.0, 1.0)),

    ParamDef("output_gain", T.FLOAT, section="output",
             default=0.5,
             range=(0.0, 1.0)),

    ParamDef("threshold", T.FLOAT, section="output",
             default=0.5, bypass=1.0,
             range=(0.0, 1.0)),

    # --- Internal ---
    ParamDef("seed", T.INT, section="output",
             default=42, hidden=True, randomize_skip=True),
]

SCHEMA = ParamSchema(_PARAMS)

# ── Legacy API (unchanged) ────────────────────────────────────────────

default_params = SCHEMA.default_params
bypass_params = SCHEMA.bypass_params
PARAM_RANGES = SCHEMA.param_ranges()
PARAM_SECTIONS = SCHEMA.param_sections()
CHOICE_RANGES = SCHEMA.choice_ranges()
