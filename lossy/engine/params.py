"""Parameter schema for Lossy codec emulation plugin.

All callers (GUI, CLI, presets) use the same params dict contract.
Defined declaratively using ParamDef — legacy dicts are derived automatically.
"""

from shared.params import ParamType as T, ParamDef, ParamSchema

SR = 44100

# Legacy speed→window_size mapping (for preset backward compat)
_LEGACY_WINDOW_SIZES = [4096, 2048, 1024, 512, 256]

# Named constants for choice params
QUANTIZER_NAMES = ["Uniform", "Compand"]
PACKET_NAMES = ["Clean", "Packet Loss", "Packet Repeat"]
FILTER_NAMES = ["Bypass", "Bandpass", "Notch"]
SLOPE_OPTIONS = [6, 24, 96]
SLOPE_SECTIONS = {6: 1, 24: 2, 96: 8}
FREEZE_NAMES = ["Slushy", "Solid"]
VERB_POSITION_NAMES = ["Pre", "Post"]

# Bounce target parameter names (subset of params that can be modulated)
BOUNCE_TARGETS = ["loss", "window_size", "crush", "decimate", "verb", "filter_freq", "gate"]
BOUNCE_TARGET_NAMES = ["Loss", "Window", "Crush", "Decimate", "Verb", "Filter Freq", "Gate"]

# ── Schema ────────────────────────────────────────────────────────────

_PARAMS = [
    # --- Spectral loss ---
    ParamDef("inverse", T.BOOL, section="spectral",
             default=0),

    ParamDef("jitter", T.FLOAT, section="spectral",
             default=0.0, bypass=0.0,
             range=(0.0, 1.0)),

    ParamDef("loss", T.FLOAT, section="spectral",
             default=0.5, bypass=0.0,
             range=(0.0, 1.0)),

    ParamDef("window_size", T.INT, section="spectral",
             default=2048,
             range=(64, 16384)),

    ParamDef("hop_divisor", T.INT, section="spectral",
             default=4,
             range=(1, 8)),

    ParamDef("n_bands", T.INT, section="spectral",
             default=21,
             range=(2, 64)),

    ParamDef("global_amount", T.FLOAT, section="spectral",
             default=1.0,
             range=(0.0, 1.0)),

    ParamDef("phase_loss", T.FLOAT, section="spectral",
             default=0.0,
             range=(0.0, 1.0)),

    ParamDef("quantizer", T.CHOICE, section="spectral",
             default=0,
             choices=QUANTIZER_NAMES),

    ParamDef("pre_echo", T.FLOAT, section="spectral",
             default=0.0,
             range=(0.0, 1.0)),

    ParamDef("noise_shape", T.FLOAT, section="spectral",
             default=0.0,
             range=(0.0, 1.0)),

    ParamDef("weighting", T.FLOAT, section="hidden",
             default=1.0,
             range=(0.0, 1.0)),

    ParamDef("hf_threshold", T.FLOAT, section="spectral",
             default=0.3,
             range=(0.0, 1.0)),

    ParamDef("transient_ratio", T.FLOAT, section="spectral",
             default=4.0,
             range=(1.5, 20.0)),

    ParamDef("slushy_rate", T.FLOAT, section="spectral",
             default=0.03,
             range=(0.001, 0.5)),

    # --- Crush ---
    ParamDef("crush", T.FLOAT, section="crush",
             default=0.0,
             range=(0.0, 1.0)),

    ParamDef("decimate", T.FLOAT, section="crush",
             default=0.0,
             range=(0.0, 1.0)),

    # --- Packets ---
    ParamDef("packets", T.CHOICE, section="packets",
             default=0,
             choices=PACKET_NAMES),

    ParamDef("packet_rate", T.FLOAT, section="packets",
             default=0.3, bypass=0.0,
             range=(0.0, 1.0)),

    ParamDef("packet_size", T.FLOAT, section="packets",
             default=30.0,
             range=(5.0, 200.0)),

    # --- Filter ---
    ParamDef("filter_type", T.CHOICE, section="filter",
             default=0,
             choices=FILTER_NAMES),

    ParamDef("filter_freq", T.FLOAT, section="filter",
             default=1000.0,
             range=(20.0, 20000.0)),

    ParamDef("filter_width", T.FLOAT, section="filter",
             default=0.5,
             range=(0.0, 1.0)),

    ParamDef("filter_slope", T.CHOICE, section="filter",
             default=1,
             choices=[str(s) for s in SLOPE_OPTIONS]),

    # --- Reverb ---
    ParamDef("verb", T.FLOAT, section="effects",
             default=0.0,
             range=(0.0, 1.0)),

    ParamDef("decay", T.FLOAT, section="effects",
             default=0.5,
             range=(0.0, 1.0)),

    ParamDef("verb_position", T.CHOICE, section="effects",
             default=0,
             choices=VERB_POSITION_NAMES),

    # --- Freeze ---
    ParamDef("freeze", T.BOOL, section="effects",
             default=0),

    ParamDef("freeze_mode", T.CHOICE, section="effects",
             default=0,
             choices=FREEZE_NAMES),

    ParamDef("freezer", T.FLOAT, section="effects",
             default=1.0, bypass=0.0,
             range=(0.0, 1.0)),

    # --- Gate ---
    ParamDef("gate", T.FLOAT, section="effects",
             default=0.0,
             range=(0.0, 1.0)),

    # --- Hidden ---
    ParamDef("threshold", T.FLOAT, section="hidden",
             default=0.5, bypass=1.0,
             range=(0.0, 1.0)),

    ParamDef("auto_gain", T.FLOAT, section="hidden",
             default=0.0,
             range=(0.0, 1.0)),

    ParamDef("loss_gain", T.FLOAT, section="hidden",
             default=0.5,
             range=(0.0, 1.0)),

    # --- Bounce ---
    ParamDef("bounce", T.BOOL, section="bounce",
             default=0),

    ParamDef("bounce_target", T.CHOICE, section="bounce",
             default=0,
             choices=BOUNCE_TARGET_NAMES),

    ParamDef("bounce_rate", T.FLOAT, section="bounce",
             default=0.3, bypass=0.0,
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


def migrate_legacy_params(params):
    """Convert legacy preset keys to current schema."""
    # Legacy speed → window_size
    if "speed" in params and "window_size" not in params:
        speed = float(params.pop("speed"))
        sizes = _LEGACY_WINDOW_SIZES
        idx = min(int(speed * len(sizes)), len(sizes) - 1)
        params["window_size"] = sizes[idx]
    elif "speed" in params:
        params.pop("speed")
    # Legacy mode → inverse + jitter
    if "mode" in params:
        mode = int(params.pop("mode"))
        if mode == 1:
            params.setdefault("inverse", 1)
        elif mode == 2:
            # Old jitter mode used loss to control jitter amount
            params.setdefault("jitter", params.get("loss", 0.5))
            params["loss"] = 0.0
    return params
