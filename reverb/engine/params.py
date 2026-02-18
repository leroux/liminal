"""Parameter schema for the FDN reverb.

This is the shared contract between GUI, ML, and manual scripting.
All parameter sources produce a dict in this format.

Defined declaratively using ParamDef — legacy dicts (PARAM_RANGES,
PARAM_SECTIONS, CHOICE_RANGES) are derived automatically.
"""

from shared.params import ParamType as T, ParamDef, ParamSchema

SR = 44100

# ── Schema ────────────────────────────────────────────────────────────

_PARAMS = [
    # --- Per-node arrays (8 nodes) ---
    ParamDef("delay_times", T.INT_ARRAY, section="delay_times",
             default=[int(ms / 1000 * SR) for ms in
                      [29.7, 37.1, 41.3, 47.9, 53.1, 59.3, 67.7, 73.1]],
             bypass=[int(29.7 / 1000 * SR)] * 8,
             range=(1, int(300 / 1000 * SR)), array_size=8),

    ParamDef("damping_coeffs", T.FLOAT_ARRAY, section="damping",
             default=[0.3] * 8, bypass=[0.0] * 8,
             range=(0.0, 0.99), array_size=8),

    ParamDef("input_gains", T.FLOAT_ARRAY, section="input_gains",
             default=[1.0 / 8] * 8,
             range=(0.0, 0.5), array_size=8),

    ParamDef("output_gains", T.FLOAT_ARRAY, section="output_gains",
             default=[1.0] * 8,
             range=(0.0, 2.0), array_size=8),

    ParamDef("node_pans", T.FLOAT_ARRAY, section="node_pans",
             default=[-1.0, -0.714, -0.429, -0.143, 0.143, 0.429, 0.714, 1.0],
             range=(-1.0, 1.0), array_size=8),

    # --- Global ---
    ParamDef("feedback_gain", T.FLOAT, section="global",
             default=0.85, bypass=0.0,
             range=(0.0, 2.0)),

    ParamDef("wet_dry", T.FLOAT, section="global",
             default=0.5, bypass=0.0,
             range=(0.0, 1.0)),

    ParamDef("diffusion", T.FLOAT, section="global",
             default=0.5, bypass=0.0,
             range=(0.0, 0.7)),

    ParamDef("diffusion_stages", T.CHOICE, section="global",
             default=4, range=(1, 4),
             choices=["1", "2", "3", "4"]),

    ParamDef("diffusion_delays", T.INT_ARRAY, section="global",
             default=[int(ms / 1000 * SR) for ms in [5.3, 7.9, 11.7, 16.1]],
             array_size=4, hidden=True, randomize_skip=True),

    ParamDef("saturation", T.FLOAT, section="global",
             default=0.0,
             range=(0.0, 1.0)),

    ParamDef("pre_delay", T.INT, section="global",
             default=int(10.0 / 1000 * SR), bypass=0,
             range=(0, int(250 / 1000 * SR))),

    ParamDef("stereo_width", T.FLOAT, section="global",
             default=1.0,
             range=(0.0, 1.0)),

    # --- Matrix ---
    ParamDef("matrix_type", T.CHOICE, section="matrix",
             default="householder",
             choices=["householder", "hadamard", "diagonal",
                      "random_orthogonal", "circulant", "stautner_puckette"],
             randomize_skip=True),  # handled specially by GUI

    ParamDef("matrix_seed", T.INT, section="matrix",
             default=42, randomize_skip=True),

    # --- Modulation ---
    ParamDef("mod_master_rate", T.FLOAT, section="modulation",
             default=0.0,
             range=(0.0, 1000.0)),

    ParamDef("mod_node_rate_mult", T.FLOAT_ARRAY, section="modulation",
             default=[1.0] * 8, array_size=8,
             randomize_skip=True),

    ParamDef("mod_correlation", T.FLOAT, section="modulation",
             default=1.0,
             range=(0.0, 1.0)),

    ParamDef("mod_waveform", T.CHOICE, section="modulation",
             default=0, range=(0, 2),
             choices=["Sine", "Triangle", "Sample & Hold"]),

    ParamDef("mod_depth_delay", T.FLOAT_ARRAY, section="modulation",
             default=[0.0] * 8,
             range=(0.0, 100.0), array_size=8),

    ParamDef("mod_depth_damping", T.FLOAT_ARRAY, section="modulation",
             default=[0.0] * 8,
             range=(0.0, 0.5), array_size=8),

    ParamDef("mod_depth_output", T.FLOAT_ARRAY, section="modulation",
             default=[0.0] * 8,
             range=(0.0, 1.0), array_size=8),

    ParamDef("mod_depth_matrix", T.FLOAT, section="modulation",
             default=0.0,
             range=(0.0, 1.0)),

    ParamDef("mod_rate_scale_delay", T.FLOAT, section="modulation",
             default=1.0,
             range=(0.01, 10.0)),

    ParamDef("mod_rate_scale_damping", T.FLOAT, section="modulation",
             default=1.0,
             range=(0.01, 10.0)),

    ParamDef("mod_rate_scale_output", T.FLOAT, section="modulation",
             default=1.0,
             range=(0.01, 10.0)),

    ParamDef("mod_rate_matrix", T.FLOAT, section="modulation",
             default=0.0,
             range=(0.0, 1000.0)),

    ParamDef("mod_matrix2_type", T.CHOICE, section="modulation",
             default="random_orthogonal",
             choices=["householder", "hadamard", "diagonal",
                      "random_orthogonal", "circulant", "stautner_puckette"],
             randomize_skip=True),

    ParamDef("mod_matrix2_seed", T.INT, section="modulation",
             default=137, randomize_skip=True),
]

SCHEMA = ParamSchema(_PARAMS)

# ── Legacy API (unchanged) ────────────────────────────────────────────

default_params = SCHEMA.default_params
bypass_params = SCHEMA.bypass_params
PARAM_RANGES = SCHEMA.param_ranges()
PARAM_SECTIONS = SCHEMA.param_sections()
CHOICE_RANGES = SCHEMA.choice_ranges()
