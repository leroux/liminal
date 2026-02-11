"""Parameter schema for the Fractal audio fractalization effect.

All callers (GUI, CLI, presets) use the same params dict contract.
"""

SR = 44100

# Interpolation mode names
INTERP_NAMES = ["Nearest", "Linear"]

# Filter type names
FILTER_NAMES = ["Bypass", "Lowpass", "Highpass", "Bandpass"]

# Post-filter type names
POST_FILTER_NAMES = ["Bypass", "Lowpass", "Highpass"]

# Bounce target parameter names (subset of params that can be modulated)
BOUNCE_TARGETS = ["scale_ratio", "amplitude_decay", "num_scales", "saturation",
                  "filter_freq", "crush", "spectral"]
BOUNCE_TARGET_NAMES = ["Ratio", "Decay", "Scales", "Saturate",
                       "Filter", "Crush", "Spectral"]


def default_params():
    return {
        # --- Core fractal ---
        "num_scales": 3,            # 2-8 fractal scale layers
        "scale_ratio": 0.5,         # 0.1-0.9 compression ratio per scale
        "amplitude_decay": 0.707,   # 0.1-1.0 gain decay per scale (sqrt(0.5) = pink noise)
        "interp": 0,                # 0=nearest (aliased), 1=linear (smooth)
        "reverse_scales": 0,        # 0=off, 1=reverse tiled chunks
        "scale_offset": 0.0,        # 0.0-1.0 phase offset for tile start

        # --- Iteration / feedback ---
        "iterations": 1,            # 1-4 re-feed through fractalizer
        "iter_decay": 0.8,          # 0.3-1.0 gain between iterations
        "saturation": 0.0,          # 0.0-1.0 tanh saturation in feedback

        # --- Spectral fractal ---
        "spectral": 0.0,            # 0.0-1.0 blend time vs spectral domain
        "window_size": 2048,        # 256-8192 STFT window for spectral mode

        # --- Pre-filter ---
        "filter_type": 0,           # 0=bypass, 1=lowpass, 2=highpass, 3=bandpass
        "filter_freq": 2000.0,      # 20.0-20000.0 Hz
        "filter_q": 0.707,          # 0.1-10.0 resonance

        # --- Post-filter ---
        "post_filter_type": 0,      # 0=bypass, 1=lowpass, 2=highpass
        "post_filter_freq": 8000.0, # 20.0-20000.0 Hz

        # --- Effects ---
        "gate": 0.0,                # 0.0-1.0 noise gate threshold
        "crush": 0.0,               # 0.0-1.0 bitcrusher
        "decimate": 0.0,            # 0.0-1.0 sample rate reduction

        # --- Bounce (parameter modulation) ---
        "bounce": 0,                # 0=off, 1=on
        "bounce_target": 0,         # index into BOUNCE_TARGETS
        "bounce_rate": 0.3,         # 0.0-1.0 LFO rate
        "bounce_lfo_min": 0.1,      # 0.01-50.0 Hz
        "bounce_lfo_max": 5.0,      # 0.01-50.0 Hz

        # --- Output ---
        "wet_dry": 1.0,             # 0.0-1.0 mix
        "output_gain": 0.5,         # 0.0-1.0 output level (-36 to +36 dB)
        "threshold": 0.5,           # 0.0-1.0 limiter threshold

        # --- Internal ---
        "seed": 42,
    }


def bypass_params():
    """All params at their no-effect / clean position."""
    p = default_params()
    p.update({
        "num_scales": 2,
        "scale_ratio": 0.5,
        "amplitude_decay": 0.0,
        "iterations": 1,
        "saturation": 0.0,
        "spectral": 0.0,
        "filter_type": 0,
        "post_filter_type": 0,
        "gate": 0.0,
        "crush": 0.0,
        "decimate": 0.0,
        "bounce": 0,
        "wet_dry": 1.0,
        "output_gain": 0.5,
        "threshold": 1.0,
    })
    return p


PARAM_RANGES = {
    "num_scales":       (2, 8),
    "scale_ratio":      (0.1, 0.9),
    "amplitude_decay":  (0.1, 1.0),
    "scale_offset":     (0.0, 1.0),
    "iterations":       (1, 4),
    "iter_decay":       (0.3, 1.0),
    "saturation":       (0.0, 1.0),
    "spectral":         (0.0, 1.0),
    "window_size":      (256, 8192),
    "filter_freq":      (20.0, 20000.0),
    "filter_q":         (0.1, 10.0),
    "post_filter_freq": (20.0, 20000.0),
    "gate":             (0.0, 1.0),
    "crush":            (0.0, 1.0),
    "decimate":         (0.0, 1.0),
    "bounce_rate":      (0.0, 1.0),
    "bounce_lfo_min":   (0.01, 50.0),
    "bounce_lfo_max":   (0.01, 50.0),
    "wet_dry":          (0.0, 1.0),
    "output_gain":      (0.0, 1.0),
    "threshold":        (0.0, 1.0),
}

# Section groupings for lock feature
PARAM_SECTIONS = {
    "fractal": ["num_scales", "scale_ratio", "amplitude_decay", "interp",
                "reverse_scales", "scale_offset"],
    "iteration": ["iterations", "iter_decay", "saturation"],
    "spectral": ["spectral", "window_size"],
    "filter": ["filter_type", "filter_freq", "filter_q",
               "post_filter_type", "post_filter_freq"],
    "effects": ["gate", "crush", "decimate"],
    "bounce": ["bounce", "bounce_target", "bounce_rate",
               "bounce_lfo_min", "bounce_lfo_max"],
    "output": ["wet_dry", "output_gain", "threshold"],
}

# Integer/choice parameter value counts (for randomization)
CHOICE_RANGES = {
    "interp": len(INTERP_NAMES),
    "reverse_scales": 2,
    "filter_type": len(FILTER_NAMES),
    "post_filter_type": len(POST_FILTER_NAMES),
    "bounce": 2,
    "bounce_target": len(BOUNCE_TARGETS),
}
