"""Parameter schema for Lossy codec emulation plugin.

All callers (GUI, CLI, presets) use the same params dict contract.
"""

SR = 44100

# Legacy speed→window_size mapping (for preset backward compat)
_LEGACY_WINDOW_SIZES = [4096, 2048, 1024, 512, 256]

# Mode names
MODE_NAMES = ["Standard", "Inverse", "Jitter"]

# Quantizer type names
QUANTIZER_NAMES = ["Uniform", "Compand"]

# Packet mode names
PACKET_NAMES = ["Clean", "Packet Loss", "Packet Repeat"]

# Filter type names
FILTER_NAMES = ["Bypass", "Bandpass", "Notch"]

# Filter slope options (dB/oct) and corresponding biquad cascade counts
SLOPE_OPTIONS = [6, 24, 96]
SLOPE_SECTIONS = {6: 1, 24: 2, 96: 8}

# Freeze mode names
FREEZE_NAMES = ["Slushy", "Solid"]

# Verb routing options
VERB_POSITION_NAMES = ["Pre", "Post"]

# Bounce target parameter names (subset of params that can be modulated)
BOUNCE_TARGETS = ["loss", "window_size", "crush", "decimate", "verb", "filter_freq", "gate"]
BOUNCE_TARGET_NAMES = ["Loss", "Window", "Crush", "Decimate", "Verb", "Filter Freq", "Gate"]


def default_params():
    return {
        # --- Spectral loss ---
        "mode": 0,              # 0=standard, 1=inverse, 2=jitter
        "loss": 0.5,            # 0.0 (clean) to 1.0 (destroyed)
        "window_size": 2048,    # FFT window size in samples (64-16384)
        "hop_divisor": 4,       # hop = window_size / hop_divisor (1-8, 4=75% overlap)
        "n_bands": 21,          # number of Bark-like bands for gating (2-64)
        "global_amount": 1.0,   # master intensity multiplier
        "phase_loss": 0.0,      # phase quantization 0=off to 1=extreme
        "quantizer": 0,         # 0=uniform, 1=compand (power-law codec-style)
        "pre_echo": 0.0,        # pre-echo enhancement 0=off to 1=max
        "noise_shape": 0.0,     # envelope-following quantization noise shaping
        "weighting": 1.0,       # 0=equal freq weighting, 1=psychoacoustic ATH model
        "hf_threshold": 0.3,    # loss level where HF rolloff begins (0.0-1.0)
        "transient_ratio": 4.0, # energy ratio threshold for pre-echo detection (1.5-20.0)
        "slushy_rate": 0.03,    # freeze slushy drift speed (0.001-0.5)

        # --- Crush (time-domain degradation) ---
        "crush": 0.0,           # bitcrusher 0=off to 1=extreme
        "decimate": 0.0,        # sample rate reduction 0=off to 1=extreme

        # --- Packet processing ---
        "packets": 0,           # 0=clean, 1=loss, 2=repeat
        "packet_rate": 0.3,     # probability of entering bad state
        "packet_size": 30.0,    # packet chunk size in ms

        # --- Filter ---
        "filter_type": 0,       # 0=bypass, 1=bandpass, 2=notch
        "filter_freq": 1000.0,  # center frequency Hz
        "filter_width": 0.5,    # 0=narrow/high-Q to 1=wide/low-Q
        "filter_slope": 1,      # index into SLOPE_OPTIONS: 0=6dB, 1=24dB, 2=96dB

        # --- Reverb ---
        "verb": 0.0,            # lo-fi reverb mix 0.0-1.0
        "decay": 0.5,           # reverb decay 0=short to 1=long
        "verb_position": 0,     # 0=pre (before loss), 1=post (after filter)

        # --- Freeze ---
        "freeze": 0,            # 0=off, 1=on
        "freeze_mode": 0,       # 0=slushy, 1=solid
        "freezer": 1.0,         # frozen/live blend (0=live, 1=frozen)

        # --- Gate ---
        "gate": 0.0,            # noise gate threshold 0=off

        # --- Hidden options ---
        "threshold": 0.5,       # limiter threshold 0=heavy limiting to 1=light
        "auto_gain": 0.0,       # automatic gain compensation 0=off to 1=full
        "loss_gain": 0.5,       # wet signal gain 0=-36dB, 0.5=0dB, 1=+36dB

        # --- Bounce (parameter modulation) ---
        "bounce": 0,            # 0=off, 1=on
        "bounce_target": 0,     # index into BOUNCE_TARGETS
        "bounce_rate": 0.3,     # LFO rate 0=slow to 1=fast
        "bounce_lfo_min": 0.1,  # LFO minimum frequency Hz
        "bounce_lfo_max": 5.0,  # LFO maximum frequency Hz

        # --- Output ---
        "wet_dry": 1.0,         # 0=dry, 1=wet

        # --- Internal ---
        "seed": 42,             # random seed for reproducibility
    }


def bypass_params():
    """All params at their no-effect / clean position."""
    return {
        "mode": 0, "loss": 0.0, "window_size": 2048, "hop_divisor": 4,
        "n_bands": 21, "global_amount": 1.0,
        "phase_loss": 0.0, "quantizer": 0, "pre_echo": 0.0,
        "noise_shape": 0.0, "weighting": 1.0,
        "hf_threshold": 0.3, "transient_ratio": 4.0, "slushy_rate": 0.03,
        "crush": 0.0, "decimate": 0.0,
        "packets": 0, "packet_rate": 0.0, "packet_size": 30.0,
        "filter_type": 0, "filter_freq": 1000.0, "filter_width": 0.5,
        "filter_slope": 1,
        "verb": 0.0, "decay": 0.5, "verb_position": 0,
        "freeze": 0, "freeze_mode": 0, "freezer": 0.0,
        "gate": 0.0,
        "threshold": 1.0, "auto_gain": 0.0, "loss_gain": 0.5,
        "bounce": 0, "bounce_target": 0, "bounce_rate": 0.0,
        "bounce_lfo_min": 0.1, "bounce_lfo_max": 5.0,
        "wet_dry": 1.0,
        "seed": 42,
    }


def migrate_legacy_params(params):
    """Convert legacy preset with 'speed' to 'window_size'."""
    if "speed" in params and "window_size" not in params:
        speed = float(params.pop("speed"))
        sizes = _LEGACY_WINDOW_SIZES
        idx = min(int(speed * len(sizes)), len(sizes) - 1)
        params["window_size"] = sizes[idx]
    elif "speed" in params:
        params.pop("speed")
    return params


PARAM_RANGES = {
    "loss":            (0.0, 1.0),
    "window_size":     (64, 16384),
    "hop_divisor":     (1, 8),
    "n_bands":         (2, 64),
    "global_amount":   (0.0, 1.0),
    "phase_loss":      (0.0, 1.0),
    "pre_echo":        (0.0, 1.0),
    "noise_shape":     (0.0, 1.0),
    "weighting":       (0.0, 1.0),
    "hf_threshold":    (0.0, 1.0),
    "transient_ratio": (1.5, 20.0),
    "slushy_rate":     (0.001, 0.5),
    "crush":           (0.0, 1.0),
    "decimate":        (0.0, 1.0),
    "packet_rate":     (0.0, 1.0),
    "packet_size":     (5.0, 200.0),
    "filter_freq":     (20.0, 20000.0),
    "filter_width":    (0.0, 1.0),
    "verb":            (0.0, 1.0),
    "decay":           (0.0, 1.0),
    "freezer":         (0.0, 1.0),
    "gate":            (0.0, 1.0),
    "threshold":       (0.0, 1.0),
    "auto_gain":       (0.0, 1.0),
    "loss_gain":       (0.0, 1.0),
    "bounce_rate":     (0.0, 1.0),
    "bounce_lfo_min":  (0.01, 50.0),
    "bounce_lfo_max":  (0.01, 50.0),
    "wet_dry":         (0.0, 1.0),
}

# Section groupings for lock feature
PARAM_SECTIONS = {
    "spectral": ["mode", "loss", "window_size", "hop_divisor", "n_bands",
                  "global_amount", "phase_loss", "pre_echo", "noise_shape",
                  "quantizer", "hf_threshold", "transient_ratio", "slushy_rate"],
    "crush": ["crush", "decimate"],
    "packets": ["packets", "packet_rate", "packet_size"],
    "filter": ["filter_type", "filter_freq", "filter_width", "filter_slope"],
    "effects": ["verb", "decay", "verb_position", "gate",
                "freeze", "freeze_mode", "freezer"],
    "hidden": ["threshold", "auto_gain", "loss_gain", "weighting"],
    "bounce": ["bounce", "bounce_target", "bounce_rate",
               "bounce_lfo_min", "bounce_lfo_max"],
    "output": ["wet_dry"],
}

# Integer/choice parameter value counts (for randomization)
CHOICE_RANGES = {
    "mode": len(MODE_NAMES),
    "quantizer": len(QUANTIZER_NAMES),
    "packets": len(PACKET_NAMES),
    "filter_type": len(FILTER_NAMES),
    "filter_slope": len(SLOPE_OPTIONS),
    "freeze": 2,
    "freeze_mode": len(FREEZE_NAMES),
    "verb_position": len(VERB_POSITION_NAMES),
    "bounce": 2,
    "bounce_target": len(BOUNCE_TARGETS),
}
