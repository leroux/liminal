"""Parameter dict schema and defaults for the FDN reverb.

This is the shared contract between GUI, ML, and manual scripting.
All parameter sources produce a dict in this format.
"""

SR = 44100


def default_params() -> dict:
    """A reasonable 'medium room' starting point."""
    return {
        # 8 delay times in samples — prime-ish ms values for dense echo pattern
        "delay_times": [
            int(29.7 / 1000 * SR),   # 1310
            int(37.1 / 1000 * SR),   # 1637
            int(41.3 / 1000 * SR),   # 1821
            int(47.9 / 1000 * SR),   # 2113
            int(53.1 / 1000 * SR),   # 2342
            int(59.3 / 1000 * SR),   # 2615
            int(67.7 / 1000 * SR),   # 2986
            int(73.1 / 1000 * SR),   # 3224
        ],

        # Per-node damping (one-pole coefficient). 0=no filter, higher=darker.
        "damping_coeffs": [0.3] * 8,

        # Global feedback gain. 0=no reverb, 0.85=medium, >1.0=unstable
        "feedback_gain": 0.85,

        # How input distributes to each node (equal by default)
        "input_gains": [1.0 / 8] * 8,

        # How each node contributes to output (equal by default)
        "output_gains": [1.0] * 8,

        # Pre-delay in samples (0 to ~250ms)
        "pre_delay": int(10.0 / 1000 * SR),  # 10ms

        # Input diffusion — allpass gain (0=bypass, up to ~0.7)
        "diffusion": 0.5,

        # Number of input diffusion allpass stages
        "diffusion_stages": 4,

        # Diffusion allpass delay times in samples (prime-ish)
        "diffusion_delays": [
            int(5.3 / 1000 * SR),
            int(7.9 / 1000 * SR),
            int(11.7 / 1000 * SR),
            int(16.1 / 1000 * SR),
        ],

        # Wet/dry mix. 0=fully dry, 1=fully wet
        "wet_dry": 0.5,

        # Feedback matrix topology
        # Options: householder, hadamard, diagonal, random_orthogonal,
        #          circulant, stautner_puckette
        "matrix_type": "householder",
        "matrix_seed": 42,  # only used for random_orthogonal
    }
