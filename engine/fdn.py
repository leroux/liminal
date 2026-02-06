"""8-node Feedback Delay Network — the core reverb engine.

Signal flow:
    Input → [Pre-delay] → [Input Diffusion] → FDN Loop → [Wet/Dry Mix] → Output

FDN Loop (per sample):
    1. Read from 8 delay lines
    2. Apply per-node one-pole damping
    3. Multiply by feedback matrix
    4. Scale by feedback gain + saturate (tanh)
    5. Add input (distributed via input gains)
    6. Write back to delay lines
    7. Sum weighted outputs (output gains)
"""

import numpy as np


N = 8  # number of delay lines


def render_fdn(input_audio: np.ndarray, params: dict) -> np.ndarray:
    """The single entry point. GUI sliders, ML optimizers, and batch
    rendering all call this same function.

    Args:
        input_audio: mono float64 array
        params: parameter dict (see engine/params.py)

    Returns:
        output audio, same length as input
    """
    from engine.numba_fdn import render_fdn_fast
    return render_fdn_fast(input_audio, params)
