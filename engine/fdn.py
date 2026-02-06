"""8-node Feedback Delay Network — the core reverb engine.

Signal flow:
    Input → [Pre-delay] → [Input Diffusion] → FDN Loop → [Wet/Dry Mix] → Output

FDN Loop (per sample):
    1. Read from 8 delay lines
    2. Apply per-node one-pole damping
    3. Multiply by Householder feedback matrix
    4. Scale by feedback gain
    5. Add input (distributed via input gains)
    6. Write back to delay lines
    7. Sum weighted outputs (output gains)
"""

import numpy as np

from primitives.delay_line import DelayLine
from primitives.filters import OnePoleFilter, AllpassFilter
from primitives.matrix import build_matrix_apply


N = 8  # number of delay lines


class FDN:
    """Stateful 8-node FDN reverb processor."""

    def __init__(self, params: dict):
        self.params = params
        max_delay = max(params["delay_times"]) + 1

        # Core FDN components
        self.delays = [DelayLine(max_delay=max_delay) for _ in range(N)]
        self.dampers = [OnePoleFilter(coeff=params["damping_coeffs"][i]) for i in range(N)]

        # Pre-delay
        pre_delay_samples = max(1, int(params["pre_delay"]))
        self.pre_delay = DelayLine(max_delay=pre_delay_samples + 1)
        self.pre_delay_samples = pre_delay_samples

        # Input diffusion (allpass chain)
        n_stages = min(params["diffusion_stages"], len(params["diffusion_delays"]))
        gain = params["diffusion"]
        self.diffusers = [
            AllpassFilter(params["diffusion_delays"][i], gain=gain)
            for i in range(n_stages)
        ]

        # Feedback matrix
        matrix_type = params.get("matrix_type", "householder")
        matrix_seed = params.get("matrix_seed", 42)
        self.apply_matrix = build_matrix_apply(matrix_type, N, seed=matrix_seed)

    def process_sample(self, x: float) -> float:
        p = self.params

        # Pre-delay
        self.pre_delay.write(x)
        x_delayed = self.pre_delay.read(self.pre_delay_samples)

        # Input diffusion
        diffused = x_delayed
        for ap in self.diffusers:
            diffused = ap.process(diffused)

        # Read from all delay lines
        reads = np.empty(N)
        for i in range(N):
            reads[i] = self.delays[i].read(p["delay_times"][i])

        # Output: weighted sum of delay reads
        output_gains = p["output_gains"]
        wet = 0.0
        for i in range(N):
            wet += reads[i] * output_gains[i]

        # Damping
        for i in range(N):
            reads[i] = self.dampers[i].process(reads[i])

        # Feedback matrix
        mixed = self.apply_matrix(reads)

        # Scale by feedback gain, add input, write back
        feedback_gain = p["feedback_gain"]
        input_gains = p["input_gains"]
        for i in range(N):
            self.delays[i].write(feedback_gain * mixed[i] + input_gains[i] * diffused)

        # Wet/dry mix
        mix = p["wet_dry"]
        return (1.0 - mix) * x + mix * wet

    def reset(self):
        for dl in self.delays:
            dl.reset()
        for d in self.dampers:
            d.reset()
        for ap in self.diffusers:
            ap.reset()
        self.pre_delay.reset()


def render_fdn(input_audio: np.ndarray, params: dict) -> np.ndarray:
    """The single entry point. GUI sliders, ML optimizers, and batch
    rendering all call this same function.

    Args:
        input_audio: mono float64 array
        params: parameter dict (see engine/params.py)

    Returns:
        output audio, same length as input
    """
    fdn = FDN(params)
    output = np.empty_like(input_audio)
    for i in range(len(input_audio)):
        output[i] = fdn.process_sample(input_audio[i])
    return output
