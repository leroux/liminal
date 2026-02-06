"""Time-domain degradation: bitcrusher + sample rate reducer.

Complementary to the spectral processing.  These create amplitude distortion
(staircase waveforms) and aliasing (metallic overtones) rather than the
spectral artifacts produced by STFT-based loss.

Crush    -- reduce amplitude quantization levels (16-bit down to ~4-bit).
             No dithering -- the correlated distortion IS the desired effect.
Decimate -- zero-order hold sample rate reduction with phase accumulator.
             No anti-alias filter -- the aliasing IS the desired effect.
"""

import numpy as np
from numba import njit


def crush_and_decimate(audio, params):
    """Apply bitcrusher and/or sample rate reducer.

    Args:
        audio: mono float64 array
        params: parameter dict

    Returns:
        processed mono float64 array, same length
    """
    g = float(params.get("global_amount", 1.0))
    crush = float(params.get("crush", 0.0)) * g
    decimate = float(params.get("decimate", 0.0)) * g

    if crush <= 0.0 and decimate <= 0.0:
        return audio.copy()

    return _crush_decimate(audio, crush, decimate)


@njit(cache=True)
def _crush_decimate(audio, crush, decimate):
    n = len(audio)
    out = np.empty(n, dtype=np.float64)

    # Bitcrusher: crush 0 -> off, 1 -> extreme (16 down to 4 bits)
    if crush > 0:
        bits = 16.0 - 12.0 * crush
        quant = 2.0 ** (bits - 1.0)
        for i in range(n):
            out[i] = np.floor(audio[i] * quant + 0.5) / quant
    else:
        for i in range(n):
            out[i] = audio[i]

    # Sample rate reducer (zero-order hold with fractional phase accumulator)
    if decimate > 0:
        rate_factor = 1.0 + 31.0 * decimate   # hold for 1..32 samples
        phase = 0.0
        held = 0.0
        for i in range(n):
            phase += 1.0
            if phase >= rate_factor:
                held = out[i]
                phase -= rate_factor
            out[i] = held

    return out
