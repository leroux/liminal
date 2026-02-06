"""Packet loss and repeat simulation.

Uses a Gilbert-Elliott two-state Markov model for bursty dropout patterns
that mimic real network audio degradation.

Packet Loss  -- replace dropped packets with silence.
Packet Repeat -- replace dropped packets with the last good packet (stutter).

Short Hann crossfades at packet boundaries prevent clicks.
"""

import numpy as np
from engine.params import SR


# Crossfade length at packet boundaries (samples).  ~3 ms at 44.1 kHz.
_XFADE_SAMPLES = int(0.003 * SR)


def packet_process(audio, params):
    """Apply packet loss or repeat simulation.

    Args:
        audio: mono float64 array
        params: parameter dict

    Returns:
        processed mono float64 array, same length
    """
    mode = int(params.get("packets", 0))
    if mode == 0:  # Clean
        return audio.copy()

    g = float(params.get("global_amount", 1.0))
    rate = float(params.get("packet_rate", 0.3)) * g
    packet_ms = float(params.get("packet_size", 30.0))
    seed = int(params.get("seed", 42))

    if rate <= 0.0:
        return audio.copy()

    packet_samples = max(1, int(packet_ms * SR / 1000.0))
    rng = np.random.RandomState(seed + 1000)

    output = audio.copy()

    # Gilbert-Elliott: Good <-> Bad
    p_g2b = rate * 0.3          # probability good -> bad
    p_b2g = 0.4                 # recovery probability (avg burst ~ 2.5 packets)
    in_bad = False
    prev_bad = False

    last_good = np.zeros(packet_samples, dtype=np.float64)

    # Pre-compute crossfade windows
    xfade = min(_XFADE_SAMPLES, packet_samples // 4)
    fade_in = np.hanning(xfade * 2)[:xfade] if xfade > 0 else np.array([])
    fade_out = np.hanning(xfade * 2)[xfade:] if xfade > 0 else np.array([])

    for start in range(0, len(output), packet_samples):
        end = min(start + packet_samples, len(output))
        chunk_len = end - start

        if in_bad:
            if mode == 1:       # packet loss -> silence
                output[start:end] = 0.0
            elif mode == 2:     # packet repeat -> stutter
                output[start:end] = last_good[:chunk_len]

            # Crossfade at the boundary entering bad state
            if not prev_bad and xfade > 0 and start > 0:
                xf = min(xfade, start, chunk_len)
                output[start:start + xf] *= fade_in[:xf]
                # Blend with tail of previous packet
                output[start:start + xf] += audio[start:start + xf] * fade_out[-xf:]

            if rng.random() < p_b2g:
                prev_bad = True
                in_bad = False
            else:
                prev_bad = True
        else:
            # Crossfade at the boundary leaving bad state
            if prev_bad and xfade > 0:
                xf = min(xfade, chunk_len)
                output[start:start + xf] *= fade_in[:xf]

            last_good[:chunk_len] = audio[start:end]
            prev_bad = False
            if rng.random() < p_g2b:
                in_bad = True

    return output
