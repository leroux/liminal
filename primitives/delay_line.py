"""Circular buffer delay line — the fundamental DSP building block."""

import numpy as np


class DelayLine:
    """Fixed-maximum-length circular buffer with integer and fractional delay reads.

    Usage:
        dl = DelayLine(max_delay=44100)
        dl.write(sample)
        out = dl.read(delay_samples)           # integer delay
        out = dl.read_linear(delay_fractional)  # linear interpolation
        out = dl.read_cubic(delay_fractional)   # cubic Lagrange interpolation
    """

    def __init__(self, max_delay: int):
        self.buffer = np.zeros(max_delay, dtype=np.float64)
        self.length = max_delay
        self.write_idx = 0

    def write(self, sample: float):
        """Write a sample and advance the write pointer."""
        self.buffer[self.write_idx] = sample
        self.write_idx = (self.write_idx + 1) % self.length

    def read(self, delay: int) -> float:
        """Read a sample from `delay` steps in the past.

        delay=0 returns the most recently written sample.
        delay=1 returns the sample before that, etc.
        """
        idx = (self.write_idx - 1 - delay) % self.length
        return self.buffer[idx]

    def read_linear(self, delay: float) -> float:
        """Read with linear interpolation between two adjacent samples.

        Blends the two nearest integer delay positions proportionally.
        Good enough for most uses; cheap (one multiply, one add).
        """
        int_delay = int(delay)
        frac = delay - int_delay
        s0 = self.read(int_delay)
        s1 = self.read(int_delay + 1)
        return s0 + frac * (s1 - s0)

    def read_cubic(self, delay: float) -> float:
        """Read with 4-point cubic Lagrange interpolation.

        Uses 4 neighboring samples for a smoother curve through the points.
        Less high-frequency roll-off than linear; matters for modulated delays.
        """
        int_delay = int(delay)
        frac = delay - int_delay
        # 4 samples: one before, two around, one after the fractional position
        sm1 = self.read(int_delay - 1)  # one sample ahead
        s0 = self.read(int_delay)
        s1 = self.read(int_delay + 1)
        s2 = self.read(int_delay + 2)
        # Lagrange 3rd-order coefficients
        c0 = s0
        c1 = 0.5 * (s1 - sm1)
        c2 = sm1 - 2.5 * s0 + 2.0 * s1 - 0.5 * s2
        c3 = 0.5 * (s2 - sm1) + 1.5 * (s0 - s1)
        return ((c3 * frac + c2) * frac + c1) * frac + c0

    def reset(self):
        """Clear the buffer."""
        self.buffer[:] = 0.0
        self.write_idx = 0
