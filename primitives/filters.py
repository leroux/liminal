"""Filters — one-pole, biquad, allpass. Built from scratch."""

import numpy as np


class OnePoleFilter:
    """One-pole lowpass filter (damping).

    y[n] = (1 - a) * x[n] + a * y[n-1]

    a=0: no filtering (output = input)
    a close to 1: heavy lowpass (only very low frequencies pass)
    """

    def __init__(self, coeff: float = 0.5):
        self.coeff = coeff
        self.y1 = 0.0  # previous output

    def process(self, x: float) -> float:
        self.y1 = (1.0 - self.coeff) * x + self.coeff * self.y1
        return self.y1

    def reset(self):
        self.y1 = 0.0


class BiquadFilter:
    """Second-order (biquad) filter — 5 coefficients, 2 state variables.

    Difference equation (Direct Form 1):
        y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]

    Use the static methods to compute coefficients for each filter type.
    """

    def __init__(self, b0=1.0, b1=0.0, b2=0.0, a1=0.0, a2=0.0):
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        self.a1 = a1
        self.a2 = a2
        self.x1 = 0.0  # x[n-1]
        self.x2 = 0.0  # x[n-2]
        self.y1 = 0.0  # y[n-1]
        self.y2 = 0.0  # y[n-2]

    def process(self, x: float) -> float:
        y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2 \
            - self.a1 * self.y1 - self.a2 * self.y2
        self.x2 = self.x1
        self.x1 = x
        self.y2 = self.y1
        self.y1 = y
        return y

    def set_coeffs(self, b0, b1, b2, a1, a2):
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        self.a1 = a1
        self.a2 = a2

    def reset(self):
        self.x1 = self.x2 = self.y1 = self.y2 = 0.0

    @staticmethod
    def lowpass(freq, q, sr):
        """Lowpass coefficients. freq in Hz, q is resonance (0.707 = Butterworth)."""
        w0 = 2.0 * np.pi * freq / sr
        alpha = np.sin(w0) / (2.0 * q)
        cos_w0 = np.cos(w0)
        a0 = 1.0 + alpha
        b0 = (1.0 - cos_w0) / 2.0 / a0
        b1 = (1.0 - cos_w0) / a0
        b2 = b0
        a1 = (-2.0 * cos_w0) / a0
        a2 = (1.0 - alpha) / a0
        return BiquadFilter(b0, b1, b2, a1, a2)

    @staticmethod
    def highpass(freq, q, sr):
        """Highpass coefficients."""
        w0 = 2.0 * np.pi * freq / sr
        alpha = np.sin(w0) / (2.0 * q)
        cos_w0 = np.cos(w0)
        a0 = 1.0 + alpha
        b0 = (1.0 + cos_w0) / 2.0 / a0
        b1 = -(1.0 + cos_w0) / a0
        b2 = b0
        a1 = (-2.0 * cos_w0) / a0
        a2 = (1.0 - alpha) / a0
        return BiquadFilter(b0, b1, b2, a1, a2)

    @staticmethod
    def bandpass(freq, q, sr):
        """Bandpass coefficients (constant skirt gain)."""
        w0 = 2.0 * np.pi * freq / sr
        alpha = np.sin(w0) / (2.0 * q)
        cos_w0 = np.cos(w0)
        a0 = 1.0 + alpha
        b0 = alpha / a0
        b1 = 0.0
        b2 = -alpha / a0
        a1 = (-2.0 * cos_w0) / a0
        a2 = (1.0 - alpha) / a0
        return BiquadFilter(b0, b1, b2, a1, a2)

    @staticmethod
    def low_shelf(freq, gain_db, sr):
        """Low shelf — boost/cut below freq. gain_db in dB."""
        A = 10.0 ** (gain_db / 40.0)
        w0 = 2.0 * np.pi * freq / sr
        alpha = np.sin(w0) / 2.0 * np.sqrt(2.0)  # Q=0.707 (Butterworth slope)
        cos_w0 = np.cos(w0)
        two_sqrt_A_alpha = 2.0 * np.sqrt(A) * alpha
        a0 = (A + 1) + (A - 1) * cos_w0 + two_sqrt_A_alpha
        b0 = (A * ((A + 1) - (A - 1) * cos_w0 + two_sqrt_A_alpha)) / a0
        b1 = (2.0 * A * ((A - 1) - (A + 1) * cos_w0)) / a0
        b2 = (A * ((A + 1) - (A - 1) * cos_w0 - two_sqrt_A_alpha)) / a0
        a1 = (-2.0 * ((A - 1) + (A + 1) * cos_w0)) / a0
        a2 = ((A + 1) + (A - 1) * cos_w0 - two_sqrt_A_alpha) / a0
        return BiquadFilter(b0, b1, b2, a1, a2)

    @staticmethod
    def high_shelf(freq, gain_db, sr):
        """High shelf — boost/cut above freq. gain_db in dB."""
        A = 10.0 ** (gain_db / 40.0)
        w0 = 2.0 * np.pi * freq / sr
        alpha = np.sin(w0) / 2.0 * np.sqrt(2.0)
        cos_w0 = np.cos(w0)
        two_sqrt_A_alpha = 2.0 * np.sqrt(A) * alpha
        a0 = (A + 1) - (A - 1) * cos_w0 + two_sqrt_A_alpha
        b0 = (A * ((A + 1) + (A - 1) * cos_w0 + two_sqrt_A_alpha)) / a0
        b1 = (-2.0 * A * ((A - 1) + (A + 1) * cos_w0)) / a0
        b2 = (A * ((A + 1) + (A - 1) * cos_w0 - two_sqrt_A_alpha)) / a0
        a1 = (2.0 * ((A - 1) - (A + 1) * cos_w0)) / a0
        a2 = ((A + 1) - (A - 1) * cos_w0 - two_sqrt_A_alpha) / a0
        return BiquadFilter(b0, b1, b2, a1, a2)


class AllpassFilter:
    """Delay-based allpass filter (Schroeder allpass).

    Structure:
        output = -g * input + delayed + g * delayed_output

    Passes all frequencies at equal amplitude but smears their timing.
    Chain several together to turn a sharp transient into a diffuse cloud.
    """

    def __init__(self, delay_samples: int, gain: float = 0.5):
        self.delay = delay_samples
        self.gain = gain
        self.buffer = np.zeros(delay_samples, dtype=np.float64)
        self.idx = 0

    def process(self, x: float) -> float:
        delayed = self.buffer[self.idx]
        # v = input + feedback from delayed output
        v = x + self.gain * delayed
        # output = feedforward + delayed
        y = -self.gain * v + delayed
        self.buffer[self.idx] = v
        self.idx = (self.idx + 1) % self.delay
        return y

    def reset(self):
        self.buffer[:] = 0.0
        self.idx = 0
