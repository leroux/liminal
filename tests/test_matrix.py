"""Test feedback matrix — generates WAV files.

Run: uv run python tests/test_matrix.py

Key test: compare diagonal (isolated combs, metallic) vs Householder (smooth, coupled).
"""

import numpy as np
from scipy.io import wavfile
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from primitives.delay_line import DelayLine
from primitives.filters import OnePoleFilter
from primitives.matrix import householder, apply_householder, diagonal, apply_matrix

SR = 44100
N = 8  # 8-node FDN
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "audio", "test_signals")


def save_wav(filename, audio, sr=SR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.9
    wavfile.write(path, sr, (audio * 32767).astype(np.int16))
    print(f"  Saved {path} ({len(audio)/sr:.2f}s)")


# ---------------------------------------------------------------------------
# Test 1: Verify Householder matrix properties
# ---------------------------------------------------------------------------
def test_properties():
    print("Test 1: Householder matrix properties")
    H = householder(N)

    # Unitary: H @ H^T = I
    product = H @ H.T
    assert np.allclose(product, np.eye(N)), "Not unitary!"
    print("  Unitary (energy-preserving): OK")

    # Determinant should be ±1
    det = np.linalg.det(H)
    assert np.isclose(abs(det), 1.0), f"Det = {det}, expected ±1"
    print(f"  Determinant: {det:.1f}")

    # O(N) apply matches matrix multiply
    x = np.random.randn(N)
    y_matrix = H @ x
    y_fast = apply_householder(x)
    assert np.allclose(y_matrix, y_fast), "O(N) apply doesn't match!"
    print("  O(N) apply matches matrix multiply: OK")

    print(f"  Matrix:\n{np.array2string(H, precision=3, suppress_small=True)}")


# ---------------------------------------------------------------------------
# Test 2: Mini FDN — diagonal vs Householder
# ---------------------------------------------------------------------------
def run_mini_fdn(apply_fn, label, delay_times_ms, feedback, damping_coeff):
    """Run a minimal 8-node FDN and save the output."""
    duration = 4.0
    n_samples = int(SR * duration)

    delay_samples = [int(d / 1000 * SR) for d in delay_times_ms]
    delays = [DelayLine(max_delay=max(delay_samples) + 1) for _ in range(N)]
    dampers = [OnePoleFilter(coeff=damping_coeff) for _ in range(N)]

    # Input: impulse into all nodes equally
    input_signal = np.zeros(n_samples)
    input_signal[0] = 1.0

    output = np.zeros(n_samples)

    for i in range(n_samples):
        # Read from all delay lines
        reads = np.array([delays[j].read(delay_samples[j]) for j in range(N)])

        # Apply damping
        damped = np.array([dampers[j].process(reads[j]) for j in range(N)])

        # Apply feedback matrix
        mixed = apply_fn(damped)

        # Scale by feedback gain, add input, write back
        inp = input_signal[i] / N  # distribute input equally
        for j in range(N):
            delays[j].write(feedback * mixed[j] + inp)

        # Output: sum all nodes
        output[i] = np.sum(reads) / N

    save_wav(f"16_fdn_{label}.wav", output)


def test_diagonal_vs_householder():
    print("Test 2: Diagonal (isolated combs) vs Householder (coupled)")

    # Prime-ish delay times in ms — avoids common factors for denser echo pattern
    delay_times_ms = [29.7, 37.1, 41.3, 47.9, 53.1, 59.3, 67.7, 73.1]

    # Diagonal — each delay line isolated, sounds metallic
    diag = diagonal(N)
    run_mini_fdn(lambda x: apply_matrix(diag, x), "diagonal",
                 delay_times_ms, feedback=0.85, damping_coeff=0.3)

    # Householder — all nodes coupled, sounds smooth
    run_mini_fdn(apply_householder, "householder",
                 delay_times_ms, feedback=0.85, damping_coeff=0.3)


# ---------------------------------------------------------------------------
# Test 3: Householder with different feedback gains
# ---------------------------------------------------------------------------
def test_feedback_levels():
    print("Test 3: Householder FDN — varying feedback")
    delay_times_ms = [29.7, 37.1, 41.3, 47.9, 53.1, 59.3, 67.7, 73.1]

    for fb in [0.5, 0.85, 0.95]:
        run_mini_fdn(apply_householder, f"householder_fb{fb}",
                     delay_times_ms, feedback=fb, damping_coeff=0.3)


if __name__ == "__main__":
    print(f"Sample rate: {SR} Hz")
    print(f"Output dir: {OUTPUT_DIR}\n")
    test_properties()
    print()
    test_diagonal_vs_householder()
    print()
    test_feedback_levels()
    print("\nDone! Compare diagonal vs householder — diagonal is metallic, householder is smooth.")
