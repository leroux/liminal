"""Hand-tuning & edge cases — generates WAV files + saves presets as JSON.

Run: uv run python tests/test_edge_cases.py
"""

import numpy as np
from scipy.io import wavfile
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.fdn import render_fdn
from engine.params import default_params, SR

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "audio", "test_signals")
PRESET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "gui", "presets")


def save_wav(filename, audio, sr=SR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.9
    wavfile.write(path, sr, (audio * 32767).astype(np.int16))
    print(f"  Saved {path} ({len(audio)/sr:.2f}s)")


def save_preset(name, params):
    os.makedirs(PRESET_DIR, exist_ok=True)
    path = os.path.join(PRESET_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"  Preset saved: {path}")


def make_noise_burst(total_seconds=4.0):
    signal = np.zeros(int(SR * total_seconds))
    burst_len = int(SR * 15 / 1000)
    burst = np.random.randn(burst_len)
    burst *= np.linspace(1.0, 0.0, burst_len)
    signal[:burst_len] = burst
    return signal


def make_chords(total_seconds=6.0):
    def pluck(freq, duration):
        t = np.arange(int(SR * duration)) / SR
        sig = np.sin(2*np.pi*freq*t) + 0.5*np.sin(2*np.pi*freq*2*t)
        return sig * np.exp(-t * 3.0)

    total = int(SR * total_seconds)
    audio = np.zeros(total)
    chords = [(0, [261.6, 329.6, 392.0]), (1.5, [220.0, 261.6, 329.6]),
              (3.0, [174.6, 220.0, 261.6]), (4.5, [196.0, 246.9, 293.7])]
    for t_off, freqs in chords:
        offset = int(t_off * SR)
        for f in freqs:
            p = pluck(f, 1.8)
            end = min(offset + len(p), total)
            audio[offset:end] += p[:end-offset] * 0.25
    return audio


def render_and_save(params, burst, chords, prefix, label):
    print(f"\n--- {label} ---")
    t0 = time.time()
    # Burst test (100% wet for clarity)
    p = dict(params)
    p["wet_dry"] = 1.0
    out_burst = render_fdn(burst, p)
    # Chords test (use preset wet_dry)
    out_chords = render_fdn(chords, params)
    elapsed = time.time() - t0
    save_wav(f"{prefix}_burst.wav", out_burst)
    save_wav(f"{prefix}_chords.wav", out_chords)
    return elapsed


# ---------------------------------------------------------------------------
# Edge Case 1: feedback=0 (multi-tap delay, no recirculation)
# ---------------------------------------------------------------------------
def test_multitap(burst, chords):
    params = default_params()
    params["feedback_gain"] = 0.0
    params["wet_dry"] = 0.5
    render_and_save(params, burst, chords, "40_multitap", "feedback=0 (multi-tap delay)")
    save_preset("multitap_delay", params)


# ---------------------------------------------------------------------------
# Edge Case 2: all damping=1 (maximum lowpass, almost no highs pass)
# ---------------------------------------------------------------------------
def test_max_damping(burst, chords):
    params = default_params()
    params["damping_coeffs"] = [0.99] * 8
    params["feedback_gain"] = 0.9
    params["wet_dry"] = 0.6
    render_and_save(params, burst, chords, "41_maxdamp", "damping=0.99 (extreme lowpass)")
    save_preset("max_damping", params)


# ---------------------------------------------------------------------------
# Edge Case 3: short delays <10ms (resonator territory)
# ---------------------------------------------------------------------------
def test_resonator(burst, chords):
    params = default_params()
    # Very short delays — will produce pitched tones
    params["delay_times"] = [
        int(2.3 / 1000 * SR),   # ~435 Hz
        int(3.1 / 1000 * SR),   # ~323 Hz
        int(3.7 / 1000 * SR),   # ~270 Hz
        int(4.3 / 1000 * SR),   # ~233 Hz
        int(5.3 / 1000 * SR),   # ~189 Hz
        int(6.1 / 1000 * SR),   # ~164 Hz
        int(7.1 / 1000 * SR),   # ~141 Hz
        int(8.3 / 1000 * SR),   # ~120 Hz
    ]
    params["feedback_gain"] = 0.95
    params["damping_coeffs"] = [0.1] * 8
    params["wet_dry"] = 0.7
    render_and_save(params, burst, chords, "42_resonator", "short delays <10ms (resonator)")
    save_preset("resonator", params)


# ---------------------------------------------------------------------------
# Edge Case 4: feedback >1.0 (controlled explosion with soft clipping)
# ---------------------------------------------------------------------------
def test_explosion(burst, chords):
    params = default_params()
    params["feedback_gain"] = 1.05
    params["damping_coeffs"] = [0.5] * 8  # damping fights the explosion
    params["wet_dry"] = 0.5

    # Render with soft clipping to prevent full blowup
    p_wet = dict(params)
    p_wet["wet_dry"] = 1.0
    out = render_fdn(burst, p_wet)
    # Soft clip: tanh saturation
    out = np.tanh(out * 2.0) * 0.5
    save_wav("43_explosion_burst.wav", out)

    out = render_fdn(chords, params)
    out = np.tanh(out * 2.0) * 0.5
    save_wav("43_explosion_chords.wav", out)
    print("  (soft-clipped with tanh to tame the explosion)")
    save_preset("controlled_explosion", params)


# ---------------------------------------------------------------------------
# Hand-tuned preset: "Decent Room"
# ---------------------------------------------------------------------------
def test_decent_room(burst, chords):
    params = default_params()
    # Medium delay times — room-sized
    params["delay_times"] = [
        int(19.3 / 1000 * SR),
        int(23.7 / 1000 * SR),
        int(29.1 / 1000 * SR),
        int(34.3 / 1000 * SR),
        int(39.7 / 1000 * SR),
        int(44.9 / 1000 * SR),
        int(51.3 / 1000 * SR),
        int(57.1 / 1000 * SR),
    ]
    params["feedback_gain"] = 0.75
    params["damping_coeffs"] = [0.4] * 8
    params["pre_delay"] = int(8.0 / 1000 * SR)
    params["diffusion"] = 0.6
    params["wet_dry"] = 0.35
    render_and_save(params, burst, chords, "44_room", "Hand-tuned: Decent Room")
    save_preset("decent_room", params)


# ---------------------------------------------------------------------------
# Hand-tuned preset: "Large Hall"
# ---------------------------------------------------------------------------
def test_large_hall(burst, chords):
    params = default_params()
    params["delay_times"] = [
        int(53.1 / 1000 * SR),
        int(67.3 / 1000 * SR),
        int(79.7 / 1000 * SR),
        int(97.1 / 1000 * SR),
        int(113.3 / 1000 * SR),
        int(131.7 / 1000 * SR),
        int(149.3 / 1000 * SR),
        int(167.9 / 1000 * SR),
    ]
    params["feedback_gain"] = 0.92
    params["damping_coeffs"] = [0.5] * 8
    params["pre_delay"] = int(25.0 / 1000 * SR)
    params["diffusion"] = 0.55
    params["wet_dry"] = 0.45
    render_and_save(params, burst, chords, "45_hall", "Hand-tuned: Large Hall")
    save_preset("large_hall", params)


# ---------------------------------------------------------------------------
# Hand-tuned preset: "Plate"
# ---------------------------------------------------------------------------
def test_plate(burst, chords):
    params = default_params()
    # Tight, bright delays — plate reverb character
    params["delay_times"] = [
        int(13.1 / 1000 * SR),
        int(17.3 / 1000 * SR),
        int(21.7 / 1000 * SR),
        int(27.1 / 1000 * SR),
        int(31.3 / 1000 * SR),
        int(37.9 / 1000 * SR),
        int(43.1 / 1000 * SR),
        int(47.3 / 1000 * SR),
    ]
    params["feedback_gain"] = 0.82
    params["damping_coeffs"] = [0.15] * 8  # bright
    params["pre_delay"] = int(3.0 / 1000 * SR)
    params["diffusion"] = 0.65
    params["wet_dry"] = 0.4
    render_and_save(params, burst, chords, "46_plate", "Hand-tuned: Plate")
    save_preset("plate", params)


# ---------------------------------------------------------------------------
# Hand-tuned preset: "Dark Cathedral"
# ---------------------------------------------------------------------------
def test_cathedral(burst, chords):
    params = default_params()
    params["delay_times"] = [
        int(89.3 / 1000 * SR),
        int(107.1 / 1000 * SR),
        int(127.7 / 1000 * SR),
        int(151.3 / 1000 * SR),
        int(173.9 / 1000 * SR),
        int(199.1 / 1000 * SR),
        int(223.7 / 1000 * SR),
        int(251.3 / 1000 * SR),
    ]
    params["feedback_gain"] = 0.94
    params["damping_coeffs"] = [0.7] * 8  # dark
    params["pre_delay"] = int(40.0 / 1000 * SR)
    params["diffusion"] = 0.6
    params["wet_dry"] = 0.5
    render_and_save(params, burst, chords, "47_cathedral", "Hand-tuned: Dark Cathedral")
    save_preset("dark_cathedral", params)


if __name__ == "__main__":
    print(f"Sample rate: {SR} Hz")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Preset dir: {PRESET_DIR}")

    burst = make_noise_burst(total_seconds=4.0)
    chords = make_chords(total_seconds=8.0)

    # Edge cases
    test_multitap(burst, chords)
    test_max_damping(burst, chords)
    test_resonator(burst, chords)
    test_explosion(burst, chords)

    # Hand-tuned presets
    test_decent_room(burst, chords)
    test_large_hall(burst, chords)
    test_plate(burst, chords)
    test_cathedral(burst, chords)

    print("\n\nDone! Presets saved in gui/presets/. Compare the different characters.")
