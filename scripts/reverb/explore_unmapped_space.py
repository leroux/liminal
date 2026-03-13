#!/usr/bin/env python3
"""Explore the parameter space dimensions that the simplified macros can't reach.

For each "locked out" dimension, generate examples that maximize variation
in that dimension while keeping other params in a reasonable range.
Render through the Rust DSP and compare against the closest macro-reachable version.

Outputs WAV files so you can actually listen to the difference.
"""

import json
import sys
import wave
import struct
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.simplified_params import ReverbParams, SR, N
from reverb_rust import render_fdn

OUT_DIR = Path(__file__).parent / "unmapped_exploration"


def render_ir(params: dict, duration: float = 3.0) -> np.ndarray:
    """Render impulse response through Rust DSP, return mono."""
    ir_len = int(SR * duration)
    impulse = np.zeros(ir_len)
    impulse[0] = 1.0
    clean = {k: v for k, v in params.items() if not k.startswith("_")}
    out = render_fdn(impulse, json.dumps(clean))
    return out.reshape(-1, 2).mean(axis=1)


def save_wav(path: Path, signal: np.ndarray, sr: int = 44100):
    """Save mono signal as 16-bit WAV."""
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * 0.9
    with wave.open(str(path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        for s in signal:
            w.writeframes(struct.pack("<h", int(s * 32767)))


def base_params() -> dict:
    """A nice medium room as the baseline."""
    s = ReverbParams(size=0.5, decay=0.85, brightness=0.7,
                                diffusion=0.5, mix=1.0, saturation=0.0,
                                pre_delay_ms=10.0, stereo_width=1.0)
    return s.to_fdn_params()


def macro_closest(params: dict) -> dict:
    """Round-trip through simplified macros — the closest the macros can get."""
    s = ReverbParams.from_fdn_params(params)
    return s.to_fdn_params()


def spectral_centroid(signal: np.ndarray) -> float:
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / SR)
    total = np.sum(spectrum)
    return float(np.sum(freqs * spectrum) / total) if total > 1e-15 else 0.0


def estimate_rt60(signal: np.ndarray) -> float:
    energy = signal ** 2
    decay = np.cumsum(energy[::-1])[::-1]
    decay = decay / max(decay[0], 1e-15)
    decay_db = 10 * np.log10(decay + 1e-15)
    idx_5 = int(np.searchsorted(-decay_db, 5))
    idx_25 = int(np.searchsorted(-decay_db, 25))
    if idx_25 > idx_5 and idx_25 < len(decay_db):
        t = np.arange(idx_5, idx_25) / SR
        db = decay_db[idx_5:idx_25]
        if len(t) > 1:
            slope = np.polyfit(t, db, 1)[0]
            if slope < -0.1:
                return -60.0 / slope
    return float("inf")


def rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(signal ** 2)))


def explore_nonuniform_damping():
    """Different damping per node — frequency-dependent decay."""
    experiments = {
        "damping_low_nodes_dark": {
            # Low nodes (long delays) heavily damped, high nodes bright
            "damping_coeffs": [0.8, 0.7, 0.6, 0.5, 0.2, 0.15, 0.1, 0.05],
        },
        "damping_high_nodes_dark": {
            # Opposite: high nodes damped, low nodes ring
            "damping_coeffs": [0.05, 0.1, 0.15, 0.2, 0.5, 0.6, 0.7, 0.8],
        },
        "damping_alternating": {
            # Alternating bright/dark nodes — comb-like decay
            "damping_coeffs": [0.8, 0.1, 0.8, 0.1, 0.8, 0.1, 0.8, 0.1],
        },
        "damping_one_resonant": {
            # One node barely damped, rest heavily damped — resonant peak
            "damping_coeffs": [0.8, 0.8, 0.8, 0.02, 0.8, 0.8, 0.8, 0.8],
        },
    }
    return "nonuniform_damping", experiments


def explore_nonuniform_gains():
    """Non-uniform input/output gains — resonant emphasis."""
    experiments = {
        "gains_emphasize_low": {
            "input_gains": [0.3, 0.25, 0.2, 0.1, 0.05, 0.04, 0.03, 0.03],
            "output_gains": [2.0, 1.5, 1.0, 0.8, 0.5, 0.3, 0.2, 0.1],
        },
        "gains_emphasize_high": {
            "input_gains": [0.03, 0.03, 0.04, 0.05, 0.1, 0.2, 0.25, 0.3],
            "output_gains": [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0],
        },
        "gains_sparse": {
            # Only 2 nodes active — very sparse reverb
            "input_gains": [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5],
            "output_gains": [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0],
        },
        "gains_single_node": {
            # Single node — pure comb filter
            "input_gains": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "output_gains": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
    }
    return "nonuniform_gains", experiments


def explore_broken_ratios():
    """Delay times that break the fixed ratio template."""
    experiments = {
        "ratios_clustered_short": {
            # All delays very close together — flutter echo territory
            "delay_times": [800, 820, 840, 860, 880, 900, 920, 940],
        },
        "ratios_clustered_long": {
            # Same but longer
            "delay_times": [4000, 4050, 4100, 4150, 4200, 4250, 4300, 4350],
        },
        "ratios_harmonic": {
            # Harmonic series — pitched resonance
            "delay_times": [441, 882, 1323, 1764, 2205, 2646, 3087, 3528],
        },
        "ratios_power_of_two": {
            # Powers of 2 — strong coloration
            "delay_times": [256, 512, 1024, 2048, 256, 512, 1024, 2048],
        },
        "ratios_extreme_spread": {
            # Huge gap between shortest and longest
            "delay_times": [44, 88, 176, 352, 2000, 4000, 8000, 13000],
        },
        "ratios_prime": {
            # Prime number samples — maximally incommensurate, dense reverb
            "delay_times": [547, 839, 1087, 1297, 1583, 1871, 2203, 2549],
        },
    }
    return "broken_ratios", experiments


def explore_combined_extremes():
    """Combinations of non-uniform params that interact."""
    experiments = {
        "metallic_resonator": {
            # Short clustered delays + non-uniform damping + sparse gains
            "delay_times": [200, 210, 220, 230, 240, 250, 260, 270],
            "damping_coeffs": [0.01, 0.9, 0.01, 0.9, 0.01, 0.9, 0.01, 0.9],
            "feedback_gain": 0.95,
        },
        "frequency_dependent_space": {
            # Simulating a real room: low freq decays longer
            "delay_times": [1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300],
            "damping_coeffs": [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            "feedback_gain": 0.88,
        },
        "lo_fi_degraded": {
            # Heavy non-uniform damping + saturation + sparse
            "damping_coeffs": [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55],
            "input_gains": [0.3, 0.0, 0.2, 0.0, 0.3, 0.0, 0.2, 0.0],
            "output_gains": [1.5, 0.0, 1.0, 0.0, 1.5, 0.0, 1.0, 0.0],
            "saturation": 0.8,
            "feedback_gain": 0.92,
        },
    }
    return "combined_extremes", experiments


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = base_params()

    explorations = [
        explore_nonuniform_damping(),
        explore_nonuniform_gains(),
        explore_broken_ratios(),
        explore_combined_extremes(),
    ]

    print(f"{'=' * 90}")
    print("EXPLORING UNMAPPED PARAMETER SPACE")
    print("Rendering sounds the simplified macros CANNOT produce")
    print(f"{'=' * 90}")

    # Render baseline
    base_ir = render_ir(base)
    save_wav(OUT_DIR / "00_baseline_macro.wav", base_ir)
    base_rt60 = estimate_rt60(base_ir)
    base_sc = spectral_centroid(base_ir)
    base_rms_val = rms(base_ir)
    print(f"\nBaseline: RT60={base_rt60:.2f}s  SC={base_sc:.0f}Hz  RMS={base_rms_val:.4f}")

    all_results = []

    for category, experiments in explorations:
        print(f"\n{'─' * 90}")
        print(f"  {category.upper()}")
        print(f"{'─' * 90}")

        for name, overrides in experiments.items():
            # Build full params with overrides
            params = base.copy()
            for k, v in overrides.items():
                params[k] = v

            # Render the "unreachable" version
            try:
                ir = render_ir(params)
            except Exception as e:
                print(f"  SKIP {name}: {e}")
                continue

            ir_rms = rms(ir)
            if ir_rms < 1e-10:
                print(f"  SKIP {name}: silent output")
                continue

            # Render the closest macro version
            macro_params = macro_closest(params)
            macro_ir = render_ir(macro_params)

            # Compare
            rt60_full = estimate_rt60(ir)
            rt60_macro = estimate_rt60(macro_ir)
            sc_full = spectral_centroid(ir)
            sc_macro = spectral_centroid(macro_ir)

            min_len = min(len(ir), len(macro_ir))
            diff_rms = rms(ir[:min_len] - macro_ir[:min_len])
            diff_db = 20 * np.log10(diff_rms / max(ir_rms, 1e-15))

            # Energy envelope correlation
            from scripts.reverb_pca import energy_envelope
            env_full = energy_envelope(ir)
            env_macro = energy_envelope(macro_ir)
            ml = min(len(env_full), len(env_macro))
            if ml > 10 and np.std(env_full[:ml]) > 1e-10:
                env_corr = float(np.corrcoef(env_full[:ml], env_macro[:ml])[0, 1])
            else:
                env_corr = 1.0

            # Save WAVs
            save_wav(OUT_DIR / f"{category}_{name}_full.wav", ir)
            save_wav(OUT_DIR / f"{category}_{name}_macro.wav", macro_ir)

            interesting = ""
            if env_corr < 0.8:
                interesting = " ★ SOUNDS DIFFERENT"
            elif env_corr < 0.95:
                interesting = " ~ somewhat different"

            print(f"  {name:<35} RT60: {rt60_full:.2f}→{rt60_macro:.2f}  "
                  f"SC: {sc_full:.0f}→{sc_macro:.0f}Hz  "
                  f"EnvCorr: {env_corr:.3f}  DiffDB: {diff_db:+.1f}{interesting}")

            all_results.append({
                "category": category,
                "name": name,
                "env_corr": env_corr,
                "rt60_full": rt60_full,
                "rt60_macro": rt60_macro,
                "sc_full": sc_full,
                "sc_macro": sc_macro,
                "diff_db": diff_db,
            })

    # Summary
    print(f"\n{'=' * 90}")
    print("SUMMARY: What are we missing?")
    print(f"{'=' * 90}")

    different = [r for r in all_results if r["env_corr"] < 0.90]
    similar = [r for r in all_results if r["env_corr"] >= 0.90]

    print(f"\n  Sounds DIFFERENT from closest macro ({len(different)}/{len(all_results)}):")
    for r in sorted(different, key=lambda x: x["env_corr"]):
        print(f"    {r['category']}/{r['name']:<35} EnvCorr={r['env_corr']:.3f}  "
              f"RT60: {r['rt60_full']:.2f}→{r['rt60_macro']:.2f}  "
              f"SC: {r['sc_full']:.0f}→{r['sc_macro']:.0f}Hz")

    print(f"\n  Sounds SIMILAR to closest macro ({len(similar)}/{len(all_results)}):")
    for r in sorted(similar, key=lambda x: x["env_corr"]):
        print(f"    {r['category']}/{r['name']:<35} EnvCorr={r['env_corr']:.3f}")

    n_files = len(list(OUT_DIR.glob("*.wav")))
    print(f"\n  {n_files} WAV files saved to {OUT_DIR}/")
    print("  Listen to *_full.wav vs *_macro.wav pairs to hear the difference.")


if __name__ == "__main__":
    main()
