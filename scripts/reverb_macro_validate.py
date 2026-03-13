#!/usr/bin/env python3
"""Validate that macro controls can reconstruct existing presets.

For each hand-designed preset:
1. Load the full FdnParams
2. Fit the best ReverbParams via from_fdn_params (reverse mapping)
3. Expand back to full params via to_fdn_params (forward mapping)
4. Compare original vs reconstructed parameter values
5. Report reconstruction error

Throwaway analysis script — not part of the production codebase.
"""

import json
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.simplified_params import ReverbParams, SR, N

PRESET_DIR = Path(__file__).parent.parent / "reverb" / "gui" / "presets"


def load_presets() -> list[tuple[str, dict]]:
    """Load all preset JSON files."""
    presets = []
    for p in sorted(PRESET_DIR.glob("*.json")):
        with open(p) as f:
            data = json.load(f)
        presets.append((p.stem, data))
    return presets


def get_defaults() -> dict:
    """Return default ReverbParams."""
    delay_ms = [29.7, 37.1, 41.3, 47.9, 53.1, 59.3, 67.7, 73.1]
    diffusion_ms = [5.3, 7.9, 11.7, 16.1]
    return {
        "delay_times": [int(ms / 1000.0 * SR) for ms in delay_ms],
        "damping_coeffs": [0.3] * N,
        "input_gains": [1.0 / N] * N,
        "output_gains": [1.0] * N,
        "node_pans": [-1.0, -0.714, -0.429, -0.143, 0.143, 0.429, 0.714, 1.0],
        "feedback_gain": 0.85,
        "wet_dry": 0.5,
        "diffusion": 0.5,
        "diffusion_stages": 4,
        "diffusion_delays": [int(ms / 1000.0 * SR) for ms in diffusion_ms],
        "saturation": 0.0,
        "pre_delay": int(10.0 / 1000.0 * SR),
        "stereo_width": 1.0,
        "matrix_type": "householder",
        "matrix_seed": 42,
        "mod_master_rate": 0.0,
        "mod_node_rate_mult": [1.0] * N,
        "mod_correlation": 1.0,
        "mod_waveform": 0,
        "mod_depth_delay": [0.0] * N,
        "mod_depth_damping": [0.0] * N,
        "mod_depth_output": [0.0] * N,
        "mod_depth_matrix": 0.0,
    }


def fill_defaults(preset: dict) -> dict:
    """Fill missing keys with defaults."""
    defaults = get_defaults()
    result = defaults.copy()
    result.update(preset)
    return result


def compare_params(original: dict, reconstructed: dict) -> dict:
    """Compare original and reconstructed params. Returns error metrics."""
    errors = {}

    # Per-node arrays: compare as normalized vectors
    for key in ["delay_times", "damping_coeffs", "input_gains", "output_gains",
                 "node_pans", "diffusion_delays"]:
        orig = np.array(original.get(key, [0.0] * N), dtype=np.float64)
        recon = np.array(reconstructed.get(key, [0.0] * N), dtype=np.float64)

        # Normalize delay_times to ms for meaningful comparison
        if key in ("delay_times", "diffusion_delays"):
            orig = orig / SR * 1000.0
            recon = recon / SR * 1000.0

        if np.max(np.abs(orig)) > 1e-10:
            rel_err = np.mean(np.abs(orig - recon)) / np.mean(np.abs(orig))
        else:
            rel_err = np.mean(np.abs(orig - recon))

        abs_err = np.mean(np.abs(orig - recon))
        errors[key] = {"relative": rel_err, "absolute": abs_err}

    # Scalar params
    for key in ["feedback_gain", "wet_dry", "diffusion", "saturation",
                 "pre_delay", "stereo_width"]:
        orig = float(original.get(key, 0.0))
        recon = float(reconstructed.get(key, 0.0))

        if key == "pre_delay":
            orig = orig / SR * 1000.0
            recon = recon / SR * 1000.0

        abs_err = abs(orig - recon)
        rel_err = abs_err / max(abs(orig), 1e-10)
        errors[key] = {"relative": rel_err, "absolute": abs_err}

    # Matrix type (categorical)
    orig_mt = original.get("matrix_type", "householder")
    recon_mt = reconstructed.get("matrix_type", "householder")
    errors["matrix_type"] = {"match": orig_mt == recon_mt, "original": orig_mt, "reconstructed": recon_mt}

    return errors


def classify_preset(name: str, meta: dict | None) -> str:
    cat = ""
    if meta and isinstance(meta, dict):
        cat = meta.get("category", "")
    if cat == "ML Generated" or name.startswith("z_gen_"):
        return "ml-generated"
    if name.startswith("z_"):
        return "experimental"
    return "hand-designed"


def main():
    presets = load_presets()
    defaults = get_defaults()

    print(f"{'=' * 100}")
    print("MACRO CONTROL VALIDATION: Preset Reconstruction Error")
    print(f"{'=' * 100}\n")

    # Track per-category errors
    category_errors: dict[str, list] = {"hand-designed": [], "ml-generated": [], "experimental": []}
    all_results = []

    for name, preset in presets:
        meta = preset.get("_meta", None)
        category = classify_preset(name, meta)

        # Fill defaults
        full_original = fill_defaults(preset)

        # Reverse map → simplified → forward map
        simplified = ReverbParams.from_fdn_params(full_original)
        reconstructed = simplified.to_fdn_params()

        # Compare
        errors = compare_params(full_original, reconstructed)

        # Compute overall reconstruction score
        arr_keys = ["delay_times", "damping_coeffs", "input_gains", "output_gains"]
        scalar_keys = ["feedback_gain", "wet_dry", "diffusion", "saturation", "pre_delay", "stereo_width"]

        arr_errors = [errors[k]["relative"] for k in arr_keys]
        scalar_errors = [errors[k]["absolute"] for k in scalar_keys]
        overall_arr = np.mean(arr_errors)
        overall_scalar = np.mean(scalar_errors)

        mt_match = errors["matrix_type"]["match"]

        all_results.append({
            "name": name,
            "category": category,
            "arr_error": overall_arr,
            "scalar_error": overall_scalar,
            "mt_match": mt_match,
            "simplified": simplified.to_dict(),
            "errors": errors,
        })

        category_errors[category].append(overall_arr)

    # Sort by error (worst first)
    all_results.sort(key=lambda r: r["arr_error"], reverse=True)

    # Print detailed results
    print(f"{'Preset':<45} {'Category':<15} {'Array RelErr':>12} {'Scalar AbsErr':>13} {'Matrix':>7}")
    print("-" * 100)

    for r in all_results:
        mt_str = "OK" if r["mt_match"] else f"MISS ({r['errors']['matrix_type']['original']}→{r['errors']['matrix_type']['reconstructed']})"
        print(f"{r['name']:<45} {r['category']:<15} {r['arr_error']:>12.4f} {r['scalar_error']:>13.4f} {mt_str:>7}")

    # Summary per category
    print(f"\n{'=' * 100}")
    print("SUMMARY BY CATEGORY")
    print(f"{'=' * 100}")

    for cat in ["hand-designed", "ml-generated", "experimental"]:
        errs = category_errors[cat]
        if not errs:
            continue
        errs = np.array(errs)
        print(f"\n  {cat} ({len(errs)} presets):")
        print(f"    Mean array relative error: {errs.mean():.4f}")
        print(f"    Max array relative error:  {errs.max():.4f}")
        print(f"    Median:                    {np.median(errs):.4f}")
        print(f"    < 10% error:               {np.sum(errs < 0.10)}/{len(errs)}")
        print(f"    < 20% error:               {np.sum(errs < 0.20)}/{len(errs)}")

    # Print worst offenders detail
    print(f"\n{'=' * 100}")
    print("TOP 10 WORST RECONSTRUCTIONS (highest array error)")
    print(f"{'=' * 100}")

    for r in all_results[:10]:
        print(f"\n  {r['name']} ({r['category']}) — overall array error: {r['arr_error']:.4f}")
        for key in ["delay_times", "damping_coeffs", "input_gains", "output_gains"]:
            err = r["errors"][key]
            print(f"    {key:<20} rel={err['relative']:.4f}  abs={err['absolute']:.4f}")
        print(f"    Simplified: size={r['simplified']['size']:.3f} decay={r['simplified']['decay']:.3f} "
              f"brightness={r['simplified']['brightness']:.3f}")

    # Print best hand-designed reconstructions
    hand_results = [r for r in all_results if r["category"] == "hand-designed"]
    hand_results.sort(key=lambda r: r["arr_error"])
    print(f"\n{'=' * 100}")
    print("BEST HAND-DESIGNED RECONSTRUCTIONS (lowest array error)")
    print(f"{'=' * 100}")
    for r in hand_results[:10]:
        print(f"  {r['name']:<45} error={r['arr_error']:.4f}")

    # Round-trip test: simplified → full → simplified should be stable
    print(f"\n{'=' * 100}")
    print("ROUND-TRIP STABILITY TEST")
    print(f"{'=' * 100}")

    max_roundtrip_err = 0.0
    for name, preset in presets:
        full = fill_defaults(preset)
        s1 = ReverbParams.from_fdn_params(full)
        full2 = s1.to_fdn_params()
        s2 = ReverbParams.from_fdn_params(full2)

        # Compare s1 vs s2 (should be nearly identical)
        d1 = s1.to_dict()
        d2 = s2.to_dict()
        for key in d1:
            if isinstance(d1[key], (int, float)):
                err = abs(d1[key] - d2[key])
                max_roundtrip_err = max(max_roundtrip_err, err)
                if err > 0.01:
                    print(f"  WARNING: {name}.{key}: {d1[key]:.4f} → {d2[key]:.4f} (err={err:.4f})")

    print(f"\n  Max round-trip error (simplified → full → simplified): {max_roundtrip_err:.6f}")
    if max_roundtrip_err < 0.01:
        print("  PASS: Round-trip is stable.")
    else:
        print("  WARN: Round-trip has significant drift.")

    # --- Audio-level validation ---
    audio_validate(presets, defaults)


def estimate_rt60(signal: np.ndarray, sr: float) -> float:
    """Estimate RT60 from energy decay curve using Schroeder integration."""
    energy = signal ** 2
    # Schroeder backward integration
    decay = np.cumsum(energy[::-1])[::-1]
    decay = decay / max(decay[0], 1e-15)
    decay_db = 10 * np.log10(decay + 1e-15)

    # Find -60dB point (or extrapolate from -20dB to -30dB range)
    idx_5 = np.searchsorted(-decay_db, 5)
    idx_25 = np.searchsorted(-decay_db, 25)
    if idx_25 > idx_5 and idx_25 < len(decay_db):
        # Linear fit on the -5 to -25dB region, extrapolate to -60dB
        t = np.arange(idx_5, idx_25) / sr
        db = decay_db[idx_5:idx_25]
        if len(t) > 1:
            slope = np.polyfit(t, db, 1)[0]
            if slope < -0.1:
                return -60.0 / slope
    return float('inf')


def spectral_centroid(signal: np.ndarray, sr: float) -> float:
    """Compute spectral centroid in Hz."""
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / sr)
    total = np.sum(spectrum)
    if total > 1e-15:
        return float(np.sum(freqs * spectrum) / total)
    return 0.0


def energy_envelope(signal: np.ndarray, window_ms: float = 50.0, sr: float = 44100.0) -> np.ndarray:
    """Compute smoothed RMS energy envelope."""
    window = max(1, int(sr * window_ms / 1000.0))
    squared = signal ** 2
    # Cumulative sum for efficient moving average
    cumsum = np.cumsum(np.concatenate([[0], squared]))
    envelope = np.sqrt((cumsum[window:] - cumsum[:-window]) / window)
    return envelope


def audio_validate(presets: list[tuple[str, dict]], defaults: dict):
    """Render impulse responses through the Rust DSP and compare perceptual characteristics.

    Reverb IRs are chaotic — sample-by-sample comparison is meaningless.
    Instead, compare perceptual properties: RT60, spectral centroid, energy envelope shape.
    """
    from reverb_rust import render_fdn

    print(f"\n{'=' * 100}")
    print("AUDIO VALIDATION: Perceptual Comparison (Rust DSP)")
    print(f"{'=' * 100}")

    ir_seconds = 3.0
    ir_len = int(SR * ir_seconds)
    impulse = np.zeros(ir_len)
    impulse[0] = 1.0

    results = []

    for name, preset in presets:
        meta = preset.get("_meta", None)
        category = classify_preset(name, meta)

        full_original = fill_defaults(preset)
        dsp_original = {k: v for k, v in full_original.items() if not k.startswith("_")}

        simplified = ReverbParams.from_fdn_params(full_original)
        reconstructed = simplified.to_fdn_params()

        # Render both through Rust DSP
        out_orig = render_fdn(impulse, json.dumps(dsp_original)).reshape(-1, 2).mean(axis=1)
        out_recon = render_fdn(impulse, json.dumps(reconstructed)).reshape(-1, 2).mean(axis=1)

        # RT60 comparison
        rt60_orig = estimate_rt60(out_orig, SR)
        rt60_recon = estimate_rt60(out_recon, SR)
        if rt60_orig < 100 and rt60_orig > 0.01:
            rt60_err = abs(rt60_orig - rt60_recon) / rt60_orig
        else:
            rt60_err = 0.0  # infinite or near-zero RT60, skip

        # Spectral centroid comparison
        sc_orig = spectral_centroid(out_orig, SR)
        sc_recon = spectral_centroid(out_recon, SR)
        if sc_orig > 1.0:
            sc_err = abs(sc_orig - sc_recon) / sc_orig
        else:
            sc_err = 0.0

        # Energy envelope correlation (smoothed RMS shape)
        env_orig = energy_envelope(out_orig, window_ms=50.0, sr=SR)
        env_recon = energy_envelope(out_recon, window_ms=50.0, sr=SR)
        min_len = min(len(env_orig), len(env_recon))
        if min_len > 10 and np.std(env_orig[:min_len]) > 1e-10:
            env_corr = float(np.corrcoef(env_orig[:min_len], env_recon[:min_len])[0, 1])
        else:
            env_corr = 1.0

        # RMS level difference (dB)
        rms_orig = np.sqrt(np.mean(out_orig ** 2))
        rms_recon = np.sqrt(np.mean(out_recon ** 2))
        if rms_orig > 1e-15:
            level_diff_db = 20 * np.log10(rms_recon / rms_orig)
        else:
            level_diff_db = 0.0

        results.append({
            "name": name,
            "category": category,
            "rt60_orig": rt60_orig,
            "rt60_recon": rt60_recon,
            "rt60_err": rt60_err,
            "sc_orig": sc_orig,
            "sc_recon": sc_recon,
            "sc_err": sc_err,
            "env_corr": env_corr,
            "level_diff_db": level_diff_db,
        })

    # Sort by envelope correlation (worst first)
    results.sort(key=lambda r: r["env_corr"])

    print(f"\n{'Preset':<40} {'Cat':<6} {'RT60 orig':>9} {'RT60 recon':>10} {'RT60 err%':>9} "
          f"{'SC orig':>8} {'SC recon':>8} {'EnvCorr':>8} {'LvldB':>6}")
    print("-" * 115)

    for r in results:
        rt60_o = f"{r['rt60_orig']:.2f}" if r['rt60_orig'] < 100 else "inf"
        rt60_r = f"{r['rt60_recon']:.2f}" if r['rt60_recon'] < 100 else "inf"
        cat = "hand" if r["category"] == "hand-designed" else "ml"
        print(f"{r['name']:<40} {cat:<6} {rt60_o:>9}s {rt60_r:>9}s {r['rt60_err']*100:>8.1f}% "
              f"{r['sc_orig']:>7.0f}Hz {r['sc_recon']:>7.0f}Hz {r['env_corr']:>8.4f} {r['level_diff_db']:>+5.1f}")

    # Summary
    hand = [r for r in results if r["category"] == "hand-designed"]
    ml = [r for r in results if r["category"] == "ml-generated"]

    for label, group in [("Hand-designed", hand), ("ML-generated", ml)]:
        if not group:
            continue
        rt60_errs = [r["rt60_err"] for r in group if r["rt60_orig"] < 100 and r["rt60_orig"] > 0.01]
        sc_errs = [r["sc_err"] for r in group]
        env_corrs = [r["env_corr"] for r in group]
        level_diffs = [abs(r["level_diff_db"]) for r in group]

        print(f"\n  {label} ({len(group)} presets):")
        if rt60_errs:
            print(f"    RT60 error:     median={np.median(rt60_errs)*100:.1f}%  "
                  f"<10%: {sum(1 for e in rt60_errs if e < 0.10)}/{len(rt60_errs)}  "
                  f"<20%: {sum(1 for e in rt60_errs if e < 0.20)}/{len(rt60_errs)}")
        print(f"    Centroid error: median={np.median(sc_errs)*100:.1f}%  "
              f"<10%: {sum(1 for e in sc_errs if e < 0.10)}/{len(sc_errs)}")
        print(f"    Envelope corr:  median={np.median(env_corrs):.4f}  "
              f">0.90: {sum(1 for c in env_corrs if c > 0.90)}/{len(env_corrs)}  "
              f">0.95: {sum(1 for c in env_corrs if c > 0.95)}/{len(env_corrs)}")
        print(f"    Level diff:     median={np.median(level_diffs):.1f}dB  "
              f"<3dB: {sum(1 for d in level_diffs if d < 3)}/{len(level_diffs)}")


if __name__ == "__main__":
    main()
