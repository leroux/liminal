#!/usr/bin/env python3
"""Render A/B comparison WAVs for lossy macro experiment E.

For each selected preset, renders:
  - {preset}_original.wav     — full 41-param preset through DSP
  - {preset}_experiment_E.wav — macro-reconstructed (31 params, 8 fixed + 2 derived)

Uses real audio inputs from explore/inputs/ for musical context.

Requires: lossy_rust PyO3 bindings
"""

import json
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wav

from lossy_rust import render_lossy

SR = 44100
PRESET_DIR = Path(__file__).parent.parent.parent / "lossy" / "gui" / "presets"
INPUT_DIR = Path(__file__).parent.parent.parent / "explore" / "inputs"
OUT_DIR = Path(__file__).parent / "listening_test"


def get_defaults():
    return {
        "inverse": 0, "jitter": 0.0, "loss": 0.5, "window_size": 2048,
        "hop_divisor": 4, "n_bands": 21, "global_amount": 1.0,
        "phase_loss": 0.0, "quantizer": 0, "pre_echo": 0.0,
        "noise_shape": 0.0, "weighting": 1.0, "hf_threshold": 0.3,
        "transient_ratio": 4.0, "slushy_rate": 0.03, "crush": 0.0,
        "decimate": 0.0, "packets": 0, "packet_rate": 0.3,
        "packet_size": 30.0, "filter_type": 0, "filter_freq": 1000.0,
        "filter_width": 0.5, "filter_slope": 1, "verb": 0.0,
        "decay": 0.5, "verb_position": 0, "freeze": 0, "freeze_mode": 0,
        "freezer": 1.0, "gate": 0.0, "threshold": 0.5, "auto_gain": 0.0,
        "loss_gain": 0.5, "bounce": 0, "bounce_target": 0,
        "bounce_rate": 0.3, "bounce_lfo_min": 0.1, "bounce_lfo_max": 5.0,
        "wet_dry": 1.0, "seed": 42,
    }


# Experiment E config: fix 8 dead + derive decay from verb, filter_slope from filter_width
FIXED = {
    "seed": 42, "pre_echo": 0.0, "transient_ratio": 4.0,
    "noise_shape": 0.0, "slushy_rate": 0.03,
    "freeze_mode": 0, "verb_position": 0, "global_amount": 1.0,
}

# Anchored fits from PCA analysis
DECAY_SLOPE = 0.517       # decay = 0.5 + 0.517 * verb
FILTER_SLOPE_SLOPE = -2.612  # filter_slope = 1 + -2.612 * (filter_width - 0.5)


def reconstruct_E(preset, defaults):
    """Experiment E roundtrip: fix 8 params, derive decay and filter_slope."""
    full = {**defaults, **{k: v for k, v in preset.items() if k != "_meta"}}
    recon = full.copy()

    for k, v in FIXED.items():
        recon[k] = v

    # Derive decay from verb (gated: only when verb > 0)
    verb_val = float(full.get("verb", 0.0))
    if verb_val > 1e-6:
        recon["decay"] = max(0.0, min(1.0, 0.5 + DECAY_SLOPE * verb_val))
    else:
        recon["decay"] = defaults["decay"]

    # Derive filter_slope from filter_width (always active)
    fw = float(full.get("filter_width", 0.5))
    recon["filter_slope"] = int(round(max(0, min(2, 1 + FILTER_SLOPE_SLOPE * (fw - 0.5)))))

    return recon


def load_audio(path):
    """Load WAV, return mono float64 at 44100Hz, normalized to [-0.8, 0.8]."""
    sr, data = wav.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32768.0
    elif data.dtype == np.float32:
        data = data.astype(np.float64)
    if len(data.shape) > 1:
        data = data[:, 0]  # take left channel
    # Trim to 5 seconds max
    max_samples = int(5.0 * sr)
    data = data[:max_samples]
    # Normalize
    peak = np.max(np.abs(data))
    if peak > 1e-6:
        data = data * (0.8 / peak)
    return data


def save_wav(path, signal):
    data = np.clip(signal, -1.0, 1.0).astype(np.float32)
    wav.write(str(path), SR, data)


def render_preset(signal, preset, defaults):
    params = {**defaults, **{k: v for k, v in preset.items()
              if k != "_meta" and k != "tail_length"}}
    return render_lossy(signal, json.dumps(params))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    defaults = get_defaults()

    # Load inputs
    inputs = {
        "drum": load_audio(INPUT_DIR / "lofidrum.wav"),
        "chord": load_audio(INPUT_DIR / "sustained_chord.wav"),
        "hits": load_audio(INPUT_DIR / "sparse_hits.wav"),
    }

    # Save dry inputs for reference
    for name, signal in inputs.items():
        save_wav(OUT_DIR / f"00_dry_{name}.wav", signal)

    # Select presets across quality tiers for experiment E
    # Good (<5% diff): presets that survive macro reduction perfectly
    # Warn (5-20%): noticeable but mild differences
    # Bad (>20%): significantly different
    test_presets = {
        # GOOD tier — these should sound identical
        "codec_32kbps": "good",
        "robot_voice": "good",
        "cathedral_decay": "good",
        "glitch_stutter": "good",

        # WARN tier — subtle differences
        "singing_filter": "warn",
        "notch_comb": "warn",
        "time_smear": "warn",
        "reverse_shadow": "warn",

        # BAD tier — clearly different
        "chipmunk_machine": "bad",
        "degraded_stutter_loop": "bad",
        "nils_frahm": "bad",
        "slow_dissolve": "bad",
    }

    # Load presets
    loaded = {}
    for name in test_presets:
        path = PRESET_DIR / f"{name}.json"
        if path.exists():
            with open(path) as f:
                loaded[name] = json.load(f)
        else:
            print(f"  SKIP: {path} not found")

    # Render each preset with each input
    report_lines = []
    report_lines.append("# Lossy Macro Experiment E — Listening Test")
    report_lines.append("")
    report_lines.append("## What is this?")
    report_lines.append("")
    report_lines.append("Experiment E reduces the lossy plugin from 41 to 31 macro controls by:")
    report_lines.append("- **Fixing 8 params** at defaults: seed, pre_echo, transient_ratio,")
    report_lines.append("  noise_shape, slushy_rate, freeze_mode, verb_position, global_amount")
    report_lines.append("- **Deriving 2 params** from others:")
    report_lines.append("  - `decay = 0.5 + 0.517 * verb` (only when verb > 0)")
    report_lines.append("  - `filter_slope = round(1 - 2.6 * (filter_width - 0.5))`")
    report_lines.append("")
    report_lines.append("Each preset is rendered with 3 different inputs (drum loop, sustained")
    report_lines.append("chord, sparse percussive hits) so you can hear the effect on different")
    report_lines.append("material.")
    report_lines.append("")
    report_lines.append("## Dry inputs")
    report_lines.append("")
    report_lines.append("- `00_dry_drum.wav` — lo-fi drum loop (rhythmic, transient-heavy)")
    report_lines.append("- `00_dry_chord.wav` — sustained chord (tonal, steady-state)")
    report_lines.append("- `00_dry_hits.wav` — sparse percussive hits (silence + transients)")
    report_lines.append("")
    report_lines.append("## How to listen")
    report_lines.append("")
    report_lines.append("For each preset, A/B the `_original` vs `_macro_E` file:")
    report_lines.append("")
    report_lines.append("1. **Listen to the original first** to understand the effect character")
    report_lines.append("2. **Switch to the macro version** — does it sound the same?")
    report_lines.append("3. **Focus on:**")
    report_lines.append("   - Overall tonal balance (bright/dark)")
    report_lines.append("   - Reverb tail character (if the preset uses verb)")
    report_lines.append("   - Filter resonance shape")
    report_lines.append("   - Any artifacts that appear or disappear")
    report_lines.append("   - Dynamic behavior (gating, compression feel)")
    report_lines.append("")
    report_lines.append("## What to expect")
    report_lines.append("")
    report_lines.append("- **GOOD presets**: Should sound identical or nearly so. These don't")
    report_lines.append("  use any of the 8 fixed params, and their verb/decay and")
    report_lines.append("  filter_width/filter_slope relationships match the derivation.")
    report_lines.append("- **WARN presets**: Subtle differences. Usually a slightly different")
    report_lines.append("  reverb decay time or filter slope. You might need headphones to hear it.")
    report_lines.append("- **BAD presets**: Clearly different. These rely on params that the")
    report_lines.append("  macro system eliminates (e.g. noise_shape, independent decay values,")
    report_lines.append("  verb_position=1). The question is: is the macro version still")
    report_lines.append("  *musically useful* even if different?")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## Preset details")
    report_lines.append("")

    for preset_name, tier in test_presets.items():
        if preset_name not in loaded:
            continue

        preset = loaded[preset_name]
        recon = reconstruct_E(preset, defaults)

        # What changed?
        full_orig = {**defaults, **{k: v for k, v in preset.items() if k != "_meta"}}
        changes = []
        for k, v in FIXED.items():
            orig = full_orig.get(k, defaults[k])
            if orig != v:
                changes.append(f"{k}: {orig} → {v}")
        # Check derived params
        orig_decay = full_orig.get("decay", defaults["decay"])
        recon_decay = recon["decay"]
        if abs(float(orig_decay) - float(recon_decay)) > 0.01:
            changes.append(f"decay: {orig_decay} → {recon_decay:.3f} (derived from verb={full_orig.get('verb', 0.0)})")
        orig_slope = full_orig.get("filter_slope", defaults["filter_slope"])
        recon_slope = recon["filter_slope"]
        if orig_slope != recon_slope:
            changes.append(f"filter_slope: {orig_slope} → {recon_slope} (derived from filter_width={full_orig.get('filter_width', 0.5)})")

        report_lines.append(f"### {preset_name} [{tier.upper()}]")
        report_lines.append("")
        cat = preset.get("_meta", {}).get("category", "unknown") if isinstance(preset.get("_meta"), dict) else "unknown"
        desc = preset.get("_meta", {}).get("description", "") if isinstance(preset.get("_meta"), dict) else ""
        report_lines.append(f"Category: {cat}")
        if desc:
            report_lines.append(f"Description: {desc}")
        report_lines.append("")
        if changes:
            report_lines.append("**What changed in macro version:**")
            for c in changes:
                report_lines.append(f"- {c}")
            report_lines.append("")
            report_lines.append("**What to listen for:** " + (
                "These changes are minor — listen carefully for tonal or dynamic differences."
                if tier == "warn" else
                "These changes significantly alter the effect. Compare the overall character."
                if tier == "bad" else
                "Should sound identical — the changed params have negligible perceptual impact."
            ))
        else:
            report_lines.append("**No params changed** — original and macro versions are identical.")
        report_lines.append("")

        # Render with each input
        for input_name, signal in inputs.items():
            prefix = f"{tier}_{preset_name}_{input_name}"
            try:
                orig_clean = {k: v for k, v in full_orig.items()
                              if k != "_meta" and k != "tail_length"}
                recon_clean = {k: v for k, v in recon.items()
                               if k != "_meta" and k != "tail_length"}

                out_orig = render_lossy(signal, json.dumps(orig_clean))
                out_recon = render_lossy(signal, json.dumps(recon_clean))

                save_wav(OUT_DIR / f"{prefix}_original.wav", out_orig)
                save_wav(OUT_DIR / f"{prefix}_macro_E.wav", out_recon)

                # Compute diff stats
                ml = min(len(out_orig), len(out_recon))
                rms_orig = float(np.sqrt(np.mean(out_orig[:ml] ** 2)))
                diff_rms = float(np.sqrt(np.mean((out_orig[:ml] - out_recon[:ml]) ** 2)))
                rel = diff_rms / rms_orig if rms_orig > 1e-15 else 0.0
                print(f"  {prefix}: rel_diff={rel:.4f}")
            except Exception as e:
                print(f"  ERROR {prefix}: {e}")

        report_lines.append(f"Files: `{tier}_{preset_name}_*_original.wav` vs `*_macro_E.wav`")
        report_lines.append("")

    # Write report
    report_path = OUT_DIR / "LISTENING_TEST.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\nReport: {report_path}")
    print(f"Audio files: {OUT_DIR}/")
    print(f"Total files: {len(list(OUT_DIR.glob('*.wav')))}")


if __name__ == "__main__":
    main()
