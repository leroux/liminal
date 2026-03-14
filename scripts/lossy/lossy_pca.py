#!/usr/bin/env python3
"""Lossy parameter space analysis — full methodology from docs/pca_parameter_reduction.md.

Steps covered:
  1. Build feature matrix from presets (98 JSON files)
  2. Pre-transform skewed params (log filter_freq, log2 window_size)
  3. Standardize (zero mean, unit variance)
  4. Run PCA — scree, cumulative variance, loadings
  5. Interpret components (co-loading patterns)
  6. Parameter group stats (variance, uniformity)
  7. Random parameter sampling (3 groups: presets, random, combined)
  8. Category analysis (hand-designed vs ML-generated)
  9. Cluster analysis (k-means on PCA space, silhouette)
 10. Dead parameter detection:
     a. Variance analysis (near-zero variance features)
     b. Loading analysis (no significant loading on top PCs)
     c. DSP sensitivity analysis — sweep each param through Rust DSP, measure output change
 11. Audio validation — render original vs reconstructed through Rust DSP,
     compare spectral distance, energy envelope correlation

Requires: lossy_rust PyO3 bindings (build with: cd rust && uv run maturin develop -m crates/lossy-python/Cargo.toml --release)

Throwaway analysis script — not part of the production codebase.
"""

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

PRESET_DIR = Path(__file__).parent.parent.parent / "lossy" / "gui" / "presets"
OUT_DIR = Path(__file__).parent / "pca_output"
SR = 44100.0

# ── Helpers ──────────────────────────────────────────────────────────────────


def get_defaults() -> dict:
    """Default LossyParams matching Rust."""
    return {
        "inverse": 0,
        "jitter": 0.0,
        "loss": 0.5,
        "window_size": 2048,
        "hop_divisor": 4,
        "n_bands": 21,
        "global_amount": 1.0,
        "phase_loss": 0.0,
        "quantizer": 0,
        "pre_echo": 0.0,
        "noise_shape": 0.0,
        "weighting": 1.0,
        "hf_threshold": 0.3,
        "transient_ratio": 4.0,
        "slushy_rate": 0.03,
        "crush": 0.0,
        "decimate": 0.0,
        "packets": 0,
        "packet_rate": 0.3,
        "packet_size": 30.0,
        "filter_type": 0,
        "filter_freq": 1000.0,
        "filter_width": 0.5,
        "filter_slope": 1,
        "verb": 0.0,
        "decay": 0.5,
        "verb_position": 0,
        "freeze": 0,
        "freeze_mode": 0,
        "freezer": 1.0,
        "gate": 0.0,
        "threshold": 0.5,
        "auto_gain": 0.0,
        "loss_gain": 0.5,
        "bounce": 0,
        "bounce_target": 0,
        "bounce_rate": 0.3,
        "bounce_lfo_min": 0.1,
        "bounce_lfo_max": 5.0,
        "wet_dry": 1.0,
        "seed": 42,
    }


# Parameter ranges for random sampling (matching Rust param_range + integer constraints)
PARAM_RANGES = {
    "loss": (0.0, 1.0),
    "jitter": (0.0, 1.0),
    "global_amount": (0.0, 1.0),
    "phase_loss": (0.0, 1.0),
    "pre_echo": (0.0, 1.0),
    "noise_shape": (0.0, 1.0),
    "weighting": (0.0, 1.0),
    "hf_threshold": (0.0, 1.0),
    "transient_ratio": (1.5, 20.0),
    "slushy_rate": (0.001, 0.5),
    "crush": (0.0, 1.0),
    "decimate": (0.0, 1.0),
    "packet_rate": (0.0, 1.0),
    "packet_size": (5.0, 200.0),
    "filter_freq": (20.0, 20000.0),
    "filter_width": (0.0, 1.0),
    "verb": (0.0, 1.0),
    "decay": (0.0, 1.0),
    "freezer": (0.0, 1.0),
    "gate": (0.0, 1.0),
    "threshold": (0.0, 1.0),
    "auto_gain": (0.0, 1.0),
    "loss_gain": (0.0, 1.0),
    "bounce_rate": (0.0, 1.0),
    "bounce_lfo_min": (0.01, 50.0),
    "bounce_lfo_max": (0.01, 50.0),
    "wet_dry": (0.0, 1.0),
}


def load_presets() -> list[tuple[str, dict]]:
    """Load all preset JSON files. Returns (name, params_dict)."""
    presets = []
    for p in sorted(PRESET_DIR.glob("*.json")):
        if p.name == "favorites.json":
            continue
        with open(p) as f:
            raw = json.load(f)
        presets.append((p.stem, raw))
    print(f"Loaded {len(presets)} presets")
    return presets


def classify_preset(name: str, preset: dict) -> str:
    """Classify preset by category from _meta."""
    meta = preset.get("_meta")
    if meta and isinstance(meta, dict):
        cat = meta.get("category", "")
        if cat == "ML Generated":
            return "ml-generated"
        if cat:
            return cat.lower().replace(" ", "-")
    if name.startswith("z_gen_"):
        return "ml-generated"
    return "hand-designed"


# ── Step 1+2: Feature extraction with transforms ────────────────────────────

# Features: all params except seed and _meta.
# Log-transform: filter_freq (octave perception), window_size (powers of 2)
# All others: leave as numeric (integers included as ordinal)

FEATURE_KEYS = [
    # Spectral loss (continuous)
    "loss", "jitter", "global_amount", "phase_loss", "pre_echo",
    "noise_shape", "weighting", "hf_threshold", "transient_ratio", "slushy_rate",
    # Lo-fi
    "crush", "decimate",
    # Packets
    "packet_rate", "packet_size",
    # Filter
    "filter_width",
    # Reverb
    "verb", "decay",
    # Freeze
    "freezer",
    # Gate/dynamics
    "gate", "threshold", "auto_gain", "loss_gain",
    # Bounce
    "bounce_rate", "bounce_lfo_min", "bounce_lfo_max",
    # Mix
    "wet_dry",
]

# Integer params treated as ordinal/binary
INT_KEYS = [
    "inverse", "quantizer",
    "hop_divisor", "n_bands",
    "packets", "filter_type", "filter_slope",
    "verb_position", "freeze", "freeze_mode",
    "bounce", "bounce_target",
]

# Log-transformed
LOG_KEYS = [
    ("filter_freq", "log_filter_freq"),
    ("window_size", "log2_window_size"),
]


def build_feature_names() -> list[str]:
    names = list(FEATURE_KEYS)
    names.extend(INT_KEYS)
    names.extend([log_name for _, log_name in LOG_KEYS])
    return names


def preset_to_features(preset: dict, defaults: dict) -> np.ndarray:
    """Convert preset to feature vector (Step 1+2)."""
    features = []

    # Continuous params (direct)
    for key in FEATURE_KEYS:
        features.append(float(preset.get(key, defaults[key])))

    # Integer params (ordinal)
    for key in INT_KEYS:
        features.append(float(preset.get(key, defaults[key])))

    # Log-transformed params
    for raw_key, _ in LOG_KEYS:
        val = float(preset.get(raw_key, defaults[raw_key]))
        if raw_key == "window_size":
            features.append(math.log2(max(val, 1)))
        else:
            features.append(math.log(max(val, 0.01)))

    return np.array(features, dtype=np.float64)


# ── Step 3+4: PCA ───────────────────────────────────────────────────────────


def run_pca(X: np.ndarray, labels: list[str], feature_names: list[str],
            title: str, save_plot: bool = True) -> tuple[PCA, np.ndarray]:
    """Standardize, run PCA, print results, plot. Returns (pca, X_pca)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    cumvar = np.cumsum(pca.explained_variance_ratio_)

    print(f"\n{'=' * 80}")
    print(f"PCA: {title}")
    print(f"{'=' * 80}")
    print(f"Features: {X.shape[1]}  Samples: {X.shape[0]}")

    # Step 4: cumulative variance thresholds
    for t in [0.80, 0.90, 0.95, 0.99]:
        n = int(np.argmax(cumvar >= t) + 1)
        print(f"  {t*100:.0f}%: {n} PCs")

    print(f"\nPer-component (first 15):")
    for i in range(min(15, len(pca.explained_variance_ratio_))):
        print(f"  PC{i+1}: {pca.explained_variance_ratio_[i]:.4f} "
              f"(cum: {cumvar[i]:.4f})")

    # Step 5: top loadings per PC
    print(f"\nLoadings (first 10 PCs):")
    for pc in range(min(10, pca.n_components_)):
        loadings = pca.components_[pc]
        top = np.argsort(np.abs(loadings))[::-1]
        print(f"\n  PC{pc+1} ({pca.explained_variance_ratio_[pc]:.3f}):")
        for j in range(min(6, len(top))):
            idx = top[j]
            print(f"    {feature_names[idx]:25s} {loadings[idx]:+.3f}")

    # Step 9: cluster analysis
    n95 = int(np.argmax(cumvar >= 0.95) + 1)
    n_use = min(n95, X_pca.shape[1])
    X_reduced = X_pca[:, :n_use]

    if X.shape[0] >= 6:
        best_k, best_score = 2, -1
        k_range = range(2, min(8, X.shape[0] // 2))
        for k in k_range:
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            sl = km.fit_predict(X_reduced)
            score = silhouette_score(X_reduced, sl)
            if score > best_score:
                best_score = score
                best_k = k

        km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
        cluster_labels = km.fit_predict(X_reduced)

        print(f"\nClusters (k={best_k}, silhouette={best_score:.3f}):")
        for c in range(best_k):
            members = [labels[i] for i in range(len(labels)) if cluster_labels[i] == c]
            if len(members) <= 20:
                print(f"  {c}: {', '.join(members)}")
            else:
                print(f"  {c}: ({len(members)} members) {', '.join(members[:10])}...")
    else:
        cluster_labels = np.zeros(X.shape[0], dtype=int)

    # Plot
    if save_plot:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(title, fontsize=14)

        ax = axes[0]
        n_show = min(25, len(pca.explained_variance_ratio_))
        ax.bar(range(1, n_show + 1), pca.explained_variance_ratio_[:n_show], alpha=0.7)
        ax.plot(range(1, n_show + 1), cumvar[:n_show], "ro-", markersize=4)
        ax.axhline(y=0.90, color="g", linestyle="--", alpha=0.5, label="90%")
        ax.axhline(y=0.95, color="orange", linestyle="--", alpha=0.5, label="95%")
        ax.set_xlabel("PC")
        ax.set_ylabel("Explained Variance")
        ax.set_title("Scree Plot")
        ax.legend(fontsize=8)

        ax = axes[1]
        ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, s=30)
        if len(labels) <= 100:
            for i, lb in enumerate(labels):
                ax.annotate(lb, (X_pca[i, 0], X_pca[i, 1]), fontsize=3, alpha=0.5, rotation=25)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.set_title("PC1 vs PC2")

        ax = axes[2]
        sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap="tab10", alpha=0.7, s=30)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"Clusters (k={len(np.unique(cluster_labels))})")
        plt.colorbar(sc, ax=ax)

        plt.tight_layout()
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUT_DIR / f"pca_{title.lower().replace(' ', '_').replace('+', '_')}.png"
        plt.savefig(path, dpi=150)
        print(f"\n  Plot: {path}")
        plt.close()

    return pca, X_pca


# ── Step 6: Parameter group stats ──────────────────────────────────────────


def param_usage_stats(presets: list[tuple[str, dict]], defaults: dict) -> None:
    """Per-parameter variance and usage statistics (Step 6)."""
    print(f"\n{'=' * 80}")
    print("PARAMETER USAGE STATS (Step 6)")
    print(f"{'=' * 80}")

    all_keys = list(FEATURE_KEYS) + INT_KEYS + [k for k, _ in LOG_KEYS]

    stats = []
    for key in sorted(set(defaults.keys()) - {"_meta", "meta", "seed"}):
        values = []
        non_default = 0
        for _, preset in presets:
            val = preset.get(key, defaults[key])
            values.append(float(val))
            if val != defaults[key]:
                non_default += 1
        values = np.array(values)
        stats.append({
            "key": key,
            "non_default": non_default,
            "pct": non_default / len(presets) * 100,
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "cv": np.std(values) / np.mean(values) if np.mean(values) > 1e-10 else 0.0,
        })

    # Sort by usage
    stats.sort(key=lambda s: s["non_default"], reverse=True)

    print(f"\n  {'Parameter':<20} {'Used':>5} {'Pct':>5} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'CV':>6}")
    print(f"  {'-'*20} {'-'*5} {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
    for s in stats:
        print(f"  {s['key']:<20} {s['non_default']:>5} {s['pct']:>4.0f}% "
              f"{s['mean']:>8.3f} {s['std']:>8.3f} {s['min']:>8.3f} {s['max']:>8.3f} {s['cv']:>6.3f}")


# ── Co-variance analysis ──────────────────────────────────────────────────


def covariance_analysis(X: np.ndarray, feature_names: list[str]) -> None:
    """Find highly correlated parameter pairs (potential merge candidates)."""
    print(f"\n{'=' * 80}")
    print("CO-VARIANCE ANALYSIS (potential merge candidates)")
    print(f"{'=' * 80}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    corr = np.corrcoef(X_scaled.T)

    # Find pairs with |correlation| > 0.5
    pairs = []
    n = len(feature_names)
    for i in range(n):
        for j in range(i + 1, n):
            r = corr[i, j]
            if abs(r) > 0.5 and not np.isnan(r):
                pairs.append((feature_names[i], feature_names[j], r))

    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    print(f"\n  Correlated pairs (|r| > 0.5):")
    for a, b, r in pairs[:30]:
        print(f"    {a:25s} ↔ {b:25s}  r={r:+.3f}")

    if not pairs:
        print("  No highly correlated pairs found.")


# ── Step 7: Random sampling ─────────────────────────────────────────────────


def generate_random_params(n: int, rng: np.random.Generator) -> list[dict]:
    """Generate random LossyParams by uniform sampling (Step 7)."""
    defaults = get_defaults()
    results = []
    for _ in range(n):
        p = defaults.copy()
        # Continuous params
        for key, (lo, hi) in PARAM_RANGES.items():
            p[key] = float(rng.uniform(lo, hi))
        # Integer params
        p["inverse"] = int(rng.choice([0, 1]))
        p["quantizer"] = int(rng.choice([0, 1]))
        p["window_size"] = int(rng.choice([256, 512, 1024, 2048, 4096, 8192, 16384]))
        p["hop_divisor"] = int(rng.choice([1, 2, 4, 8]))
        p["n_bands"] = int(rng.integers(2, 65))
        p["packets"] = int(rng.choice([0, 1, 2]))
        p["filter_type"] = int(rng.choice([0, 1, 2]))
        p["filter_slope"] = int(rng.choice([0, 1, 2]))
        p["verb_position"] = int(rng.choice([0, 1]))
        p["freeze"] = int(rng.choice([0, 1]))
        p["freeze_mode"] = int(rng.choice([0, 1]))
        p["bounce"] = int(rng.choice([0, 1]))
        p["bounce_target"] = int(rng.integers(0, 7))
        results.append(p)
    return results


# ── Step 10: Dead parameter detection ───────────────────────────────────────


def dead_param_variance(X: np.ndarray, feature_names: list[str]) -> list[str]:
    """Find near-zero variance features (Step 10a)."""
    variances = np.var(X, axis=0)
    dead = []
    print(f"\n{'=' * 80}")
    print("DEAD PARAMETERS: Variance Analysis (Step 10a)")
    print(f"{'=' * 80}")
    for i, (name, var) in enumerate(zip(feature_names, variances)):
        if var < 1e-6:
            dead.append(name)
            print(f"  DEAD (zero var):  {name}  var={var:.2e}")
    if not dead:
        print("  No zero-variance features found.")

    # Show lowest-variance features
    sorted_idx = np.argsort(variances)
    print(f"\n  Lowest variance features:")
    for i in sorted_idx[:15]:
        print(f"    {feature_names[i]:25s}  var={variances[i]:.4e}")
    return dead


def dead_param_loading(pca: PCA, feature_names: list[str], threshold: float = 0.95) -> list[str]:
    """Find features with no significant loading on top PCs (Step 10b)."""
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_top = int(np.argmax(cumvar >= threshold) + 1)

    max_loading = np.max(np.abs(pca.components_[:n_top]), axis=0)

    dead = []
    print(f"\n{'=' * 80}")
    print(f"DEAD PARAMETERS: Loading Analysis (Step 10b, top {n_top} PCs for {threshold:.0%})")
    print(f"{'=' * 80}")

    sorted_idx = np.argsort(max_loading)
    print(f"  Weakest loadings:")
    for i in sorted_idx[:20]:
        tag = " <-- DEAD" if max_loading[i] < 0.15 else ""
        print(f"    {feature_names[i]:25s}  max|loading|={max_loading[i]:.3f}{tag}")
        if max_loading[i] < 0.15:
            dead.append(feature_names[i])
    return dead


def dead_param_dsp_sensitivity(presets: list[tuple[str, dict]], defaults: dict) -> dict:
    """Sweep each param, render through Rust DSP, measure output change (Step 10c).

    Requires lossy_rust PyO3 bindings.
    """
    try:
        from lossy_rust import render_lossy
    except ImportError:
        print("\n  SKIP: lossy_rust not available (build with: cd rust && uv run maturin develop -m crates/lossy-python/Cargo.toml --release)")
        return {}

    print(f"\n{'=' * 80}")
    print("DEAD PARAMETERS: DSP Sensitivity Analysis (Step 10c)")
    print(f"{'=' * 80}")

    # Use a signal with BOTH broadband content and transients
    # (lossy is nonlinear and input-dependent; white noise alone misses
    # pre_echo/transient_ratio which only fire on energy jumps)
    rng = np.random.default_rng(42)
    n_samples = int(SR * 2.0)
    test_signal = np.zeros(n_samples)
    # Broadband noise segments separated by silence (creates transients)
    for t_sec in [0.1, 0.4, 0.7, 1.0, 1.3, 1.6]:
        start = int(t_sec * SR)
        end = min(start + int(0.15 * SR), n_samples)
        test_signal[start:end] = rng.normal(0, 0.4, end - start)

    # Pick ~5 representative presets (skip ML-generated if any)
    test_presets = []
    for name, preset in presets:
        cat = classify_preset(name, preset)
        if cat != "ml-generated":
            test_presets.append((name, preset))
    step = max(1, len(test_presets) // 5)
    test_presets = test_presets[::step][:5]
    print(f"  Test presets: {[n for n, _ in test_presets]}")

    # Parameters to sweep
    sweep_params = {}
    for key, (lo, hi) in PARAM_RANGES.items():
        sweep_params[key] = np.linspace(lo, hi, 5)
    # Integer params
    sweep_params["inverse"] = [0, 1]
    sweep_params["quantizer"] = [0, 1]
    sweep_params["window_size"] = [256, 1024, 2048, 4096, 16384]
    sweep_params["hop_divisor"] = [1, 2, 4, 8]
    sweep_params["n_bands"] = [2, 10, 21, 40, 64]
    sweep_params["packets"] = [0, 1, 2]
    sweep_params["filter_type"] = [0, 1, 2]
    sweep_params["filter_slope"] = [0, 1, 2]
    sweep_params["verb_position"] = [0, 1]
    sweep_params["freeze"] = [0, 1]
    sweep_params["freeze_mode"] = [0, 1]
    sweep_params["bounce"] = [0, 1]
    sweep_params["bounce_target"] = [0, 1, 2, 3, 4, 5, 6]

    sensitivity = {}

    for param_name, sweep_values in sweep_params.items():
        rms_changes = []

        for preset_name, preset in test_presets:
            base_params = {**defaults, **{k: v for k, v in preset.items() if k != "_meta"}}
            base_json = json.dumps(base_params)

            try:
                out_base = render_lossy(test_signal, base_json)
                base_rms = float(np.sqrt(np.mean(out_base ** 2)))
            except Exception:
                continue

            if base_rms < 1e-10:
                continue

            max_change = 0.0
            for val in sweep_values:
                swept = base_params.copy()
                swept[param_name] = int(val) if isinstance(defaults[param_name], int) else float(val)
                try:
                    out_swept = render_lossy(test_signal, json.dumps(swept))
                    min_len = min(len(out_base), len(out_swept))
                    diff_rms = float(np.sqrt(np.mean((out_base[:min_len] - out_swept[:min_len]) ** 2)))
                    change = diff_rms / base_rms
                    max_change = max(max_change, change)
                except Exception:
                    pass

            rms_changes.append(max_change)

        mean_change = float(np.mean(rms_changes)) if rms_changes else 0.0
        sensitivity[param_name] = mean_change

    # Sort by sensitivity (lowest = most dead)
    sorted_params = sorted(sensitivity.items(), key=lambda x: x[1])
    print(f"\n  Parameter sensitivity (max RMS change across sweep / base RMS):")
    for param, change in sorted_params:
        tag = " <-- LOW" if change < 0.05 else ""
        print(f"    {param:20s}  {change:.4f}{tag}")

    return sensitivity


# ── Step 10: Perceptual-space PCA ────────────────────────────────────────────


def extract_audio_features(signal: np.ndarray) -> np.ndarray:
    """Extract perceptual features from rendered audio.

    Returns a feature vector with: spectral centroid, spectral bandwidth,
    spectral flatness, spectral rolloff, RMS level, crest factor,
    zero crossing rate, and 13 MFCCs (mean over frames).
    """
    # Global stats
    rms = float(np.sqrt(np.mean(signal ** 2)))
    peak = float(np.max(np.abs(signal)))
    crest = peak / rms if rms > 1e-15 else 0.0

    # Frame-based spectral analysis
    frame_size = 2048
    hop = 512
    n_frames = max(1, (len(signal) - frame_size) // hop)

    centroids = []
    bandwidths = []
    flatnesses = []
    rolloffs = []
    zcrs = []
    mfcc_accum = np.zeros(13)

    freqs = np.fft.rfftfreq(frame_size, d=1.0 / SR)
    # Mel filterbank (simplified: 13 triangular filters from 20Hz to 8000Hz)
    mel_lo = 2595 * np.log10(1 + 20 / 700)
    mel_hi = 2595 * np.log10(1 + 8000 / 700)
    mel_centers = np.linspace(mel_lo, mel_hi, 15)
    hz_centers = 700 * (10 ** (mel_centers / 2595) - 1)

    for i in range(n_frames):
        start = i * hop
        frame = signal[start:start + frame_size]
        if len(frame) < frame_size:
            break

        # Windowed spectrum
        windowed = frame * np.hanning(frame_size)
        spectrum = np.abs(np.fft.rfft(windowed))
        power = spectrum ** 2
        total_power = np.sum(power)

        if total_power < 1e-30:
            continue

        # Spectral centroid
        centroid = float(np.sum(freqs * power) / total_power)
        centroids.append(centroid)

        # Spectral bandwidth
        bw = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * power) / total_power))
        bandwidths.append(bw)

        # Spectral flatness (geometric mean / arithmetic mean of power)
        log_power = np.log(power + 1e-30)
        geo_mean = np.exp(np.mean(log_power))
        arith_mean = np.mean(power)
        flatness = geo_mean / arith_mean if arith_mean > 1e-30 else 0.0
        flatnesses.append(flatness)

        # Spectral rolloff (freq below which 85% of energy)
        cumsum = np.cumsum(power)
        rolloff_idx = np.searchsorted(cumsum, 0.85 * total_power)
        rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])
        rolloffs.append(rolloff)

        # ZCR
        zcr = float(np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame)))
        zcrs.append(zcr)

        # Simplified MFCCs (13 mel bands)
        mel_energies = np.zeros(13)
        for j in range(13):
            lo_hz, mid_hz, hi_hz = hz_centers[j], hz_centers[j + 1], hz_centers[j + 2]
            lo_bin = np.searchsorted(freqs, lo_hz)
            mid_bin = np.searchsorted(freqs, mid_hz)
            hi_bin = np.searchsorted(freqs, hi_hz)
            # Triangular filter
            for b in range(lo_bin, min(hi_bin, len(power))):
                if b < mid_bin:
                    w = (freqs[b] - lo_hz) / max(mid_hz - lo_hz, 1.0)
                else:
                    w = (hi_hz - freqs[b]) / max(hi_hz - mid_hz, 1.0)
                mel_energies[j] += max(w, 0) * power[b]
        mel_energies = np.log(mel_energies + 1e-30)
        # DCT (type II) to get cepstral coefficients
        for j in range(13):
            mfcc_accum[j] += np.sum(mel_energies * np.cos(np.pi * j * (np.arange(13) + 0.5) / 13))

    n_valid = max(1, len(centroids))
    mfcc_accum /= n_valid

    features = [
        rms,
        crest,
        np.mean(centroids) if centroids else 0.0,
        np.std(centroids) if len(centroids) > 1 else 0.0,
        np.mean(bandwidths) if bandwidths else 0.0,
        np.mean(flatnesses) if flatnesses else 0.0,
        np.mean(rolloffs) if rolloffs else 0.0,
        np.mean(zcrs) if zcrs else 0.0,
    ]
    features.extend(mfcc_accum.tolist())
    return np.array(features, dtype=np.float64)


PERCEPTUAL_FEATURE_NAMES = [
    "rms", "crest_factor",
    "spectral_centroid_mean", "spectral_centroid_std",
    "spectral_bandwidth", "spectral_flatness",
    "spectral_rolloff", "zcr",
] + [f"mfcc_{i}" for i in range(13)]


def perceptual_space_pca(presets: list[tuple[str, dict]], defaults: dict) -> None:
    """Render all presets through DSP, extract audio features, run PCA (Step 10)."""
    try:
        from lossy_rust import render_lossy
    except ImportError:
        print("\n  SKIP perceptual PCA: lossy_rust not available")
        return

    print(f"\n{'=' * 80}")
    print("PERCEPTUAL-SPACE PCA (Step 10)")
    print(f"{'=' * 80}")

    # Test signal: noise bursts with transients (nonlinear effect needs program material)
    rng = np.random.default_rng(42)
    n_samples = int(SR * 2.0)
    test_signal = np.zeros(n_samples)
    for t_sec in [0.1, 0.4, 0.7, 1.0, 1.3, 1.6]:
        start = int(t_sec * SR)
        end = min(start + int(0.15 * SR), n_samples)
        test_signal[start:end] = rng.normal(0, 0.4, end - start)

    X_perceptual = []
    names = []
    for name, preset in presets:
        params = {**defaults, **{k: v for k, v in preset.items() if k != "_meta"}}
        try:
            output = render_lossy(test_signal, json.dumps(params))
            features = extract_audio_features(output)
            X_perceptual.append(features)
            names.append(name)
        except Exception as e:
            print(f"  SKIP {name}: {e}")

    if len(X_perceptual) < 5:
        print("  Too few successful renders for PCA")
        return

    X_perceptual = np.array(X_perceptual)
    print(f"  Rendered {len(names)} presets, {X_perceptual.shape[1]} audio features")

    pca_perc, _ = run_pca(X_perceptual, names, PERCEPTUAL_FEATURE_NAMES,
                          "Perceptual Space")

    # Compare dimensionality
    cumvar_perc = np.cumsum(pca_perc.explained_variance_ratio_)
    n90_perc = int(np.argmax(cumvar_perc >= 0.90) + 1)
    n95_perc = int(np.argmax(cumvar_perc >= 0.95) + 1)
    print(f"\n  Perceptual space: 90%={n90_perc} PCs, 95%={n95_perc} PCs")
    print(f"  (vs parameter space: 90%=22 PCs, 95%=27 PCs)")
    if n90_perc < 22:
        print(f"  >> Perceptual space needs FEWER PCs — some parameter variation is inaudible")
    elif n90_perc > 22:
        print(f"  >> Perceptual space needs MORE PCs — nonlinear DSP creates perceptual distinctions not visible in param space")


# ── Linear fits for co-varying params ─────────────────────────────────────────


def compute_anchored_fits(presets: list[tuple[str, dict]], defaults: dict) -> dict:
    """Compute anchored linear fits for co-varying param pairs.

    Anchored fit: line passes through (default_master, default_derived).
    Slope estimated from presets where master is non-default.
    Formula: derived = default_derived + slope * (master - default_master)

    Returns dict: (master, derived) -> (slope, r, n_samples).
    """
    param_values = {}
    for key in defaults:
        if key in ("_meta", "meta", "seed"):
            continue
        param_values[key] = np.array([
            float(preset.get(key, defaults[key])) for _, preset in presets
        ])

    pairs = [
        ("verb", "decay"),
        ("crush", "decimate"),
        ("filter_width", "filter_slope"),
        ("packet_rate", "packet_size"),
        ("auto_gain", "threshold"),
        ("auto_gain", "loss_gain"),
        ("quantizer", "noise_shape"),
        ("freeze", "freezer"),
        ("jitter", "phase_loss"),
    ]

    fits = {}
    print(f"\n{'=' * 80}")
    print("ANCHORED FITS for co-varying param pairs")
    print(f"  Formula: derived = default + slope * (master - master_default)")
    print(f"{'=' * 80}")
    print(f"  {'Master':<20} {'Derived':<20} {'Slope':>10} {'r':>8} {'n_active':>8}")
    print(f"  {'-'*20} {'-'*20} {'-'*10} {'-'*8} {'-'*8}")

    for master, derived in pairs:
        x = param_values[master]
        y = param_values[derived]
        x0 = float(defaults[master])
        y0 = float(defaults[derived])

        # Fit only on presets where master is non-default
        mask = np.abs(x - x0) > 1e-6
        n_active = int(mask.sum())

        if n_active < 3:
            fits[(master, derived)] = (0.0, 0.0, n_active)
            print(f"  {master:<20} {derived:<20} {'0.000':>10} {'N/A':>8} {n_active:>8}")
            continue

        # Anchored slope: minimize sum((y - y0 - slope*(x - x0))^2) for active presets
        dx = x[mask] - x0
        dy = y[mask] - y0
        denom = float(np.sum(dx ** 2))
        slope = float(np.sum(dx * dy) / denom) if denom > 1e-10 else 0.0

        # Correlation on active presets
        r = float(np.corrcoef(x[mask], y[mask])[0, 1]) if n_active > 2 else 0.0

        fits[(master, derived)] = (slope, r, n_active)
        print(f"  {master:<20} {derived:<20} {slope:>10.3f} {r:>+8.3f} {n_active:>8}")

    return fits


# ── Experiment configs ────────────────────────────────────────────────────────

# Subsystem gates: which params gate which subsystems.
# Derived params only activate when the gate param is non-default.
SUBSYSTEM_GATES = {
    "decay": "verb",         # decay only matters when verb > 0
    "decimate": "crush",     # decimate and crush are independent but co-vary
    "filter_slope": None,    # always active (filter is always on)
    "packet_size": "packets",  # only matters when packets > 0
    "threshold": None,       # always active
    "loss_gain": None,       # always active
    "freezer": "freeze",     # only matters when freeze > 0
}


def build_experiments(fits: dict, defaults: dict) -> dict:
    """Build experiment configurations for macro roundtrip testing.

    Each experiment defines:
      fixed:   param -> fixed_value  (eliminated, set to constant)
      derived: param -> (master_param, slope)  (anchored at defaults)
      gate:    derived_param -> gate_param  (only derive when gate is non-default)
    """
    def fit(master, derived):
        slope, _r, _n = fits.get((master, derived), (0.0, 0.0, 0))
        return (master, slope)

    common_gate = {k: v for k, v in SUBSYSTEM_GATES.items()}

    return {
        "A_fixed_only": {
            "desc": "Fix 5 truly dead params, no merges",
            "fixed": {
                "seed": 42,
                "pre_echo": 0.0,
                "transient_ratio": 4.0,
                "noise_shape": 0.0,
                "slushy_rate": 0.03,
            },
            "derived": {},
            "gate": {},
        },
        "B_anchored_moderate": {
            "desc": "Fix 8 + 4 anchored merges",
            "fixed": {
                "seed": 42,
                "pre_echo": 0.0,
                "transient_ratio": 4.0,
                "noise_shape": 0.0,
                "slushy_rate": 0.03,
                "freeze_mode": 0,
                "verb_position": 0,
                "global_amount": 1.0,
            },
            "derived": {
                "decay": fit("verb", "decay"),
                "decimate": fit("crush", "decimate"),
                "filter_slope": fit("filter_width", "filter_slope"),
                "packet_size": fit("packet_rate", "packet_size"),
            },
            "gate": {k: common_gate[k] for k in
                     ["decay", "decimate", "filter_slope", "packet_size"]},
        },
        "C_anchored_aggressive": {
            "desc": "Fix 10 + 7 anchored merges",
            "fixed": {
                "seed": 42,
                "pre_echo": 0.0,
                "transient_ratio": 4.0,
                "noise_shape": 0.0,
                "slushy_rate": 0.03,
                "freeze_mode": 0,
                "verb_position": 0,
                "global_amount": 1.0,
                "bounce_lfo_min": 0.1,
                "bounce_lfo_max": 5.0,
            },
            "derived": {
                "decay": fit("verb", "decay"),
                "decimate": fit("crush", "decimate"),
                "filter_slope": fit("filter_width", "filter_slope"),
                "packet_size": fit("packet_rate", "packet_size"),
                "threshold": fit("auto_gain", "threshold"),
                "loss_gain": fit("auto_gain", "loss_gain"),
                "freezer": fit("freeze", "freezer"),
            },
            "gate": {k: common_gate[k] for k in
                     ["decay", "decimate", "filter_slope", "packet_size",
                      "threshold", "loss_gain", "freezer"]},
        },
        "D_anchored_max": {
            "desc": "Fix 14 + 7 anchored merges (target ~20)",
            "fixed": {
                "seed": 42,
                "pre_echo": 0.0,
                "transient_ratio": 4.0,
                "noise_shape": 0.0,
                "slushy_rate": 0.03,
                "freeze_mode": 0,
                "verb_position": 0,
                "global_amount": 1.0,
                "bounce_lfo_min": 0.1,
                "bounce_lfo_max": 5.0,
                "hop_divisor": 4,
                "n_bands": 21,
                "inverse": 0,
                "quantizer": 0,
            },
            "derived": {
                "decay": fit("verb", "decay"),
                "decimate": fit("crush", "decimate"),
                "filter_slope": fit("filter_width", "filter_slope"),
                "packet_size": fit("packet_rate", "packet_size"),
                "threshold": fit("auto_gain", "threshold"),
                "loss_gain": fit("auto_gain", "loss_gain"),
                "freezer": fit("freeze", "freezer"),
            },
            "gate": {k: common_gate[k] for k in
                     ["decay", "decimate", "filter_slope", "packet_size",
                      "threshold", "loss_gain", "freezer"]},
        },
        # --- Extra experiments: find the sweet spot ---
        "E_best_merges_only": {
            "desc": "Fix 8 + only 2 best merges (verb→decay r=0.83, width→slope r=0.86)",
            "fixed": {
                "seed": 42,
                "pre_echo": 0.0,
                "transient_ratio": 4.0,
                "noise_shape": 0.0,
                "slushy_rate": 0.03,
                "freeze_mode": 0,
                "verb_position": 0,
                "global_amount": 1.0,
            },
            "derived": {
                "decay": fit("verb", "decay"),
                "filter_slope": fit("filter_width", "filter_slope"),
            },
            "gate": {"decay": "verb", "filter_slope": None},
        },
        "F_sweet_spot": {
            "desc": "Fix 12 + 2 best merges (~27 macros)",
            "fixed": {
                "seed": 42,
                "pre_echo": 0.0,
                "transient_ratio": 4.0,
                "noise_shape": 0.0,
                "slushy_rate": 0.03,
                "freeze_mode": 0,
                "verb_position": 0,
                "global_amount": 1.0,
                "bounce_lfo_min": 0.1,
                "bounce_lfo_max": 5.0,
                "threshold": 0.5,
                "freezer": 1.0,
            },
            "derived": {
                "decay": fit("verb", "decay"),
                "filter_slope": fit("filter_width", "filter_slope"),
            },
            "gate": {"decay": "verb", "filter_slope": None},
        },
        "G_target_22": {
            "desc": "Fix 12 + 5 merges (~24 → group into 22 logical)",
            "fixed": {
                "seed": 42,
                "pre_echo": 0.0,
                "transient_ratio": 4.0,
                "noise_shape": 0.0,
                "slushy_rate": 0.03,
                "freeze_mode": 0,
                "verb_position": 0,
                "global_amount": 1.0,
                "bounce_lfo_min": 0.1,
                "bounce_lfo_max": 5.0,
                "threshold": 0.5,
                "freezer": 1.0,
            },
            "derived": {
                "decay": fit("verb", "decay"),
                "filter_slope": fit("filter_width", "filter_slope"),
                "decimate": fit("crush", "decimate"),
                "loss_gain": fit("auto_gain", "loss_gain"),
                "packet_size": fit("packet_rate", "packet_size"),
            },
            "gate": {"decay": "verb", "filter_slope": None,
                     "decimate": "crush", "loss_gain": None,
                     "packet_size": "packets"},
        },
    }


# ── Preview: reconstruction quality ──────────────────────────────────────────


def simulate_experiment_roundtrip(preset: dict, defaults: dict, config: dict) -> dict:
    """Simulate macro roundtrip for an experiment config.

    Uses anchored derivation: derived = default + slope * (master - master_default).
    Conditional: if a gate param exists and is at default, derived stays at default.
    """
    full = {**defaults, **{k: v for k, v in preset.items() if k != "_meta"}}
    reconstructed = full.copy()

    # Fix dead params at constant values
    for key, val in config["fixed"].items():
        reconstructed[key] = val

    # Derive merged params using anchored fit (conditional on gate)
    for derived_param, (master_param, slope) in config["derived"].items():
        master_val = float(full.get(master_param, defaults[master_param]))
        master_default = float(defaults[master_param])

        # Check gate: if gate param is at default, derived stays at default
        gate_param = config.get("gate", {}).get(derived_param)
        if gate_param is not None:
            gate_val = full.get(gate_param, defaults[gate_param])
            if gate_val == defaults[gate_param]:
                reconstructed[derived_param] = defaults[derived_param]
                continue

        # Anchored derivation: derived = default + slope * (master - master_default)
        derived_default = float(defaults[derived_param])
        derived_val = derived_default + slope * (master_val - master_default)

        # Clamp to valid range
        if derived_param in PARAM_RANGES:
            lo, hi = PARAM_RANGES[derived_param]
            derived_val = max(lo, min(hi, derived_val))
        if isinstance(defaults.get(derived_param), int):
            derived_val = int(round(derived_val))
        reconstructed[derived_param] = derived_val

    return reconstructed


def run_experiment_preview(presets: list[tuple[str, dict]], defaults: dict,
                           experiments: dict) -> None:
    """Run reconstruction preview for all experiments, compare results."""
    try:
        from lossy_rust import render_lossy
    except ImportError:
        print("\n  SKIP reconstruction preview: lossy_rust not available")
        return

    # Test signal with transients
    rng = np.random.default_rng(42)
    n_samples = int(SR * 2.0)
    test_signal = np.zeros(n_samples)
    for t_sec in [0.1, 0.4, 0.7, 1.0, 1.3, 1.6]:
        start = int(t_sec * SR)
        end = min(start + int(0.15 * SR), n_samples)
        test_signal[start:end] = rng.normal(0, 0.4, end - start)

    # Pre-render all originals (shared across experiments)
    print(f"\n{'=' * 80}")
    print("PRE-RENDERING all presets through DSP...")
    print(f"{'=' * 80}")
    originals = {}
    for name, preset in presets:
        full = {**defaults, **{k: v for k, v in preset.items()
                if k != "_meta" and k != "tail_length"}}
        try:
            out = render_lossy(test_signal, json.dumps(full))
            rms = float(np.sqrt(np.mean(out ** 2)))
            if rms > 1e-15:
                originals[name] = (out, rms)
        except Exception as e:
            print(f"  SKIP {name}: {e}")
    print(f"  Rendered {len(originals)}/{len(presets)} presets successfully")

    # Run each experiment
    summary_table = []

    for exp_name, config in experiments.items():
        n_fixed = len(config["fixed"])
        n_derived = len(config["derived"])
        n_macros = 41 - n_fixed - n_derived  # approximate
        all_eliminated = set(config["fixed"].keys()) | set(config["derived"].keys())

        print(f"\n{'=' * 80}")
        print(f"EXPERIMENT {exp_name}: {config['desc']}")
        print(f"  Fixed ({n_fixed}): {sorted(config['fixed'].keys())}")
        if config["derived"]:
            for dp, (mp, slope) in config["derived"].items():
                d_def = defaults[dp]
                m_def = defaults[mp]
                gate = config.get("gate", {}).get(dp)
                gate_str = f" [gate: {gate}]" if gate else ""
                print(f"  Derived: {dp} = {d_def} + {slope:.3f} * ({mp} - {m_def}){gate_str}")
        print(f"  Result: ~{n_macros} macro controls")
        print(f"{'=' * 80}")

        results = []
        for name, preset in presets:
            if name not in originals:
                results.append({"name": name, "error": "render failed"})
                continue

            out_orig, rms_orig = originals[name]
            reconstructed = simulate_experiment_roundtrip(preset, defaults, config)
            recon_clean = {k: v for k, v in reconstructed.items()
                           if k != "_meta" and k != "tail_length"}

            try:
                out_recon = render_lossy(test_signal, json.dumps(recon_clean))
            except Exception as e:
                results.append({"name": name, "error": str(e)})
                continue

            min_len = min(len(out_orig), len(out_recon))
            diff_rms = float(np.sqrt(np.mean(
                (out_orig[:min_len] - out_recon[:min_len]) ** 2)))
            rel_diff = diff_rms / rms_orig

            # Which eliminated params had non-default values?
            full = {**defaults, **{k: v for k, v in preset.items() if k != "_meta"}}
            changed = []
            for key in sorted(all_eliminated):
                orig_val = full.get(key, defaults[key])
                if key in config["fixed"]:
                    if orig_val != config["fixed"][key]:
                        changed.append(f"{key}={orig_val}")
                elif key in config["derived"]:
                    mp, slope = config["derived"][key]
                    master_val = float(full.get(mp, defaults[mp]))
                    d_def = float(defaults[key])
                    m_def = float(defaults[mp])
                    predicted = d_def + slope * (master_val - m_def)
                    if key in PARAM_RANGES:
                        lo, hi = PARAM_RANGES[key]
                        predicted = max(lo, min(hi, predicted))
                    if isinstance(defaults.get(key), int):
                        predicted = int(round(predicted))
                    if abs(float(orig_val) - float(predicted)) > 0.01:
                        changed.append(f"{key}={orig_val}→{predicted:.2f}")

            results.append({
                "name": name,
                "rel_diff": rel_diff,
                "changed": changed,
            })

        ok = [r for r in results if "error" not in r]
        err = [r for r in results if "error" in r]
        ok.sort(key=lambda r: r["rel_diff"], reverse=True)

        n_good = sum(1 for r in ok if r["rel_diff"] <= 0.05)
        n_warn = sum(1 for r in ok if 0.05 < r["rel_diff"] <= 0.20)
        n_bad = sum(1 for r in ok if r["rel_diff"] > 0.20)

        print(f"\n  {'Preset':<35} {'RelDiff':>8} {'Changed params'}")
        print(f"  {'-'*35} {'-'*8} {'-'*50}")
        for r in ok:
            tag = ""
            if r["rel_diff"] > 0.20:
                tag = " *** HIGH"
            elif r["rel_diff"] > 0.05:
                tag = " * warn"
            changed_str = ", ".join(r["changed"][:4]) if r["changed"] else "-"
            if len(r["changed"]) > 4:
                changed_str += f" (+{len(r['changed'])-4} more)"
            print(f"  {r['name']:<35} {r['rel_diff']:>7.4f}  {changed_str}{tag}")
        for r in err:
            print(f"  {r['name']:<35} {'ERROR':>8}  {r['error']}")

        print(f"\n  Summary: {n_good} good (<5%), {n_warn} warn (5-20%), "
              f"{n_bad} bad (>20%), {len(err)} errors")

        median_diff = float(np.median([r["rel_diff"] for r in ok])) if ok else 0.0
        max_diff = float(max(r["rel_diff"] for r in ok)) if ok else 0.0

        summary_table.append({
            "name": exp_name,
            "n_macros": n_macros,
            "n_good": n_good,
            "n_warn": n_warn,
            "n_bad": n_bad,
            "n_err": len(err),
            "median": median_diff,
            "max": max_diff,
        })

        # Plot worst 5 for this experiment
        if ok:
            worst = ok[:5]
            fig, axes = plt.subplots(len(worst), 1, figsize=(14, 3 * len(worst)))
            if len(worst) == 1:
                axes = [axes]
            for i, r in enumerate(worst):
                pname = r["name"]
                preset_dict = next(p for n, p in presets if n == pname)
                full_p = {**defaults, **{k: v for k, v in preset_dict.items()
                          if k != "_meta" and k != "tail_length"}}
                recon_p = simulate_experiment_roundtrip(preset_dict, defaults, config)
                recon_p = {k: v for k, v in recon_p.items()
                           if k != "_meta" and k != "tail_length"}

                out_o = render_lossy(test_signal, json.dumps(full_p))
                out_r = render_lossy(test_signal, json.dumps(recon_p))

                ax = axes[i]
                ml = min(len(out_o), len(out_r))
                t = np.arange(ml) / SR
                ax.plot(t, out_o[:ml], alpha=0.5, linewidth=0.5, label="Original")
                ax.plot(t, out_r[:ml], alpha=0.5, linewidth=0.5, label="Reconstructed")
                ax.set_title(f"{pname} (diff={r['rel_diff']:.3f})")
                ax.set_ylabel("Amplitude")
                ax.legend(fontsize=7)
            axes[-1].set_xlabel("Time (s)")
            plt.suptitle(f"Experiment {exp_name}: worst 5", fontsize=12)
            plt.tight_layout()
            path = OUT_DIR / f"recon_{exp_name}_worst5.png"
            plt.savefig(path, dpi=150)
            plt.close()

    # Final comparison table
    print(f"\n{'=' * 80}")
    print("EXPERIMENT COMPARISON")
    print(f"{'=' * 80}")
    print(f"  {'Experiment':<25} {'Macros':>6} {'Good':>5} {'Warn':>5} {'Bad':>5} "
          f"{'Err':>4} {'Median':>8} {'Max':>8}")
    print(f"  {'-'*25} {'-'*6} {'-'*5} {'-'*5} {'-'*5} {'-'*4} {'-'*8} {'-'*8}")
    for s in summary_table:
        print(f"  {s['name']:<25} {s['n_macros']:>6} {s['n_good']:>5} {s['n_warn']:>5} "
              f"{s['n_bad']:>5} {s['n_err']:>4} {s['median']:>7.4f} {s['max']:>8.4f}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    defaults = get_defaults()
    feature_names = build_feature_names()

    # ── Step 1: Load presets ──
    presets = load_presets()

    # Build feature matrix
    X_all = []
    names_all = []
    classes = []
    for name, preset in presets:
        X_all.append(preset_to_features(preset, defaults))
        names_all.append(name)
        classes.append(classify_preset(name, preset))

    X_all = np.array(X_all)
    print(f"Feature matrix: {X_all.shape}")
    print(f"Classes: {dict(zip(*np.unique(classes, return_counts=True)))}")

    # ── Step 6: Parameter usage stats ──
    param_usage_stats(presets, defaults)

    # ── Steps 3-5: PCA on all presets ──
    pca_all, _ = run_pca(X_all, names_all, feature_names, "All Presets")

    # ── Co-variance analysis ──
    covariance_analysis(X_all, feature_names)

    # ── Step 8: Category analysis ──
    # Run PCA on each category with enough presets
    unique_classes = sorted(set(classes))
    print(f"\nCategories found: {unique_classes}")
    category_datasets = {}
    for cat in unique_classes:
        mask = [c == cat for c in classes]
        X_cat = X_all[np.array(mask)]
        names_cat = [n for n, m in zip(names_all, mask) if m]
        if len(X_cat) > 5:
            category_datasets[cat] = (X_cat, names_cat)
            run_pca(X_cat, names_cat, feature_names, f"Category: {cat}")

    # ── Step 7: Random parameter sampling ──
    rng = np.random.default_rng(42)
    n_random = 300
    random_params = generate_random_params(n_random, rng)
    X_random = np.array([preset_to_features(p, defaults) for p in random_params])
    names_random = [f"rand_{i}" for i in range(n_random)]

    run_pca(X_random, names_random, feature_names, "Random Only")

    X_combined = np.vstack([X_all, X_random])
    names_combined = names_all + names_random
    run_pca(X_combined, names_combined, feature_names, "Presets + Random Combined")

    # Dimensionality comparison table
    print(f"\n{'=' * 80}")
    print("DIMENSIONALITY COMPARISON (Step 7)")
    print(f"{'=' * 80}")
    datasets = [("Presets only", X_all)]
    for cat, (X_cat, _) in category_datasets.items():
        datasets.append((cat, X_cat))
    datasets.extend([("Random only", X_random), ("Combined", X_combined)])
    for label, X in datasets:
        scaler = StandardScaler()
        p = PCA().fit(scaler.fit_transform(X))
        cv = np.cumsum(p.explained_variance_ratio_)
        n80 = int(np.argmax(cv >= 0.80) + 1)
        n90 = int(np.argmax(cv >= 0.90) + 1)
        n95 = int(np.argmax(cv >= 0.95) + 1)
        print(f"  {label:25s}  n={X.shape[0]:>4}  80%: {n80:2d} PCs  90%: {n90:2d} PCs  95%: {n95:2d} PCs")

    # ── Step 10a,b: Dead parameter detection (statistical) ──
    dead_var = dead_param_variance(X_all, feature_names)
    dead_load = dead_param_loading(pca_all, feature_names)

    all_dead = set(dead_var) | set(dead_load)
    if all_dead:
        print(f"\n  Combined dead params (variance + loading): {sorted(all_dead)}")

    # ── Step 10c: DSP sensitivity analysis (uses Rust DSP via PyO3) ──
    sensitivity = dead_param_dsp_sensitivity(presets, defaults)

    # ── Step 10: Perceptual-space PCA ──
    perceptual_space_pca(presets, defaults)

    # ── Anchored fits for co-varying params ──
    fits = compute_anchored_fits(presets, defaults)

    # ── Multi-experiment reconstruction preview ──
    experiments = build_experiments(fits, defaults)
    run_experiment_preview(presets, defaults, experiments)

    print(f"\n{'=' * 80}")
    print(f"DONE — all plots in {OUT_DIR}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
