#!/usr/bin/env python3
"""Reverb parameter space analysis — full methodology from docs/pca_parameter_reduction.md.

Steps covered:
  1. Build feature matrix from original presets (loaded from git HEAD)
  2. Pre-transform skewed params (log delay times)
  3. Standardize (zero mean, unit variance)
  4. Run PCA — scree, cumulative variance, loadings
  5. Interpret components (co-loading patterns)
  6. Array summary stats (CV analysis)
  7. Random parameter sampling (3 groups: presets, random, combined)
  8. Category analysis (hand-designed vs ML-generated)
  9. Cluster analysis (k-means on PCA space, silhouette)
 10. Dead parameter detection:
     a. Variance analysis (near-zero variance features)
     b. Loading analysis (no significant loading on top PCs)
     c. DSP sensitivity analysis — sweep each param through Rust DSP, measure output change
 11. Audio validation — render original vs simplified-reconstructed through Rust DSP,
     compare RT60, spectral centroid, energy envelope correlation

Requires: reverb_rust PyO3 bindings (build with: cd rust && uv run maturin develop -m crates/reverb-python/Cargo.toml --release)

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

# Add project root for shared imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.simplified_params import ReverbParams

PRESET_DIR = Path(__file__).parent.parent / "reverb" / "gui" / "presets"
OUT_DIR = Path(__file__).parent / "pca_output"
SR = 44100.0
N = 8

MATRIX_TYPES = [
    "householder", "hadamard", "diagonal",
    "random_orthogonal", "circulant", "stautner_puckette",
]

# ── Helpers ──────────────────────────────────────────────────────────────────


def get_defaults() -> dict:
    """Default ReverbParams matching Rust."""
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


def load_presets_from_git() -> list[tuple[str, dict, dict]]:
    """Load original full-format presets from git HEAD.

    The on-disk presets are in simplified format (converted earlier).
    For PCA to discover the real dimensionality, we need the originals
    with per-node variation intact. Returns (name, full_params, raw_json).
    """
    import subprocess

    presets = []
    # List preset files known to git
    result = subprocess.run(
        ["git", "ls-tree", "--name-only", "HEAD", "reverb/gui/presets/"],
        capture_output=True, text=True, cwd=str(Path(__file__).parent.parent),
    )
    for line in sorted(result.stdout.strip().splitlines()):
        if not line.endswith(".json"):
            continue
        name = Path(line).stem
        content = subprocess.run(
            ["git", "show", f"HEAD:{line}"],
            capture_output=True, text=True, cwd=str(Path(__file__).parent.parent),
        )
        raw = json.loads(content.stdout)
        # Original presets are in full ReverbParams format
        presets.append((name, raw, raw))

    print(f"Loaded {len(presets)} original presets from git HEAD")
    return presets


def load_presets_from_disk() -> list[tuple[str, dict, dict]]:
    """Load current on-disk presets (simplified format), convert to full.
    Returns (name, full_params, raw_json).
    """
    presets = []
    for p in sorted(PRESET_DIR.glob("*.json")):
        with open(p) as f:
            raw = json.load(f)
        if "size" in raw or "brightness" in raw:
            meta = raw.get("_meta")
            s = ReverbParams.from_dict(raw)
            full = s.to_fdn_params()
            if meta:
                full["_meta"] = meta
        else:
            full = raw
        presets.append((p.stem, full, raw))
    return presets


def classify_preset(name: str, meta: dict | None) -> str:
    cat = ""
    if meta and isinstance(meta, dict):
        cat = meta.get("category", "")
    if cat == "ML Generated" or name.startswith("z_gen_"):
        return "ml-generated"
    if name.startswith("z_"):
        return "experimental"
    return "hand-designed"


# ── Step 1+2: Feature extraction with log-transforms ────────────────────────


def build_feature_names() -> list[str]:
    """Feature names matching preset_to_features output."""
    names = []
    for i in range(N):
        names.append(f"log_delay_ms_{i}")  # Step 2: log-transformed
    for key in ["damping", "input_gain", "output_gain"]:
        for i in range(N):
            names.append(f"{key}_{i}")
    names.extend([
        "feedback_gain", "wet_dry", "diffusion", "saturation",
        "log_pre_delay_ms", "stereo_width",  # Step 2: log pre_delay
    ])
    for t in MATRIX_TYPES:
        names.append(f"matrix_{t}")
    return names


def preset_to_features(preset: dict, defaults: dict) -> np.ndarray:
    """Convert preset to feature vector with log-transforms (Step 1+2)."""
    features = []

    # Delay times → log(ms) (Step 2: exponential perceptual mapping)
    delay_samples = preset.get("delay_times", defaults["delay_times"])
    for s in delay_samples[:N]:
        ms = max(s / SR * 1000.0, 0.01)  # floor to avoid log(0)
        features.append(math.log(ms))

    # Per-node arrays (linear scale is fine)
    for key in ["damping_coeffs", "input_gains", "output_gains"]:
        arr = preset.get(key, defaults[key])
        features.extend(arr[:N])

    # Scalars
    features.append(preset.get("feedback_gain", defaults["feedback_gain"]))
    features.append(preset.get("wet_dry", defaults["wet_dry"]))
    features.append(preset.get("diffusion", defaults["diffusion"]))
    features.append(preset.get("saturation", defaults["saturation"]))

    # Pre-delay → log(ms) (Step 2)
    pd_samples = preset.get("pre_delay", defaults["pre_delay"])
    pd_ms = max(pd_samples / SR * 1000.0, 0.01)
    features.append(math.log(pd_ms))

    features.append(preset.get("stereo_width", defaults["stereo_width"]))

    # Matrix type one-hot
    mt = preset.get("matrix_type", defaults["matrix_type"])
    if isinstance(mt, int):
        mt = MATRIX_TYPES[mt] if mt < len(MATRIX_TYPES) else "householder"
    for t in MATRIX_TYPES:
        features.append(1.0 if mt == t else 0.0)

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

    print(f"\nPer-component (first 12):")
    for i in range(min(12, len(pca.explained_variance_ratio_))):
        print(f"  PC{i+1}: {pca.explained_variance_ratio_[i]:.4f} "
              f"(cum: {cumvar[i]:.4f})")

    # Step 5: top loadings per PC
    print(f"\nLoadings (first 8 PCs):")
    for pc in range(min(8, pca.n_components_)):
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
                print(f"  {c}: ({len(members)} members)")
    else:
        cluster_labels = np.zeros(X.shape[0], dtype=int)

    # Plot
    if save_plot:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(title, fontsize=14)

        ax = axes[0]
        n_show = min(20, len(pca.explained_variance_ratio_))
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
        if len(labels) <= 80:
            for i, lb in enumerate(labels):
                ax.annotate(lb, (X_pca[i, 0], X_pca[i, 1]), fontsize=4, alpha=0.6, rotation=25)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.set_title("PC1 vs PC2")

        ax = axes[2]
        sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap="tab10", alpha=0.7, s=30)
        ax.set_xlabel(f"PC1")
        ax.set_ylabel(f"PC2")
        ax.set_title(f"Clusters (k={len(np.unique(cluster_labels))})")
        plt.colorbar(sc, ax=ax)

        plt.tight_layout()
        OUT_DIR.mkdir(exist_ok=True)
        path = OUT_DIR / f"pca_{title.lower().replace(' ', '_').replace('+', '_')}.png"
        plt.savefig(path, dpi=150)
        print(f"\n  Plot: {path}")
        plt.close()

    return pca, X_pca


# ── Step 6: Array CV stats ──────────────────────────────────────────────────


def array_cv_stats(presets: list[tuple[str, dict, dict]], defaults: dict) -> None:
    """Per-node array coefficient of variation analysis (Step 6)."""
    print(f"\n{'=' * 80}")
    print("ARRAY CV STATS (Step 6)")
    print(f"{'=' * 80}")

    for arr_name in ["delay_times", "damping_coeffs", "input_gains", "output_gains"]:
        cvs = []
        uniform = 0
        for _, preset, _ in presets:
            vals = np.array(preset.get(arr_name, defaults[arr_name])[:N], dtype=np.float64)
            mean = np.mean(vals)
            cv = np.std(vals) / mean if abs(mean) > 1e-10 else 0.0
            cvs.append(cv)
            if cv < 0.1:
                uniform += 1

        print(f"\n  {arr_name}:")
        print(f"    Mean CV: {np.mean(cvs):.4f}  Median: {np.median(cvs):.4f}")
        print(f"    Uniform (CV<0.1): {uniform}/{len(presets)} ({uniform/len(presets)*100:.0f}%)")


# ── Step 7: Random sampling ─────────────────────────────────────────────────


def generate_random_params(n: int, rng: np.random.Generator) -> list[dict]:
    """Generate random full ReverbParams by uniform sampling (Step 7)."""
    defaults = get_defaults()
    results = []
    for _ in range(n):
        p = defaults.copy()
        p["delay_times"] = [int(rng.uniform(44, 13230)) for _ in range(N)]
        p["damping_coeffs"] = [float(rng.uniform(0.0, 0.95)) for _ in range(N)]
        p["input_gains"] = [float(rng.uniform(0.01, 0.5)) for _ in range(N)]
        p["output_gains"] = [float(rng.uniform(0.1, 2.0)) for _ in range(N)]
        p["feedback_gain"] = float(rng.uniform(0.0, 1.15))
        p["wet_dry"] = float(rng.uniform(0.0, 1.0))
        p["diffusion"] = float(rng.uniform(0.0, 0.7))
        p["saturation"] = float(rng.uniform(0.0, 1.0))
        p["pre_delay"] = int(rng.uniform(0, 11025))
        p["stereo_width"] = float(rng.uniform(0.0, 1.0))
        p["matrix_type"] = str(rng.choice(MATRIX_TYPES))
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

    # Also show lowest-variance features
    sorted_idx = np.argsort(variances)
    print(f"\n  Lowest variance features:")
    for i in sorted_idx[:10]:
        print(f"    {feature_names[i]:25s}  var={variances[i]:.4e}")
    return dead


def dead_param_loading(pca: PCA, feature_names: list[str], threshold: float = 0.95) -> list[str]:
    """Find features with no significant loading on top PCs (Step 10b)."""
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_top = int(np.argmax(cumvar >= threshold) + 1)

    # Max absolute loading across top PCs
    max_loading = np.max(np.abs(pca.components_[:n_top]), axis=0)

    dead = []
    print(f"\n{'=' * 80}")
    print(f"DEAD PARAMETERS: Loading Analysis (Step 10b, top {n_top} PCs for {threshold:.0%})")
    print(f"{'=' * 80}")

    sorted_idx = np.argsort(max_loading)
    print(f"  Weakest loadings:")
    for i in sorted_idx[:15]:
        tag = " ← DEAD" if max_loading[i] < 0.15 else ""
        print(f"    {feature_names[i]:25s}  max|loading|={max_loading[i]:.3f}{tag}")
        if max_loading[i] < 0.15:
            dead.append(feature_names[i])
    return dead


def dead_param_dsp_sensitivity(presets: list[tuple[str, dict, dict]], defaults: dict) -> dict:
    """Sweep each simplified param, render through Rust DSP, measure output change (Step 10c).

    This is the ground truth — a parameter is perceptually dead if sweeping it
    across its full range produces negligible audio change.
    """
    from reverb_rust import render_fdn

    print(f"\n{'=' * 80}")
    print("DEAD PARAMETERS: DSP Sensitivity Analysis (Step 10c)")
    print(f"{'=' * 80}")

    ir_len = int(SR * 2.0)
    impulse = np.zeros(ir_len)
    impulse[0] = 1.0

    # Use a few representative presets as test points
    test_presets = []
    for name, full, raw in presets:
        cat = classify_preset(name, full.get("_meta"))
        if cat == "hand-designed":
            test_presets.append((name, full))
    # Pick ~5 spread across the preset bank
    step = max(1, len(test_presets) // 5)
    test_presets = test_presets[::step][:5]

    # Simplified params to sweep and their ranges
    sweep_params = {
        "size": np.linspace(0.0, 1.0, 5),
        "decay": np.linspace(0.3, 1.0, 5),
        "brightness": np.linspace(0.0, 1.0, 5),
        "diffusion": np.linspace(0.0, 0.7, 5),
        "mix": np.linspace(0.1, 1.0, 5),
        "saturation": np.linspace(0.0, 1.0, 5),
        "pre_delay_ms": np.linspace(0.0, 100.0, 5),
        "stereo_width": np.linspace(0.0, 1.0, 5),
        "mod_rate": np.linspace(0.0, 5.0, 5),
        "mod_depth": np.linspace(0.0, 1.0, 5),
        "mod_character": np.linspace(0.0, 1.0, 5),
        "mod_spread": np.linspace(0.0, 1.0, 5),
    }

    sensitivity = {}

    for param_name, sweep_values in sweep_params.items():
        rms_changes = []

        for preset_name, full in test_presets:
            # Get baseline simplified params
            base = ReverbParams.from_fdn_params(full)
            base_full = base.to_fdn_params()
            base_json = json.dumps(base_full)

            try:
                out_base = render_fdn(impulse, base_json)
                base_mono = out_base.reshape(-1, 2).mean(axis=1)
                base_rms = float(np.sqrt(np.mean(base_mono ** 2)))
            except Exception:
                continue

            if base_rms < 1e-10:
                continue

            # Sweep parameter
            max_change = 0.0
            for val in sweep_values:
                swept = ReverbParams(**{
                    **base.__dict__,
                    param_name: val if not isinstance(val, np.floating) else float(val),
                })
                swept_full = swept.to_fdn_params()
                try:
                    out_swept = render_fdn(impulse, json.dumps(swept_full))
                    swept_mono = out_swept.reshape(-1, 2).mean(axis=1)

                    # RMS of difference relative to base
                    min_len = min(len(base_mono), len(swept_mono))
                    diff_rms = float(np.sqrt(np.mean((base_mono[:min_len] - swept_mono[:min_len]) ** 2)))
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
        tag = " ← LOW SENSITIVITY" if change < 0.05 else ""
        print(f"    {param:20s}  {change:.4f}{tag}")

    return sensitivity


# ── Step 14: Audio validation ───────────────────────────────────────────────


def estimate_rt60(signal: np.ndarray) -> float:
    """RT60 via Schroeder backward integration."""
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


def spectral_centroid(signal: np.ndarray) -> float:
    """Spectral centroid in Hz."""
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / SR)
    total = np.sum(spectrum)
    if total > 1e-15:
        return float(np.sum(freqs * spectrum) / total)
    return 0.0


def energy_envelope(signal: np.ndarray, window_ms: float = 50.0) -> np.ndarray:
    """Smoothed RMS energy envelope."""
    window = max(1, int(SR * window_ms / 1000.0))
    squared = signal ** 2
    cumsum = np.cumsum(np.concatenate([[0], squared]))
    return np.sqrt((cumsum[window:] - cumsum[:-window]) / window)


def audio_validation(presets: list[tuple[str, dict, dict]], defaults: dict) -> None:
    """Render original vs reconstructed through Rust DSP, compare perceptually (Step 14)."""
    from reverb_rust import render_fdn

    print(f"\n{'=' * 80}")
    print("AUDIO VALIDATION: Original vs Simplified→Full (Rust DSP, Step 14)")
    print(f"{'=' * 80}")

    ir_len = int(SR * 3.0)
    impulse = np.zeros(ir_len)
    impulse[0] = 1.0

    results = []
    for name, full, raw in presets:
        meta = full.get("_meta")
        category = classify_preset(name, meta)

        # Original full params → DSP
        dsp_original = {k: v for k, v in full.items() if not k.startswith("_")}

        # Simplified round-trip → DSP
        simplified = ReverbParams.from_fdn_params(full)
        reconstructed = simplified.to_fdn_params()

        try:
            out_orig = render_fdn(impulse, json.dumps(dsp_original)).reshape(-1, 2).mean(axis=1)
            out_recon = render_fdn(impulse, json.dumps(reconstructed)).reshape(-1, 2).mean(axis=1)
        except Exception as e:
            print(f"  SKIP {name}: {e}")
            continue

        # RT60
        rt60_o = estimate_rt60(out_orig)
        rt60_r = estimate_rt60(out_recon)
        rt60_err = abs(rt60_o - rt60_r) / rt60_o if 0.01 < rt60_o < 100 else 0.0

        # Spectral centroid
        sc_o = spectral_centroid(out_orig)
        sc_r = spectral_centroid(out_recon)
        sc_err = abs(sc_o - sc_r) / sc_o if sc_o > 1.0 else 0.0

        # Energy envelope correlation
        env_o = energy_envelope(out_orig)
        env_r = energy_envelope(out_recon)
        ml = min(len(env_o), len(env_r))
        if ml > 10 and np.std(env_o[:ml]) > 1e-10:
            env_corr = float(np.corrcoef(env_o[:ml], env_r[:ml])[0, 1])
        else:
            env_corr = 1.0

        # Level difference
        rms_o = np.sqrt(np.mean(out_orig ** 2))
        rms_r = np.sqrt(np.mean(out_recon ** 2))
        lvl_db = 20 * np.log10(rms_r / rms_o) if rms_o > 1e-15 else 0.0

        results.append({
            "name": name, "category": category,
            "rt60_o": rt60_o, "rt60_r": rt60_r, "rt60_err": rt60_err,
            "sc_o": sc_o, "sc_r": sc_r, "sc_err": sc_err,
            "env_corr": env_corr, "lvl_db": lvl_db,
        })

    # Sort by envelope correlation (worst first)
    results.sort(key=lambda r: r["env_corr"])

    print(f"\n{'Preset':<35} {'Cat':<6} {'RT60o':>6} {'RT60r':>6} {'RT60%':>6} "
          f"{'SCo':>6} {'SCr':>6} {'EnvC':>6} {'dB':>5}")
    print("-" * 95)
    for r in results:
        rt_o = f"{r['rt60_o']:.2f}" if r["rt60_o"] < 100 else "inf"
        rt_r = f"{r['rt60_r']:.2f}" if r["rt60_r"] < 100 else "inf"
        cat = "hand" if r["category"] == "hand-designed" else "ml"
        print(f"{r['name']:<35} {cat:<6} {rt_o:>6} {rt_r:>6} {r['rt60_err']*100:>5.1f}% "
              f"{r['sc_o']:>5.0f}Hz {r['sc_r']:>5.0f}Hz {r['env_corr']:>6.4f} {r['lvl_db']:>+5.1f}")

    # Summary per category
    for label, group in [("Hand-designed", [r for r in results if r["category"] == "hand-designed"]),
                         ("ML-generated", [r for r in results if r["category"] == "ml-generated"])]:
        if not group:
            continue
        rt_errs = [r["rt60_err"] for r in group if 0.01 < r["rt60_o"] < 100]
        env_corrs = [r["env_corr"] for r in group]
        print(f"\n  {label} ({len(group)} presets):")
        if rt_errs:
            print(f"    RT60 err: median={np.median(rt_errs)*100:.1f}%  "
                  f"<10%: {sum(1 for e in rt_errs if e < 0.10)}/{len(rt_errs)}  "
                  f"<20%: {sum(1 for e in rt_errs if e < 0.20)}/{len(rt_errs)}")
        print(f"    Env corr: median={np.median(env_corrs):.4f}  "
              f">0.90: {sum(1 for c in env_corrs if c > 0.90)}/{len(env_corrs)}  "
              f">0.95: {sum(1 for c in env_corrs if c > 0.95)}/{len(env_corrs)}")

    # Spectrogram comparison for worst 3
    OUT_DIR.mkdir(exist_ok=True)
    worst = results[:3]
    if worst:
        from reverb_rust import render_fdn as _rfdn
        fig, axes = plt.subplots(len(worst), 2, figsize=(14, 4 * len(worst)))
        if len(worst) == 1:
            axes = axes.reshape(1, -1)
        for i, r in enumerate(worst):
            name = r["name"]
            full = None
            for n, f, _ in presets:
                if n == name:
                    full = f
                    break
            if full is None:
                continue
            dsp_orig = {k: v for k, v in full.items() if not k.startswith("_")}
            recon = ReverbParams.from_fdn_params(full).to_fdn_params()
            out_o = _rfdn(impulse, json.dumps(dsp_orig)).reshape(-1, 2).mean(axis=1)
            out_r = _rfdn(impulse, json.dumps(recon)).reshape(-1, 2).mean(axis=1)

            for j, (sig, label) in enumerate([(out_o, "Original"), (out_r, "Reconstructed")]):
                ax = axes[i, j]
                ax.specgram(sig, NFFT=1024, Fs=int(SR), noverlap=512, cmap="magma")
                ax.set_title(f"{name} — {label}")
                ax.set_ylabel("Hz")
                ax.set_ylim(0, 8000)
            axes[i, 0].set_xlabel("")
        axes[-1, 0].set_xlabel("Time (s)")
        axes[-1, 1].set_xlabel("Time (s)")
        plt.tight_layout()
        path = OUT_DIR / "audio_validation_spectrograms.png"
        plt.savefig(path, dpi=150)
        print(f"\n  Spectrogram comparison: {path}")
        plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    OUT_DIR.mkdir(exist_ok=True)
    defaults = get_defaults()
    feature_names = build_feature_names()

    # ── Step 1: Load original presets from git (with per-node variation intact) ──
    # On-disk presets are simplified format. For PCA to discover true dimensionality,
    # we need the originals that have actual per-node array variation.
    presets = load_presets_from_git()

    # Build feature matrix
    X_all = []
    names_all = []
    classes = []
    for name, full, raw in presets:
        X_all.append(preset_to_features(full, defaults))
        names_all.append(name)
        classes.append(classify_preset(name, full.get("_meta")))

    X_all = np.array(X_all)
    print(f"Feature matrix: {X_all.shape}")
    print(f"Classes: {dict(zip(*np.unique(classes, return_counts=True)))}")

    # ── Step 6: Array CV stats (on originals — shows real per-node variation) ──
    array_cv_stats(presets, defaults)

    # ── Steps 3-5: PCA on all presets ──
    pca_all, _ = run_pca(X_all, names_all, feature_names, "All Presets")

    # ── Step 8: Category analysis ──
    mask_hand = [c == "hand-designed" for c in classes]
    X_hand = X_all[mask_hand]
    names_hand = [n for n, m in zip(names_all, mask_hand) if m]
    if len(X_hand) > 3:
        pca_hand, _ = run_pca(X_hand, names_hand, feature_names, "Hand-Designed Only")

    mask_ml = [c == "ml-generated" for c in classes]
    X_ml = X_all[mask_ml]
    names_ml = [n for n, m in zip(names_all, mask_ml) if m]
    if len(X_ml) > 3:
        run_pca(X_ml, names_ml, feature_names, "ML-Generated Only")

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
    datasets = [("Presets only", X_all), ("Hand-designed", X_hand),
                ("Random only", X_random), ("Combined", X_combined)]
    if len(X_ml) > 3:
        datasets.insert(2, ("ML-generated", X_ml))
    for label, X in datasets:
        scaler = StandardScaler()
        p = PCA().fit(scaler.fit_transform(X))
        cv = np.cumsum(p.explained_variance_ratio_)
        n90 = int(np.argmax(cv >= 0.90) + 1)
        n95 = int(np.argmax(cv >= 0.95) + 1)
        print(f"  {label:25s}  n={X.shape[0]:>4}  90%: {n90:2d} PCs  95%: {n95:2d} PCs")

    # ── Step 10a,b: Dead parameter detection (statistical) ──
    dead_var = dead_param_variance(X_all, feature_names)
    dead_load = dead_param_loading(pca_all, feature_names)

    all_dead = set(dead_var) | set(dead_load)
    if all_dead:
        print(f"\n  Combined dead params (variance + loading): {sorted(all_dead)}")

    # ── Step 10c: DSP sensitivity analysis (uses Rust DSP via PyO3) ──
    sensitivity = dead_param_dsp_sensitivity(presets, defaults)

    # ── Step 14: Audio validation (original full params vs simplified round-trip, Rust DSP) ──
    # Uses original presets from git — renders both the original full params and the
    # simplified→full reconstructed params through the Rust DSP, then compares
    # RT60, spectral centroid, energy envelope, and spectrograms.
    audio_validation(presets, defaults)

    print(f"\n{'=' * 80}")
    print("DONE — all plots in scripts/pca_output/")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
