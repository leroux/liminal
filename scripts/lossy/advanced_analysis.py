#!/usr/bin/env python3
"""Advanced parameter reduction methods for lossy plugin.

Compares five approaches from docs/parameter_reduction.md:
  1. Sparse PCA — cleaner loadings, each component loads on fewer params
  2. Factor Analysis — separates intentional variation from noise
  3. Sobol Sensitivity Analysis — which params actually affect the sound
  4. Autoencoder — nonlinear compression to latent space
  5. Perceptual Distance Optimization — tune mapping to minimize audible error

All methods use the same 98-preset bank and Rust DSP via PyO3.

Requires: lossy_rust, SALib, torch, sklearn, scipy
"""

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import FactorAnalysis, PCA, SparsePCA
from sklearn.preprocessing import StandardScaler

SR = 44100.0
PRESET_DIR = Path(__file__).parent.parent.parent / "lossy" / "gui" / "presets"
OUT_DIR = Path(__file__).parent / "advanced_output"


# ── Shared data loading (from lossy_pca.py) ──────────────────────────────────


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


FEATURE_KEYS = [
    "loss", "jitter", "global_amount", "phase_loss", "pre_echo",
    "noise_shape", "weighting", "hf_threshold", "transient_ratio", "slushy_rate",
    "crush", "decimate",
    "packet_rate", "packet_size",
    "filter_width",
    "verb", "decay",
    "freezer",
    "gate", "threshold", "auto_gain", "loss_gain",
    "bounce_rate", "bounce_lfo_min", "bounce_lfo_max",
    "wet_dry",
]

INT_KEYS = [
    "inverse", "quantizer", "hop_divisor", "n_bands",
    "packets", "filter_type", "filter_slope",
    "verb_position", "freeze", "freeze_mode",
    "bounce", "bounce_target",
]

import math

def build_feature_names():
    names = list(FEATURE_KEYS) + INT_KEYS + ["log_filter_freq", "log2_window_size"]
    return names


def preset_to_features(preset, defaults):
    features = []
    for key in FEATURE_KEYS:
        features.append(float(preset.get(key, defaults[key])))
    for key in INT_KEYS:
        features.append(float(preset.get(key, defaults[key])))
    features.append(math.log(max(float(preset.get("filter_freq", defaults["filter_freq"])), 0.01)))
    features.append(math.log2(max(float(preset.get("window_size", defaults["window_size"])), 1)))
    return np.array(features, dtype=np.float64)


def load_presets():
    presets = []
    for p in sorted(PRESET_DIR.glob("*.json")):
        if p.name == "favorites.json":
            continue
        with open(p) as f:
            raw = json.load(f)
        presets.append((p.stem, raw))
    return presets


def make_test_signal():
    """Noise bursts with transients — exercises nonlinear DSP paths."""
    rng = np.random.default_rng(42)
    n = int(SR * 2.0)
    sig = np.zeros(n)
    for t in [0.1, 0.4, 0.7, 1.0, 1.3, 1.6]:
        s = int(t * SR)
        e = min(s + int(0.15 * SR), n)
        sig[s:e] = rng.normal(0, 0.4, e - s)
    return sig


PARAM_RANGES = {
    "loss": (0.0, 1.0), "jitter": (0.0, 1.0), "global_amount": (0.0, 1.0),
    "phase_loss": (0.0, 1.0), "pre_echo": (0.0, 1.0), "noise_shape": (0.0, 1.0),
    "weighting": (0.0, 1.0), "hf_threshold": (0.0, 1.0),
    "transient_ratio": (1.5, 20.0), "slushy_rate": (0.001, 0.5),
    "crush": (0.0, 1.0), "decimate": (0.0, 1.0),
    "packet_rate": (0.0, 1.0), "packet_size": (5.0, 200.0),
    "filter_freq": (20.0, 20000.0), "filter_width": (0.0, 1.0),
    "verb": (0.0, 1.0), "decay": (0.0, 1.0),
    "freezer": (0.0, 1.0), "gate": (0.0, 1.0),
    "threshold": (0.0, 1.0), "auto_gain": (0.0, 1.0), "loss_gain": (0.0, 1.0),
    "bounce_rate": (0.0, 1.0), "bounce_lfo_min": (0.01, 50.0),
    "bounce_lfo_max": (0.01, 50.0), "wet_dry": (0.0, 1.0),
}


# ── 1. Sparse PCA ────────────────────────────────────────────────────────────


def run_sparse_pca(X, feature_names):
    """Sparse PCA: each component loads on fewer params → cleaner macro assignments."""
    print(f"\n{'='*80}")
    print("1. SPARSE PCA")
    print(f"{'='*80}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Standard PCA for reference
    pca = PCA()
    pca.fit(X_scaled)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n90_std = int(np.argmax(cumvar >= 0.90) + 1)
    n95_std = int(np.argmax(cumvar >= 0.95) + 1)
    print(f"  Standard PCA: 90%={n90_std} PCs, 95%={n95_std} PCs")

    # Sparse PCA with varying alpha
    for alpha in [0.5, 1.0, 2.0, 5.0]:
        t0 = time.time()
        spca = SparsePCA(n_components=min(15, X.shape[0] - 1), alpha=alpha,
                         random_state=42, max_iter=500)
        X_sparse = spca.fit_transform(X_scaled)
        dt = time.time() - t0

        # Count non-zero loadings per component
        components = spca.components_
        nnz_per_comp = [(np.abs(components[i]) > 0.01).sum()
                        for i in range(components.shape[0])]

        # Reconstruction error
        X_recon = X_sparse @ components + scaler.mean_
        recon_err = np.mean((X_scaled - (X_sparse @ components)) ** 2)

        print(f"\n  alpha={alpha} ({dt:.1f}s):")
        print(f"    Reconstruction MSE: {recon_err:.4f}")
        print(f"    Non-zero loadings per component: {nnz_per_comp}")
        print(f"    Mean sparsity: {np.mean(nnz_per_comp):.1f} params/component")

        # Show top loadings per component
        for pc in range(min(8, components.shape[0])):
            loadings = components[pc]
            nonzero = np.where(np.abs(loadings) > 0.01)[0]
            if len(nonzero) == 0:
                continue
            sorted_nz = nonzero[np.argsort(np.abs(loadings[nonzero]))[::-1]]
            parts = [f"{feature_names[j]}={loadings[j]:+.3f}" for j in sorted_nz[:6]]
            print(f"    SC{pc+1} ({len(nonzero)} params): {', '.join(parts)}")

    # Best alpha=1.0 for detailed plot
    spca = SparsePCA(n_components=min(15, X.shape[0] - 1), alpha=1.0,
                     random_state=42, max_iter=500)
    spca.fit(X_scaled)
    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(np.abs(spca.components_[:10]), aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(10))
    ax.set_yticklabels([f"SC{i+1}" for i in range(10)])
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=90, fontsize=6)
    ax.set_title("Sparse PCA Loadings (alpha=1.0, |loading| magnitude)")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "sparse_pca_loadings.png", dpi=150)
    plt.close()
    print(f"\n  Plot: {OUT_DIR / 'sparse_pca_loadings.png'}")


# ── 2. Factor Analysis ───────────────────────────────────────────────────────


def run_factor_analysis(X, feature_names):
    """Factor Analysis: separates shared latent factors from per-param noise."""
    print(f"\n{'='*80}")
    print("2. FACTOR ANALYSIS")
    print(f"{'='*80}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Try different numbers of factors
    best_ll = -np.inf
    best_n = 5
    results = []
    for n_factors in range(3, min(25, X.shape[0] // 4)):
        fa = FactorAnalysis(n_components=n_factors, random_state=42, max_iter=1000)
        try:
            fa.fit(X_scaled)
            ll = fa.score(X_scaled)
            results.append((n_factors, ll))
            if ll > best_ll:
                best_ll = ll
                best_n = n_factors
        except Exception:
            break

    print(f"  Best n_factors={best_n} (log-likelihood={best_ll:.2f})")
    print(f"  Log-likelihood by n_factors:")
    for n, ll in results:
        marker = " <-- best" if n == best_n else ""
        print(f"    {n:>3} factors: LL={ll:.2f}{marker}")

    # Fit with best n
    fa = FactorAnalysis(n_components=best_n, random_state=42, max_iter=1000)
    fa.fit(X_scaled)

    # Noise variance per feature (higher = more idiosyncratic noise)
    noise = fa.noise_variance_
    sorted_idx = np.argsort(noise)[::-1]
    print(f"\n  Per-parameter noise variance (high = random/noisy, low = explained by factors):")
    for i in sorted_idx:
        tag = " <-- HIGH NOISE" if noise[i] > 0.8 else ""
        print(f"    {feature_names[i]:25s}  noise={noise[i]:.3f}{tag}")

    # Factor loadings
    loadings = fa.components_
    print(f"\n  Factor loadings (top 6 per factor):")
    for f in range(best_n):
        top = np.argsort(np.abs(loadings[f]))[::-1][:6]
        parts = [f"{feature_names[j]}={loadings[f, j]:+.3f}" for j in top]
        print(f"    F{f+1}: {', '.join(parts)}")

    # Compare FA noise to PCA: which params does FA think are "just noise"?
    high_noise = [feature_names[i] for i in range(len(noise)) if noise[i] > 0.8]
    low_noise = [feature_names[i] for i in range(len(noise)) if noise[i] < 0.2]
    print(f"\n  High noise params (>0.8, candidates to fix): {high_noise}")
    print(f"  Low noise params (<0.2, definitely keep): {low_noise}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.barh(range(len(noise)), noise[np.argsort(noise)[::-1]])
    ax1.set_yticks(range(len(noise)))
    ax1.set_yticklabels([feature_names[i] for i in np.argsort(noise)[::-1]], fontsize=6)
    ax1.set_xlabel("Noise Variance")
    ax1.set_title("Per-Parameter Noise (FA)")
    ax1.axvline(x=0.8, color="r", linestyle="--", alpha=0.5, label="noise threshold")
    ax1.legend()

    if results:
        ns, lls = zip(*results)
        ax2.plot(ns, lls, "bo-")
        ax2.axvline(x=best_n, color="r", linestyle="--", alpha=0.5)
        ax2.set_xlabel("Number of Factors")
        ax2.set_ylabel("Log-Likelihood")
        ax2.set_title("Factor Analysis Model Selection")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "factor_analysis.png", dpi=150)
    plt.close()

    return high_noise


# ── 3. Sobol Sensitivity Analysis ────────────────────────────────────────────


def run_sobol(defaults):
    """Sobol global sensitivity: which params matter to the SOUND, including interactions."""
    print(f"\n{'='*80}")
    print("3. SOBOL SENSITIVITY ANALYSIS")
    print(f"{'='*80}")

    try:
        from lossy_rust import render_lossy
    except ImportError:
        print("  SKIP: lossy_rust not available")
        return {}

    from SALib.analyze import sobol as sobol_analyze
    from SALib.sample import saltelli

    # Define the parameter space — continuous params only (Sobol needs continuous)
    param_names = []
    bounds = []
    for key in sorted(PARAM_RANGES.keys()):
        param_names.append(key)
        lo, hi = PARAM_RANGES[key]
        bounds.append([lo, hi])

    problem = {
        "num_vars": len(param_names),
        "names": param_names,
        "bounds": bounds,
    }

    # Generate Saltelli samples (N * (2D + 2) total evaluations)
    N = 256  # balance speed vs accuracy
    t0 = time.time()
    param_samples = saltelli.sample(problem, N, calc_second_order=False)
    n_evals = param_samples.shape[0]
    print(f"  Parameters: {len(param_names)}")
    print(f"  Saltelli samples: N={N} → {n_evals} evaluations")

    test_signal = make_test_signal()

    # Render each sample and compute RMS output
    outputs = np.zeros(n_evals)
    for i in range(n_evals):
        p = defaults.copy()
        for j, key in enumerate(param_names):
            p[key] = float(param_samples[i, j])
        try:
            out = render_lossy(test_signal, json.dumps(p))
            outputs[i] = float(np.sqrt(np.mean(out ** 2)))
        except Exception:
            outputs[i] = 0.0

        if (i + 1) % 1000 == 0:
            print(f"    Rendered {i+1}/{n_evals}...")

    dt = time.time() - t0
    print(f"  Rendering done in {dt:.1f}s ({n_evals/dt:.0f} evals/sec)")

    # Sobol analysis
    Si = sobol_analyze.analyze(problem, outputs, calc_second_order=False)

    S1 = Si["S1"]
    ST = Si["ST"]

    # Sort by total-order index
    sorted_idx = np.argsort(ST)[::-1]
    print(f"\n  {'Parameter':<20} {'S1 (first)':>12} {'ST (total)':>12} {'Interaction':>12}")
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12}")
    for i in sorted_idx:
        interaction = ST[i] - S1[i]
        tag = ""
        if ST[i] < 0.01:
            tag = " <-- DEAD"
        elif interaction > 0.05:
            tag = " <-- INTERACTIONS"
        print(f"  {param_names[i]:<20} {S1[i]:>12.4f} {ST[i]:>12.4f} {interaction:>12.4f}{tag}")

    dead_sobol = [param_names[i] for i in range(len(ST)) if ST[i] < 0.01]
    interactive = [(param_names[i], ST[i] - S1[i]) for i in range(len(ST))
                   if ST[i] - S1[i] > 0.05]
    print(f"\n  Dead params (ST < 0.01): {dead_sobol}")
    print(f"  Params with strong interactions: {interactive}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    y = np.arange(len(param_names))
    ax1.barh(y, ST[sorted_idx], alpha=0.7, label="Total-order (ST)")
    ax1.barh(y, S1[sorted_idx], alpha=0.7, label="First-order (S1)")
    ax1.set_yticks(y)
    ax1.set_yticklabels([param_names[i] for i in sorted_idx], fontsize=7)
    ax1.set_xlabel("Sensitivity Index")
    ax1.set_title("Sobol Sensitivity Indices")
    ax1.legend()

    # Interaction magnitude
    interactions = ST - S1
    ax2.barh(y, interactions[sorted_idx], alpha=0.7, color="orange")
    ax2.set_yticks(y)
    ax2.set_yticklabels([param_names[i] for i in sorted_idx], fontsize=7)
    ax2.set_xlabel("Interaction Index (ST - S1)")
    ax2.set_title("Parameter Interactions")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "sobol_sensitivity.png", dpi=150)
    plt.close()
    print(f"\n  Plot: {OUT_DIR / 'sobol_sensitivity.png'}")

    return dict(zip(param_names, ST))


# ── 4. Autoencoder ────────────────────────────────────────────────────────────


def run_autoencoder(X, feature_names):
    """Autoencoder: nonlinear compression to latent space."""
    print(f"\n{'='*80}")
    print("4. AUTOENCODER")
    print(f"{'='*80}")

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_features = X_scaled.shape[1]
    X_tensor = torch.FloatTensor(X_scaled)

    # Augment: perturbed presets (add Gaussian noise, 15% of range)
    rng = np.random.default_rng(42)
    n_aug = 500
    X_aug = np.tile(X_scaled, (n_aug // X_scaled.shape[0] + 1, 1))[:n_aug]
    X_aug += rng.normal(0, 0.15, X_aug.shape)
    X_aug_tensor = torch.FloatTensor(X_aug)

    X_train = torch.cat([X_tensor, X_aug_tensor], dim=0)
    dataset = TensorDataset(X_train)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Try different bottleneck sizes
    results = []
    for latent_dim in [5, 8, 10, 15, 20, 25]:
        # Simple symmetric autoencoder
        encoder = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, n_features),
        )
        model = nn.Sequential(encoder, decoder)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        # Train
        model.train()
        for epoch in range(300):
            for (batch,) in loader:
                optimizer.zero_grad()
                recon = model(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                optimizer.step()

        # Evaluate on real presets only
        model.eval()
        with torch.no_grad():
            recon = model(X_tensor)
            mse = float(loss_fn(recon, X_tensor))
            # Per-preset max error
            per_preset = torch.mean((recon - X_tensor) ** 2, dim=1)
            worst = float(torch.max(per_preset))
            median = float(torch.median(per_preset))

        results.append({
            "latent_dim": latent_dim,
            "mse": mse,
            "worst": worst,
            "median": median,
        })
        print(f"  latent_dim={latent_dim:>3}: MSE={mse:.4f}  worst={worst:.4f}  median={median:.4f}")

    # Compare to PCA
    print(f"\n  PCA comparison (same MSE metric on standardized data):")
    for n_comp in [5, 8, 10, 15, 20, 25]:
        pca = PCA(n_components=n_comp)
        X_pca = pca.fit_transform(X_scaled)
        X_recon = pca.inverse_transform(X_pca)
        mse = float(np.mean((X_scaled - X_recon) ** 2))
        per_preset = np.mean((X_scaled - X_recon) ** 2, axis=1)
        worst = float(np.max(per_preset))
        median = float(np.median(per_preset))
        print(f"  PCA n_comp={n_comp:>3}: MSE={mse:.4f}  worst={worst:.4f}  median={median:.4f}")

    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    ae_dims = [r["latent_dim"] for r in results]
    ae_mse = [r["mse"] for r in results]
    ax.plot(ae_dims, ae_mse, "ro-", label="Autoencoder", markersize=8)

    pca_mse = []
    for n in ae_dims:
        p = PCA(n_components=n)
        Xp = p.fit_transform(X_scaled)
        Xr = p.inverse_transform(Xp)
        pca_mse.append(float(np.mean((X_scaled - Xr) ** 2)))
    ax.plot(ae_dims, pca_mse, "bs-", label="PCA", markersize=8)

    ax.set_xlabel("Latent / Component Dimensions")
    ax.set_ylabel("Reconstruction MSE")
    ax.set_title("Autoencoder vs PCA: Reconstruction Quality")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "autoencoder_vs_pca.png", dpi=150)
    plt.close()


# ── 5. Perceptual Distance Optimization ──────────────────────────────────────


def run_perceptual_optimization(presets, defaults):
    """Optimize macro derivation functions to minimize perceptual error."""
    print(f"\n{'='*80}")
    print("5. PERCEPTUAL DISTANCE OPTIMIZATION")
    print(f"{'='*80}")

    try:
        from lossy_rust import render_lossy
    except ImportError:
        print("  SKIP: lossy_rust not available")
        return

    from scipy.optimize import minimize

    test_signal = make_test_signal()

    # Pre-render originals
    originals = {}
    for name, preset in presets:
        full = {**defaults, **{k: v for k, v in preset.items()
                if k != "_meta" and k != "tail_length"}}
        try:
            out = render_lossy(test_signal, json.dumps(full))
            rms = float(np.sqrt(np.mean(out ** 2)))
            if rms > 1e-15:
                originals[name] = (full, out, rms)
        except Exception:
            pass
    print(f"  Pre-rendered {len(originals)} presets")

    # Fixed params from experiment E
    FIXED_KEYS = ["seed", "pre_echo", "transient_ratio", "noise_shape",
                   "slushy_rate", "freeze_mode", "verb_position", "global_amount"]

    # We optimize the derivation coefficients for decay and filter_slope
    # x = [decay_slope, filter_slope_slope]
    # decay = 0.5 + x[0] * verb
    # filter_slope = round(1 + x[1] * (filter_width - 0.5))

    def objective(x):
        decay_slope, fs_slope = x
        total_err = 0.0
        count = 0
        for name, (full, out_orig, rms_orig) in originals.items():
            recon = full.copy()
            # Fixed
            for k in FIXED_KEYS:
                recon[k] = defaults[k]
            # Derived
            verb_val = float(full.get("verb", 0.0))
            if verb_val > 1e-6:
                recon["decay"] = max(0.0, min(1.0, 0.5 + decay_slope * verb_val))
            else:
                recon["decay"] = defaults["decay"]
            fw = float(full.get("filter_width", 0.5))
            recon["filter_slope"] = int(round(max(0, min(2, 1 + fs_slope * (fw - 0.5)))))
            recon_clean = {k: v for k, v in recon.items()
                          if k != "_meta" and k != "tail_length"}
            try:
                out_recon = render_lossy(test_signal, json.dumps(recon_clean))
                ml = min(len(out_orig), len(out_recon))
                diff = float(np.sqrt(np.mean((out_orig[:ml] - out_recon[:ml]) ** 2)))
                total_err += diff / rms_orig
                count += 1
            except Exception:
                pass
        return total_err / max(count, 1)

    # Initial values from PCA fits
    x0 = np.array([0.517, -2.612])

    print(f"\n  Initial coefficients: decay_slope={x0[0]:.3f}, filter_slope_slope={x0[1]:.3f}")
    init_err = objective(x0)
    print(f"  Initial mean perceptual error: {init_err:.4f}")

    # Optimize with Nelder-Mead (derivative-free, handles noisy objective)
    print(f"  Optimizing (this takes a while — each eval renders {len(originals)} presets)...")
    t0 = time.time()
    n_evals = [0]
    def callback(xk):
        n_evals[0] += 1
        if n_evals[0] % 5 == 0:
            print(f"    Iteration {n_evals[0]}: err={objective(xk):.4f} x={xk}")

    result = minimize(objective, x0, method="Nelder-Mead",
                      options={"maxiter": 100, "xatol": 0.01, "fatol": 0.001},
                      callback=callback)
    dt = time.time() - t0

    print(f"\n  Optimization done in {dt:.1f}s")
    print(f"  Optimized coefficients: decay_slope={result.x[0]:.4f}, filter_slope_slope={result.x[1]:.4f}")
    print(f"  Optimized mean perceptual error: {result.fun:.4f}")
    print(f"  Improvement: {(1 - result.fun/init_err)*100:.1f}%")

    # Now try optimizing with MORE derivation pairs
    # Add: decimate_slope (from crush), packet_size_slope (from packet_rate)
    # x = [decay_slope, fs_slope, decimate_slope, pktsize_slope]

    def objective_extended(x):
        decay_slope, fs_slope, dec_slope, pkt_slope = x
        total_err = 0.0
        count = 0
        for name, (full, out_orig, rms_orig) in originals.items():
            recon = full.copy()
            for k in FIXED_KEYS:
                recon[k] = defaults[k]
            # decay from verb
            verb_val = float(full.get("verb", 0.0))
            if verb_val > 1e-6:
                recon["decay"] = max(0.0, min(1.0, 0.5 + decay_slope * verb_val))
            else:
                recon["decay"] = defaults["decay"]
            # filter_slope from filter_width
            fw = float(full.get("filter_width", 0.5))
            recon["filter_slope"] = int(round(max(0, min(2, 1 + fs_slope * (fw - 0.5)))))
            # decimate from crush (gated)
            crush_val = float(full.get("crush", 0.0))
            if crush_val > 1e-6:
                recon["decimate"] = max(0.0, min(1.0, dec_slope * crush_val))
            else:
                recon["decimate"] = defaults["decimate"]
            # packet_size from packet_rate (gated on packets)
            if int(full.get("packets", 0)) > 0:
                pr = float(full.get("packet_rate", 0.3))
                recon["packet_size"] = max(5.0, min(200.0, 30.0 + pkt_slope * (pr - 0.3)))

            recon_clean = {k: v for k, v in recon.items()
                          if k != "_meta" and k != "tail_length"}
            try:
                out_recon = render_lossy(test_signal, json.dumps(recon_clean))
                ml = min(len(out_orig), len(out_recon))
                diff = float(np.sqrt(np.mean((out_orig[:ml] - out_recon[:ml]) ** 2)))
                total_err += diff / rms_orig
                count += 1
            except Exception:
                pass
        return total_err / max(count, 1)

    x0_ext = np.array([result.x[0], result.x[1], 0.653, 12.693])  # PCA anchored fits
    init_err_ext = objective_extended(x0_ext)
    print(f"\n  Extended model (4 derivations): initial error={init_err_ext:.4f}")

    result_ext = minimize(objective_extended, x0_ext, method="Nelder-Mead",
                          options={"maxiter": 100, "xatol": 0.01, "fatol": 0.001})
    print(f"  Optimized extended: error={result_ext.fun:.4f}")
    print(f"  Coefficients: decay={result_ext.x[0]:.3f}, filter_slope={result_ext.x[1]:.3f}, "
          f"decimate={result_ext.x[2]:.3f}, packet_size={result_ext.x[3]:.3f}")
    print(f"  Improvement over PCA fits: {(1 - result_ext.fun/init_err_ext)*100:.1f}%")

    # Compare: E (2 derivations) vs extended (4 derivations) vs no derivations
    no_deriv_err = 0.0
    count = 0
    for name, (full, out_orig, rms_orig) in originals.items():
        recon = full.copy()
        for k in FIXED_KEYS:
            recon[k] = defaults[k]
        recon_clean = {k: v for k, v in recon.items()
                      if k != "_meta" and k != "tail_length"}
        try:
            out_recon = render_lossy(test_signal, json.dumps(recon_clean))
            ml = min(len(out_orig), len(out_recon))
            diff = float(np.sqrt(np.mean((out_orig[:ml] - out_recon[:ml]) ** 2)))
            no_deriv_err += diff / rms_orig
            count += 1
        except Exception:
            pass
    no_deriv_err /= max(count, 1)

    print(f"\n  Comparison (mean perceptual error):")
    print(f"    No derivations (fix 8 only):  {no_deriv_err:.4f}")
    print(f"    Experiment E (2 PCA fits):     {init_err:.4f}")
    print(f"    E optimized (2 perceptual):    {result.fun:.4f}")
    print(f"    Extended (4 PCA fits):          {init_err_ext:.4f}")
    print(f"    Extended optimized (4 percep): {result_ext.fun:.4f}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    defaults = get_defaults()
    feature_names = build_feature_names()

    presets = load_presets()
    print(f"Loaded {len(presets)} presets")

    X = np.array([preset_to_features(p, defaults) for _, p in presets])
    names = [n for n, _ in presets]
    print(f"Feature matrix: {X.shape}")

    # 1. Sparse PCA
    run_sparse_pca(X, feature_names)

    # 2. Factor Analysis
    high_noise = run_factor_analysis(X, feature_names)

    # 3. Sobol Sensitivity
    sobol_ST = run_sobol(defaults)

    # 4. Autoencoder
    run_autoencoder(X, feature_names)

    # 5. Perceptual Distance Optimization
    run_perceptual_optimization(presets, defaults)

    # ── Summary ──
    print(f"\n{'='*80}")
    print("SUMMARY: Method Comparison")
    print(f"{'='*80}")

    print(f"\n  Standard PCA: 22 PCs for 90%, 27 for 95% (parameter space)")
    print(f"  Perceptual PCA: 8 PCs for 90% (from lossy_pca.py)")

    if high_noise:
        print(f"\n  Factor Analysis high-noise params (idiosyncratic variation, not shared factors):")
        print(f"    {high_noise}")
        print(f"    These vary across presets but NOT as part of shared patterns — consider fixing")

    if sobol_ST:
        dead = [k for k, v in sobol_ST.items() if v < 0.01]
        important = sorted(sobol_ST.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"\n  Sobol dead params (ST < 0.01 — no effect on output):")
        print(f"    {dead}")
        print(f"\n  Sobol top 10 most important params (including interactions):")
        for k, v in important:
            print(f"    {k:<20} ST={v:.4f}")

    print(f"\n  Autoencoder vs PCA: see {OUT_DIR / 'autoencoder_vs_pca.png'}")
    print(f"  Perceptual optimization: see results above for optimized derivation coefficients")

    print(f"\n{'='*80}")
    print(f"DONE — all plots in {OUT_DIR}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
