# PCA for Audio Effect Parameter Reduction

A method for empirically discovering the effective dimensionality of an audio effect's parameter space, then designing macro controls aligned to the principal components.

## Why

Complex audio effects accumulate parameters over time. An 8-node FDN reverb has 56+ individual values. But hand-designed presets only explore a fraction of that space — damping coefficients are usually uniform across nodes, input gains are always 1/8, etc. PCA reveals which dimensions actually matter.

**Benefits of a reduced parameter space:**
- **LLM tuning** — an LLM can reason about 14 intuitive controls far better than 56 raw values. Fewer tokens in the system prompt, fewer hallucinated values, faster convergence.
- **GUI** — users see knobs that map to perceptual qualities (size, brightness, decay) instead of implementation details (damping_coeffs[3]).
- **Preset interpolation** — you can smoothly morph between presets in the macro space. Interpolating raw params often produces nonsense (e.g., blending matrix types, averaging per-node arrays with different structures).
- **Automation** — DAW automation lanes for 14 params instead of 56.

## The Method

### 1. Build the Feature Matrix

Load all presets into a matrix X where each row is a preset and each column is a parameter feature.

**Flattening rules:**
- **Per-element arrays** (e.g., `delay_times[8]`) → 8 individual columns
- **Categorical params** (e.g., `matrix_type`) → one-hot encoding, one column per category
- **Scalars** → one column each
- **Normalize units** — convert samples to ms, frequencies to Hz, etc. so magnitudes are comparable before standardization
- **Fill missing values** from defaults — presets are typically sparse JSON with `serde(default)`

**What to exclude initially:**
- Conditional subsystems used by <20% of presets (e.g., modulation). Analyze these separately.
- Metadata fields (`_meta`, `name`, etc.)
- Params that are always constant across all presets (zero variance → PCA drops them anyway, but they add noise to the feature count)

### 2. Pre-transform Skewed Parameters

PCA is linear. If a parameter has an exponential perceptual mapping (e.g., delay times, frequencies), log-transform it before standardization. Otherwise PCA will over-weight the high end of the range.

**Common transforms:**
- `delay_times` → `log(delay_ms)` (doubling delay = one "unit" of size increase)
- `filter_freq` → `log(freq_hz)` (octave-based perception)
- `feedback_gain` → leave linear (already 0–1ish range)
- Categorical → one-hot (no transform needed)

If you're unsure whether to transform, run PCA both ways and compare. If log-transform produces cleaner loadings (fewer PCs for same variance), use it.

### 3. Standardize

Zero mean, unit variance (StandardScaler). Required because parameters live on different scales — delay times in ms vs. damping coefficients 0–1. Without standardization, high-magnitude features dominate the principal components.

### 3. Run PCA

```
PCA on standardized X → eigenvalues, eigenvectors (loadings)
```

**Key outputs:**
- **Cumulative explained variance** — how many components to reach 90%, 95%, 99%
- **Scree plot** — eigenvalue magnitude vs. component index (look for the "elbow")
- **Loading vectors** — which original parameters contribute most to each PC

### 4. Interpret the Components

For each top PC, look at its loading vector (the eigenvector weights):

| Pattern | Interpretation |
|---|---|
| All elements of an array load together with similar sign/magnitude | That array is effectively a single scalar. Create one macro that sets all elements uniformly. |
| Array elements load with different signs | There's meaningful per-element variation. Consider keeping the array or finding a parameterization (e.g., "tilt" = low-to-high gradient). |
| A scalar loads alone on a PC | It's already an independent control. Keep as-is. |
| Multiple params load together on the same PC | They co-vary — consider a single macro that drives all of them. |
| A param doesn't load significantly on any top PC | It's either constant across presets (eliminate) or only relevant to outliers (move to "advanced" layer). |

### 5. Validate with Array Summary Statistics

Independently of PCA, compute per-array stats across all presets:

- **Coefficient of variation** (std/mean) per preset, then count how many presets have CV < 0.1
- If >60% of presets have near-zero CV for an array, it's effectively a scalar → replace with single control

This was the key finding for the reverb: `input_gains` and `output_gains` had CV < 0.1 in 71% of presets, `damping_coeffs` in 58%.

### 6. Augment with Random Parameter Sampling

Presets only cover the explored region of the parameter space. To test whether the dimensionality reduction holds broadly, augment with randomly sampled parameter sets:

- **Uniform random** — sample each parameter independently from its valid range. This explores corners of the space that no preset touches.
- **Perturbed presets** — take each preset, add Gaussian noise (e.g., 10-20% of range) to each parameter. Tests sensitivity near known-good regions.
- **Sample count** — 200-500 random sets is usually enough to stabilize the PCA eigenvalues.

**What to look for:**
- If random sampling requires significantly more PCs for 90% variance than presets alone, the preset bank is clustered in a low-dimensional submanifold and the macro controls may not generalize to unusual parameter combinations. This is fine — the macros target the "useful" subspace.
- If the number of PCs is similar, the parameter space is inherently low-dimensional regardless of where you sample.

Run PCA on three groups: presets-only, random-only, combined. Compare the explained variance curves and loadings.

### 7. Separate Analysis by Preset Category

Run PCA on subsets:
- **Hand-designed only** — represents the "intended" parameter space
- **ML-generated only** — may exploit unused dimensions
- **All presets** — baseline

If dimensionality drops significantly for hand-designed presets (it usually does), the macro controls should target that subspace. ML-generated presets that need the full space can bypass macros.

For the reverb: all presets needed 16 PCs for 90%, but hand-designed needed only 8.

### 8. Cluster Analysis

Run k-means on the PCA-reduced space (using enough PCs for 95% variance). Use silhouette score to pick k.

This reveals:
- Whether there are natural preset "families" (e.g., rooms vs. plates vs. ambient)
- Outlier presets that don't fit the simplified model
- Whether the macro space has a clean topology or disjoint regions

### 9. Identify Dead Parameters

Some parameters may exist in presets but have negligible effect on the output. Find these before designing macros — they're candidates for elimination.

**Variance analysis (from the feature matrix):**
- Compute variance of each feature column across all presets
- Near-zero variance = parameter is constant (or nearly so) across all presets
- These are safe to fix at their default value

**Loading analysis (from PCA):**
- Parameters that don't load significantly on any of the top PCs explaining 95% variance are effectively unused
- They either don't vary across presets, or their variation is orthogonal to the meaningful dimensions

**Sensitivity analysis (requires DSP):**
- For each parameter, render a reference preset, then sweep the parameter across its range while holding others fixed
- Measure output change (RMS difference, spectral distance, or perceptual metric)
- If sweeping a parameter across its full range produces <1% change in output, it's perceptually dead for that preset
- Repeat across multiple presets to check if it's globally dead or just dead in certain regions

**Common dead parameter patterns:**
- Per-node arrays where every preset uses the same values (e.g., `input_gains` always 1/8)
- Conditional params that are irrelevant when their parent is off (e.g., `mod_depth_*` when `mod_master_rate = 0`)
- Seeds and internal config that don't affect the typical matrix type
- Rate scaling params that are always 1.0

### 10. Design Macro Controls

Map each significant PC to an intuitive control:

**Direct mappings** (PC loads on a single original param):
- Keep the param as-is. Rename if the original name is too technical (e.g., `feedback_gain` → `decay`).

**Array-to-scalar** (PC loads uniformly on all elements of an array):
- Single scalar that sets all elements to the same value.
- Mapping can be linear or nonlinear (e.g., `brightness = 1 - damping`, exponential for delay times).

**Co-varying groups** (PC loads on multiple params):
- Single macro with a defined mapping to each underlying param.
- Example: `size` → delay_times (exponential) + diffusion_delays (proportional).

**Template-based arrays** (array values vary but maintain fixed ratios):
- Single scalar controls the "base" value; per-element values = base * ratio_template.
- Detect this by checking if ratios are consistent across presets.

**Eliminated params** (near-zero variance, no significant loading):
- Fix to default value. Move to an "advanced" panel if needed.

### 11. Encode Parameter Constraints

Some parameters interact — the simplified mapping must preserve these constraints.

**Common patterns:**
- **Safety constraints** — `feedback_gain > 1.0` requires `saturation > 0` to prevent explosion. The forward mapping should enforce this.
- **Conditional subsystems** — modulation params are meaningless when `mod_rate = 0`. The forward mapping should zero them out.
- **Derived relationships** — `node_pans` derived from `stereo_width`, `diffusion_delays` derived from `size`. Don't expose both.
- **Clamping** — the simplified param ranges should make it impossible to produce out-of-range full params.

Encode these in the forward mapping (`to_full_params`), not as separate validation. The simplified params should be safe by construction.

### 12. Implement the Mapping Layer (Rust)

The mapping layer lives in the Rust DSP crate alongside the processor — not in Python. The PCA analysis script is a throwaway Python tool, but the resulting `ReverbParams` struct and its mappings are Rust code.

```
rust/crates/<plugin>-dsp/src/simplified.rs

ReverbParams::to_<plugin>_params(&self) -> <Plugin>Params   // forward
ReverbParams::from_<plugin>_params(p: &<Plugin>Params) -> Self  // reverse (lossy)
```

**Forward mapping** (simplified → full): deterministic, lossless within the macro subspace. This is what the DSP consumes.

**Reverse mapping** (full → simplified): best-fit approximation. Lossy by design — per-node variation in eliminated params is discarded. Needed for converting existing presets to the simplified representation.

Both mappings get `#[serde(default)]` for sparse JSON support and full unit test coverage.

**Key constraint:** The DSP code (`processor.rs`, `fdn.rs`, etc.) never changes. The mapping layer sits above it in the same crate.

### 13. Validate Reconstruction

For each preset:
1. Convert full → simplified (reverse)
2. Convert simplified → full (forward)
3. Compare original vs. reconstructed full params

**Metrics:**
- Per-array relative error: `mean(|orig - recon|) / mean(|orig|)`
- Per-scalar absolute error
- Categorical match (matrix type, waveform)

**Expected results:**
- Hand-designed presets: <10% error for most, <20% for nearly all
- ML-generated / experimental: higher error is expected and acceptable
- Any hand-designed preset with >20% error is either an outlier that exploits the full space, or the macro mapping needs adjustment

### 14. Validate Perceptually (Audio Comparison)

Parameter-level reconstruction error doesn't tell you if it *sounds* the same. Two parameter sets can differ numerically but produce nearly identical audio, or match closely in params but sound different due to nonlinear interactions.

**Method:**
1. Render a test signal (impulse, white noise, or music) through the DSP with the original full params
2. Render the same signal with the reconstructed params (simplified → full → DSP)
3. Compare the two outputs

**Metrics:**
- **RMS difference** — simple amplitude error. Normalize both outputs first.
- **Spectral similarity** — compare magnitude spectra (e.g., log-spectral distance, mel-frequency cepstral distance)
- **RT60 match** — for reverbs, compare decay times across frequency bands
- **Spectrogram difference** — subtract spectrograms, check if residual is perceptually negligible
- **PESQ / ViSQOL** — perceptual quality metrics if available, but usually overkill for this purpose

**Implementation:**
- Use the Rust DSP directly (call `render_fdn` or equivalent from a test harness)
- Render an impulse response (1 sample at 1.0, then silence) — captures the full reverb character in one pass
- For effects like lossy/fractal, use a short music clip instead since they're input-dependent

**Pass criteria:**
- Spectral similarity > 0.95 for hand-designed presets
- RT60 within 10% per frequency band
- RMS difference < -40dB (inaudible)

This is the ground truth validation. If parameter error is high but audio error is low, the macro mapping is fine — the "lost" parameters didn't matter perceptually.

### 15. Explore the Unmapped Space

After validating that the macro controls reproduce existing presets, investigate the parameter dimensions they *can't* reach. This answers: "are we losing any musically interesting sounds?"

**Method:**
1. Identify the locked-out dimensions — per-node variation in arrays that the macros set uniformly (e.g., non-uniform damping, asymmetric input/output gains, broken delay time ratios)
2. For each dimension, generate parameter sets that maximize variation in that dimension while keeping other params in a reasonable range
3. Render each parameter set through the DSP as an impulse response
4. Also render the closest macro-reachable version (round-trip through `from_fdn_params` → `to_fdn_params`)
5. Compare the two outputs: RT60, spectral centroid, energy envelope correlation
6. **Output WAV files** for human listening — metrics can miss perceptual differences that matter musically

**Key experiments:**
- Non-uniform damping (gradient, alternating, extreme spread, frequency-dependent)
- Non-uniform gains (crescendo, alternating emphasis, single-node solo)
- Broken delay ratios (prime numbers, extreme spread, clustered, single very long delay, detuned unison, comb filter)
- Combined extremes

**Expected results:**
Most experiments will show high envelope correlation (>0.90) — the macro-reachable version sounds similar. A few may reveal genuinely different textures:
- Extreme delay spread (one very short + one very long delay)
- Single-node comb filter effects
- These are niche sounds. If needed, they can be accessed by writing `FdnParams` directly, bypassing the macro layer.

**Implementation:** See `scripts/explore_unmapped_space.py` — generates WAV pairs (`*_full.wav` vs `*_macro.wav`) for each experiment.

## Applying to Other Plugins

### Lossy

Parameters: ~30 (spectral loss, crush, packets, filter, effects). Likely candidates for reduction:
- `window_size` is categorical in practice (powers of 2)
- Filter params (`freq`, `width`, `slope`) might co-vary with `loss` level
- Packet params are conditional (only active when `packets != 0`)

### Fractal

Parameters: ~30 (core fractal, iteration, spectral, filters, layers, bounce). Likely candidates:
- `layer_gain_1..7` — 7 values that may reduce to a "density" + "tilt" pair
- Core fractal params (`num_scales`, `scale_ratio`, `amplitude_decay`) likely co-vary
- Bounce params are conditional

### General Checklist

**Analysis (Python — throwaway scripts):**
1. Write PCA script: `scripts/<plugin>_pca.py` — load presets, flatten, standardize, PCA
2. Log-transform skewed params (delays, frequencies) before standardization
3. Compute array CV stats, variance per feature, scree plot, loadings
4. Augment with random parameter samples, compare explained variance
5. Run separately on hand-designed vs. ML-generated subsets
6. Identify dead parameters (zero variance, no loading, no perceptual effect)
7. Identify macro controls from top PCs

**Implementation (Rust):**
8. Implement `<Plugin>Params` (the macro controls) in `rust/crates/<plugin>-dsp/src/simplified.rs`
9. Forward mapping: `to_<plugin>_params()` with safety constraints baked in
10. Reverse mapping: `from_<plugin>_params()` (lossy best-fit)
11. `#[serde(default)]` for sparse JSON, unit tests for roundtrip, extremes, constraints
12. Add `pub mod simplified` + re-export to `lib.rs`

**Validation:**
13. Parameter reconstruction: < 10% relative error for hand-designed presets
14. Audio comparison: render original vs. reconstructed through DSP, compare spectra/RT60
15. `cargo test -p <plugin>-dsp` passes — DSP unchanged

## Shared Tooling

The PCA analysis is Python (sklearn), the DSP and simplified params are Rust. For the audio validation step, the Rust DSP is called from Python via PyO3 bindings (maturin). This is already how the project works — each plugin has a `<plugin>-python` crate with PyO3 bindings.

A shared Python utility for the analysis scripts could be useful:

```
scripts/
  pca_common.py          — shared helpers: load presets, flatten, array stats, PCA runner
  reverb_pca.py          — reverb-specific feature extraction + analysis
  lossy_pca.py           — lossy-specific
  fractal_pca.py         — fractal-specific
  reverb_macro_validate.py   — reverb reconstruction + audio comparison
```

The `pca_common.py` module would provide:
- `load_presets(preset_dir)` — load all JSON presets
- `flatten_arrays(preset, defaults, array_keys, scalar_keys, categorical_keys)` — generic flattening
- `array_cv_stats(presets, defaults, array_keys)` — CV analysis
- `run_pca_analysis(X, labels, feature_names, title)` — PCA + scree + loadings + clusters
- `compare_audio(render_fn, original_params, reconstructed_params, test_signal)` — render both, compute spectral distance

The Rust `simplified.rs` modules are per-plugin — no shared crate needed since each plugin's mapping logic is unique.

## Reference

- Reverb PCA script: `scripts/reverb_pca.py` (throwaway analysis)
- Reverb validation script: `scripts/reverb_macro_validate.py` (throwaway analysis)
- Reverb simplified params: `rust/crates/reverb-dsp/src/simplified.rs`
