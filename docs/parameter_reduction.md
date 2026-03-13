# Audio Effect Parameter Reduction

A methodology for empirically discovering the effective dimensionality of an audio effect's parameter space, then designing macro controls that capture the perceptually meaningful dimensions.

## Why

Complex audio effects accumulate parameters over time. An 8-node FDN reverb has 56+ individual values, a spectral processor might have 30+, a fractal generator 30+. But hand-designed presets only explore a fraction of that space — per-node arrays are often uniform, conditional subsystems are usually off, etc. The core analytical tool here — Principal Component Analysis — reveals which dimensions actually matter.

**Benefits of a reduced parameter space:**
- **LLM tuning** — an LLM can reason about 10–15 intuitive controls far better than 30–60 raw values. Fewer tokens in the system prompt, fewer hallucinated values, faster convergence.
- **GUI** — users see knobs that map to perceptual qualities (size, brightness, decay, density) instead of implementation details (damping_coeffs[3]).
- **Preset interpolation** — you can smoothly morph between presets in the macro space. Interpolating raw params often produces nonsense (e.g., blending matrix types, averaging per-node arrays with different structures).
- **Automation** — DAW automation lanes for 14 params instead of 56. Smooth macro changes → smooth audio changes (if the mapping is well-designed).

## How PCA Works (Intuition)

Principal Component Analysis finds the directions in parameter space along which your data varies most.

**The geometric picture:** Imagine each preset as a point in a 56-dimensional space (one axis per parameter). If you plotted all your presets, they wouldn't fill the full 56D volume — they'd cluster along a lower-dimensional shape, like a cloud of points that's wide in some directions and flat in others. PCA finds the axes of that cloud, ordered by how "wide" the cloud is along each axis.

**Step by step:**
1. **Center the data** — subtract the mean of each parameter so the cloud sits at the origin
2. **Compute the covariance matrix** — a matrix measuring how every pair of parameters co-varies across presets. If delay_time and feedback always increase together, their covariance is high.
3. **Eigendecomposition** — decompose the covariance matrix into eigenvectors (directions) and eigenvalues (magnitudes). The eigenvector with the largest eigenvalue is the direction of maximum variance — the "widest" axis of the cloud. The second eigenvector is the widest direction *perpendicular* to the first. And so on.
4. **Rank and truncate** — the eigenvalues tell you how much variance each direction captures. If the first 8 directions capture 95% of total variance, the other 48 dimensions are near-flat — you can ignore them without losing much.

**What each principal component means in audio terms:**
- **PC1 might be "size"** — delay times, diffusion delays, and feedback all increase together (large eigenvalue → presets vary a lot along this direction)
- **PC2 might be "brightness"** — damping and filter frequency vary together, orthogonal to size
- **PC8 might be "input gain balance"** — tiny eigenvalue → presets barely vary here → safe to lock to a default

**Key limitations to understand:**
- **PCA is linear.** It finds flat planes through the data. If the "interesting" region of parameter space curves (a nonlinear manifold), PCA can miss structure or need extra dimensions to approximate it. This is why we pre-transform skewed parameters (step 3) — a log transform can straighten a curved relationship.
- **PCA operates in parameter space, not perceptual space.** Two parameter sets that are "close" in PCA space can sound very different if the DSP has nonlinear behavior (e.g., feedback gain 0.95 vs 0.99). This is why the methodology includes a separate perceptual-space PCA (step 10) as a companion analysis.
- **Explained variance ≠ perceptual importance.** A parameter accounting for 1% of variance might be the difference between "metallic" and "natural." The variance thresholds (90%, 95%) are starting points — perceptual validation (step 17) is the real arbiter.

## How to Use This Doc

This is a sequential methodology. **Do not skip to implementation (step 12+) without completing the analysis (steps 1–11) first.** The analysis results drive the design — without them you're guessing.

When applying this to a new plugin:
1. Read the "Specializing to Your Effect" section first — it tells you what to think about before starting
2. Create tasks to track each step
3. Write and run the PCA analysis script (Python)
4. Review the results with the user (scree plot, loadings, variance, clusters)
5. Only then design macro controls based on what the data shows
6. Only then implement the Rust mapping layer

**Iteration is expected.** After perceptual validation (step 16), you may need to loop back to step 12 (redesign macros). The first pass rarely nails it.

## Specializing to Your Effect

Before starting the method, think through these questions for your specific effect. The answers shape every subsequent step — transforms, encoding, validation metrics, and what "perceptually correct" means.

### Parameter Behavior

**Which parameters have nonlinear perceptual mappings?**
Any parameter where equal numeric steps don't produce equal perceptual steps needs a transform before PCA. Common patterns:
- **Delay times / frequencies** → logarithmic (octave-based perception)
- **Feedback / recursion gains** → highly nonlinear near 1.0 (e.g., RT60 ∝ -1/ln(gain), so 0.95→0.99 is a massive perceptual change while 0.3→0.5 is subtle). Transform to the perceptual quantity (e.g., RT60, decay time) or use -log(1 - gain).
- **Exponents / power laws** → may need log or reciprocal transform
- **Thresholds with cliff behavior** — a parameter that does nothing until it crosses a threshold, then has large effect. PCA can't model this; flag it for special handling.

If unsure, run PCA both ways (raw vs. transformed) and compare. Cleaner loadings = better transform.

**Which parameters are categorical?**
PCA assumes continuous variables. One-hot encoding is the naive approach but it's problematic — PCA treats all categories as equidistant, which is usually wrong (e.g., Hadamard and Householder matrices both produce dense diffusion, while Identity produces distinct comb filtering).

Better options:
- **Encode by DSP properties** — replace the category with measured continuous features (e.g., mixing density, energy scaling, spectral coloration). This preserves meaningful distance between categories.
- **FAMD / MCA** — Factor Analysis of Mixed Data or Multiple Correspondence Analysis handle mixed continuous/categorical data natively.
- **If one-hot is unavoidable** — note that the categorical PCs should be interpreted with caution. Don't design continuous macro mappings through categorical space.

**Which parameters interact nonlinearly in the DSP?**
PCA captures linear covariance in parameter space, not in audio space. Two parameter sets that are "close" in PCA space can sound completely different if there are nonlinear interactions. Flag these pairs — they need extra perceptual validation:
- Feedback gain + saturation (saturation prevents explosion but fundamentally changes the sound)
- Delay times + feedback (short delays + high feedback = metallic resonance)
- Any parameter that changes system stability characteristics

### System Linearity

**Is the effect linear (input-independent) or nonlinear?**
This determines your validation strategy:
- **Linear effects** (basic reverb without saturation, linear filters, convolution): Impulse responses fully characterize the system. One IR comparison = complete validation.
- **Nonlinear effects** (saturation, waveshaping, dynamics, bitcrushing, clipping): Behavior depends on input level and content. You must validate with multiple test signals at multiple levels. An impulse response alone is insufficient — it may not exercise the nonlinear path.
- **Mixed**: Many effects are "mostly linear with nonlinear safety mechanisms" (e.g., reverb with saturation in the feedback loop). Test with both impulses *and* program material.

### Perceptual Metrics

**What does "sounds the same" mean for this effect?**
Choose validation metrics appropriate to the effect type:
- **Reverbs**: RT60 per frequency band, EDT, C80, spectral centroid over time, energy decay curve correlation
- **Spectral processors** (lossy, vocoders): Spectral envelope similarity, mel-cepstral distance, transient preservation
- **Fractal / generative effects**: Statistical similarity (spectral moments, amplitude distribution), since outputs may not be deterministic
- **Dynamics processors**: Gain reduction curve, attack/release timing, harmonic distortion
- **Time-varying effects** (chorus, flanger, phaser): Modulation depth and rate in the output spectrum

### Quantization and Discretization

**Are any continuous parameters discretized in practice?**
- **Delay times** are quantized to integer samples. PCA on continuous values might suggest delay times whose quantized versions collide (two delays rounding to the same sample count → comb filtering instead of diffusion).
- **Window sizes** may be restricted to powers of 2.
- **Array lengths** are integers.

Work in the quantized space, or validate that quantization doesn't break reconstruction.

### Conditional Subsystems

**Which parameters are only meaningful when a parent parameter is active?**
- Modulation depth/rate/shape when modulation is off
- Filter resonance when filter is bypassed
- Effect-specific sub-modes

Exclude these from the main PCA. Analyze them separately per-subsystem, conditioned on the parent being active. They become their own mini-macro groups.

## The Method

### 1. Identify and Remove Dead Parameters

Do this *before* PCA — it reduces dimensionality and makes PCA more reliable.

**Variance analysis (from the raw preset data):**
- Compute variance of each parameter across all presets
- Near-zero variance = parameter is constant (or nearly so) across all presets
- These are safe to fix at their default value

**Conditional dead parameters:**
- Params irrelevant when their parent is off (e.g., `mod_depth` when `mod_rate = 0`)
- Seeds and internal config that don't affect typical use
- Rate scaling params that are always 1.0

**Sensitivity analysis (if DSP is available):**
- For each parameter, render a reference preset, sweep the parameter across its range while holding others fixed
- Measure output change (RMS difference, spectral distance)
- If sweeping across the full range produces <1% change in output, it's perceptually dead for that preset
- Repeat across multiple presets to check if it's globally dead or just dead in certain regions

Record which parameters you removed and why. Some "dead" parameters may turn out to matter for specific sounds — the advanced panel can expose them.

### 2. Build the Feature Matrix

Load all presets into a matrix X where each row is a preset and each column is a parameter feature.

**Flattening rules:**
- **Per-element arrays** (e.g., `delay_times[8]`) → 8 individual columns
- **Categorical params** → encode by DSP properties (preferred) or one-hot (see "Specializing" section above)
- **Scalars** → one column each
- **Normalize units** — convert samples to ms, frequencies to Hz, etc. so magnitudes are comparable before standardization
- **Fill missing values** from defaults — presets are typically sparse JSON with `serde(default)`

**What to exclude:**
- Dead parameters identified in step 1
- Conditional subsystems used by <20% of presets (analyze separately)
- Metadata fields (`_meta`, `name`, etc.)

### 3. Pre-transform Skewed Parameters

PCA is linear. If a parameter has a nonlinear perceptual mapping, transform it before standardization. Otherwise PCA will over-weight one end of the range.

Refer to the "Parameter Behavior" section above for your effect's specific transforms. Common patterns:

| Parameter type | Transform | Rationale |
|---|---|---|
| Delay times | log(ms) | Doubling delay = one "unit" of size increase |
| Frequencies | log(Hz) | Octave-based perception |
| Feedback/recursion gains | -log(1 - gain) or map to decay time | Extreme nonlinearity near 1.0 |
| Bounded 0–1 scalars | Often leave linear | Already perceptually scaled (check this) |
| Power-law params | log | Compress range |

If unsure, run PCA both ways and compare. Cleaner loadings (fewer PCs for same variance) = better transform.

### 4. Standardize

Zero mean, unit variance (StandardScaler). Required because parameters live on different scales — delay times in ms vs. damping coefficients 0–1. Without standardization, high-magnitude features dominate the principal components.

**Note:** StandardScaler assumes roughly Gaussian distributions. For heavily bounded or skewed parameters (even after transforms), consider robust scaling (median/IQR) or min-max scaling. In practice, StandardScaler usually works fine after the log transforms in step 3.

### 5. Run PCA

```
PCA on standardized X → eigenvalues, eigenvectors (loadings)
```

**Key outputs:**
- **Cumulative explained variance** — how many components to reach 90%, 95%, 99%
- **Scree plot** — eigenvalue magnitude vs. component index (look for the "elbow")
- **Loading vectors** — which original parameters contribute most to each PC

**Critical caveat — sample size:** If you have fewer presets than features (p > n), the PCA is unreliable. You cannot estimate the covariance matrix well. In this case, treat the preset-only PCA as exploratory and rely on the augmented PCA (step 7) for definitive dimensionality estimates. Consider also using parallel analysis (compare eigenvalues to those from random data of the same shape) to distinguish real structure from noise.

### 6. Interpret the Components

For each top PC, look at its loading vector (the eigenvector weights):

| Pattern | Interpretation |
|---|---|
| All elements of an array load together with similar sign/magnitude | That array is effectively a single scalar. Create one macro that sets all elements uniformly. |
| Array elements load with different signs | There's meaningful per-element variation. Consider keeping the array or finding a parameterization (e.g., "tilt" = low-to-high gradient). |
| A scalar loads alone on a PC | It's already an independent control. Keep as-is. |
| Multiple params load together on the same PC | They co-vary — consider a single macro that drives all of them. |
| A param doesn't load significantly on any top PC | It's either constant across presets (should have been caught in step 1) or only relevant to outliers (move to "advanced" layer). |

### 7. Validate with Array Summary Statistics

Independently of PCA, compute per-array stats across all presets:

- **Coefficient of variation** (std/mean) per preset, then count how many presets have CV < 0.1
- If >60% of presets have near-zero CV for an array, it's effectively a scalar → replace with single control
- **Note:** The CV < 0.1 and 60% thresholds are starting points. Adjust based on listening — if a parameter with CV 0.15 sounds uniform to your ear, it's uniform.

Cross-check with PCA: if PCA says an array is a single scalar (step 6) and CV stats confirm it, high confidence. If they disagree, investigate.

### 8. Augment with Random Parameter Sampling

This step is **required, not optional** — it solves the small-sample problem (p > n) and tests whether dimensionality reduction holds beyond the explored preset region.

- **Uniform random** — sample each parameter independently from its valid range. Tests the full geometry of the space.
- **Perturbed presets** — take each preset, add Gaussian noise (10–20% of range) per parameter. Tests sensitivity near known-good regions. These are more informative than uniform random for macro design.
- **Sample count** — 200–500 random sets is usually enough to stabilize eigenvalues.

Run PCA on three groups: **presets-only**, **random-only**, **combined**. Compare the explained variance curves and loadings.

**What to look for:**
- If random sampling requires significantly more PCs for 90% variance → the preset bank is clustered in a low-dimensional submanifold. The macros target the "useful" subspace, which is fine — but document the coverage limitation.
- If the number of PCs is similar → the parameter space is inherently low-dimensional.

### 9. Separate Analysis by Preset Category

Run PCA on subsets:
- **Hand-designed only** — represents the "intended" parameter space
- **ML-generated only** — may exploit unused dimensions
- **All presets** — baseline

If dimensionality drops significantly for hand-designed presets (it usually does), the macro controls should target that subspace. ML-generated presets that need the full space can bypass macros.

### 10. Perceptual-Space PCA (Companion Analysis)

The preceding steps analyze parameter space. This step analyzes *perceptual* space — which is what macros should ultimately control.

**Method:**
1. Render each preset through the DSP (impulse response for linear effects, test signal for nonlinear)
2. Extract perceptual features from the rendered audio:
   - **Reverbs**: RT60 per octave band, EDT, C80, spectral centroid over time, echo density
   - **Spectral effects**: mel-cepstral coefficients, spectral flux, spectral flatness
   - **General**: RMS envelope shape, spectral centroid, spectral bandwidth, crest factor
3. Run PCA on the perceptual feature matrix
4. Compare the perceptual PCs to the parameter-space PCs

**What to look for:**
- If they agree (same number of PCs, similar groupings) → parameter-space PCA is a good proxy. Proceed with confidence.
- If they disagree → the parameter-space PCA is misleading. Two cases:
  - Perceptual PCA needs *fewer* PCs → some parameter variation is inaudible. Good news — you can simplify further.
  - Perceptual PCA needs *more* PCs → some perceptually distinct sounds are close in parameter space (nonlinear DSP). The macro mapping needs to be more careful in those regions.

The perceptual PCA should drive macro naming and range design even when parameter-space PCA drives the structure.

### 11. Cluster Analysis

Run k-means (or DBSCAN / hierarchical clustering for non-spherical clusters) on the PCA-reduced space (using enough PCs for 95% variance).

For k-means, use silhouette score to pick k, but cross-check with a dendrogram — silhouette can be misleading with unbalanced cluster sizes.

This reveals:
- Whether there are natural preset "families" (e.g., rooms vs. plates vs. ambient)
- Outlier presets that don't fit the simplified model
- Whether the macro space has a clean topology or disjoint regions
- If clusters are disjoint, you may need separate macro mappings per cluster (e.g., "mode" selector + per-mode macros)

### 12. Design Macro Controls

Map each significant PC to an intuitive control:

**Direct mappings** (PC loads on a single original param):
- Keep the param as-is. Rename if the original name is too technical.

**Array-to-scalar** (PC loads uniformly on all elements of an array):
- Single scalar that sets all elements to the same value.
- Mapping can be linear or nonlinear (e.g., exponential for delay times).

**Co-varying groups** (PC loads on multiple params):
- Single macro with a defined mapping to each underlying param.

**Template-based arrays** (array values vary but maintain fixed ratios):
- Single scalar controls the "base" value; per-element values = base * ratio_template.
- Detect this by checking if ratios are consistent across presets.

**Eliminated params** (near-zero variance, no significant loading):
- Fix to default value. Move to an "advanced" panel if needed.

### 13. Encode Parameter Constraints

Some parameters interact — the simplified mapping must preserve these constraints.

**Common patterns:**
- **Safety constraints** — e.g., high feedback requires nonlinearity to prevent explosion. The forward mapping must enforce this.
- **Conditional subsystems** — child params are meaningless when parent is off. Zero them out.
- **Derived relationships** — e.g., per-node pans derived from stereo width. Don't expose both.
- **Clamping** — the simplified param ranges should make it impossible to produce out-of-range full params.

Encode these in the forward mapping (`to_full_params`), not as separate validation. The simplified params should be safe by construction.

### 14. Validate Macro Mapping Continuity

For DAW automation and preset morphing, smooth macro changes must produce smooth audio changes.

**Method:**
1. For each macro, sweep it from min to max in small steps (100–200 steps) while holding other macros at sensible defaults
2. Render each step through the DSP
3. Compute an audio similarity metric between adjacent steps
4. Plot the metric vs. macro value — look for discontinuities (jumps, clicks)

**What causes discontinuities:**
- Categorical parameters switching (matrix type, waveform) — these must be handled as discrete modes, not continuous macros
- Exponential mappings at range boundaries
- Integer quantization of delay times causing jumps
- Threshold effects in the DSP

If a macro has unavoidable discontinuities, document it and consider restricting its automation range or adding interpolation/crossfade logic.

### 15. Implement the Mapping Layer (Rust)

The mapping layer lives in the Rust DSP crate alongside the processor — not in Python. The PCA analysis is a throwaway Python tool; the resulting params struct and mappings are Rust code.

```
rust/crates/<plugin>-dsp/src/simplified.rs

SimplifiedParams::to_full_params(&self) -> FullParams      // forward
SimplifiedParams::from_full_params(p: &FullParams) -> Self  // reverse (lossy)
```

**Forward mapping** (simplified → full): deterministic, lossless within the macro subspace. This is what the DSP consumes.

**Reverse mapping** (full → simplified): best-fit approximation. Lossy by design — per-element variation in eliminated dimensions is discarded. Needed for converting existing presets to the simplified representation.

Both mappings get `#[serde(default)]` for sparse JSON support and full unit test coverage.

**Key constraint:** The DSP processing code never changes. The mapping layer sits above it.

### 16. Validate Reconstruction (Parameter-Level)

For each preset:
1. Convert full → simplified (reverse)
2. Convert simplified → full (forward)
3. Compare original vs. reconstructed full params

**Metrics:**
- Per-array relative error: `mean(|orig - recon|) / mean(|orig|)`
- Per-scalar absolute error
- Categorical match

**Expected results:**
- Hand-designed presets: <10% error for most, <20% for nearly all
- ML-generated / experimental: higher error is expected and acceptable
- Any hand-designed preset with >20% error warrants investigation — outlier exploiting the full space, or the macro mapping needs adjustment

**Note:** These thresholds are starting points. A parameter with 15% reconstruction error may be inaudible (validated in step 17), while 5% error on a critical parameter may be clearly audible. Parameter error is a proxy — perceptual validation is the ground truth.

### 17. Validate Perceptually (Audio Comparison)

Parameter-level reconstruction error doesn't tell you if it *sounds* the same. Two parameter sets can differ numerically but produce nearly identical audio, or match closely in params but sound different due to nonlinear interactions.

**Method:**
1. Render a test signal through the DSP with the original full params
2. Render the same signal with the reconstructed params (simplified → full → DSP)
3. Compare the two outputs

**Test signal selection** (depends on effect type — see "System Linearity" in specialization section):
- **Linear effects**: Impulse response captures the full character in one pass
- **Nonlinear effects**: Use program material (music, drums, voice) at multiple input levels
- **Mixed**: Test with both impulses and program material

**Metrics** (choose appropriate ones for your effect — see "Perceptual Metrics" in specialization section):
- **RMS difference** — simple amplitude error (normalize both outputs first)
- **Spectral similarity** — log-spectral distance, mel-cepstral distance
- **Effect-specific** — RT60 for reverbs, gain curve for dynamics, etc.
- **Spectrogram difference** — subtract spectrograms, check if residual is perceptually negligible

**Pass criteria** (adjust per effect):
- Spectral similarity > 0.95 for hand-designed presets
- Effect-specific metrics within 10%
- RMS difference < -40dB (inaudible)

This is the ground truth. If parameter error is high but audio error is low, the macro mapping is fine — the "lost" parameters didn't matter perceptually.

### 18. Explore the Unmapped Space

After validating that the macro controls reproduce existing presets, investigate the parameter dimensions they *can't* reach. This answers: "are we losing any musically interesting sounds?"

**Method:**
1. Identify the locked-out dimensions — per-element variation in arrays that the macros set uniformly, parameter combinations the macros can't reach
2. For each dimension, generate parameter sets that maximize variation in that dimension while keeping other params in a reasonable range
3. Render each set through the DSP
4. Also render the closest macro-reachable version (round-trip through reverse → forward mapping)
5. Compare the two outputs with appropriate metrics
6. **Output WAV files** for human listening — metrics can miss differences that matter musically

Also explore the *boundaries* of the mapped space — edge cases of macro ranges often produce artifacts (clipping, instability, silence).

**Expected results:**
Most experiments will show high similarity — the macro-reachable version sounds close. A few may reveal genuinely different textures. These are typically niche sounds that can be accessed by writing full params directly, bypassing the macro layer.

## Alternative and Complementary Methods

PCA is the default starting point because it's fast, interpretable, and works well when parameter relationships are roughly linear. But audio effects often have nonlinear behavior, small preset banks, and parameters where statistical variance doesn't correlate with perceptual importance. The methods below address specific weaknesses. Some are drop-in replacements for PCA; others complement it.

### Sparse PCA — More Interpretable Macro Mappings

**What it does:** Standard PCA produces dense loadings — every parameter contributes a little to every component. Sparse PCA adds a penalty that forces most loadings to zero, so each component loads on only a handful of parameters.

**Why it matters for audio:** Dense loadings make macro design an interpretation exercise — you squint at loading magnitudes and decide "this is probably size." Sparse PCA gives you components like "PC1 = delay_times + diffusion_delays" directly. The macro mapping writes itself.

**When to use:** Drop-in replacement for standard PCA in step 5. Almost always an improvement for macro design. The slight loss in total explained variance per component is worth the interpretability.

**Tradeoff:** Sparse components aren't orthogonal, so explained variance doesn't sum cleanly. You may need one or two extra components to reach the same coverage. In practice this rarely matters — the goal is macro controls, not a mathematically optimal basis.

**Implementation:** `sklearn.decomposition.SparsePCA` or `SparseCoder`. The sparsity parameter (alpha) controls how many parameters each component can touch — tune it until the components are interpretable but still capture the key variation.

### Sobol Sensitivity Analysis — Which Parameters Actually Affect the Sound

**What it does:** Varies all parameters simultaneously (not one at a time) and decomposes the *output* variance into contributions from each parameter and their interactions. First-order indices tell you how much each parameter matters on its own. Total-order indices include interactions.

**Why it matters for audio:** PCA tells you which parameters *vary* across presets. Sobol tells you which parameters *matter to the sound*. These can be very different — a parameter might vary a lot across presets but have negligible audible effect, or it might be nearly constant across presets but have huge impact when changed. Sobol catches interaction effects that PCA misses entirely: "feedback_gain barely matters alone, but feedback_gain × saturation explains 30% of output variance."

**When to use:** Complement to step 1 (dead parameter detection). The current methodology uses one-at-a-time sweeps, which miss interactions. Sobol is the rigorous version. Also useful before PCA to prioritize which parameters to include in the feature matrix — if Sobol says a parameter has near-zero total-order index, it's truly dead regardless of what presets do with it.

**Tradeoff:** Expensive. Requires many DSP renders (typically 1000× the number of parameters for stable estimates). For a 56-parameter effect, that's ~56,000 renders. Feasible if each render takes <100ms (a few minutes total), impractical if renders are slow. Use a fast test signal (short impulse or 1-second noise burst) to keep individual render times down.

**Implementation:** `SALib` library in Python. Define parameter ranges, generate Saltelli samples, render each through the DSP via PyO3, compute Sobol indices. The render function is the bottleneck — batch it.

### Factor Analysis — Separating Signal from Noise in Parameter Variation

**What it does:** Like PCA, but explicitly models each observed parameter as a mix of shared latent factors plus per-parameter noise. PCA treats all variance as signal; Factor Analysis separates "this parameter co-varies with others because of a shared underlying cause" from "this parameter jiggles randomly on its own."

**Why it matters for audio:** Some parameter variation across presets is intentional (a sound designer adjusting brightness) and some is incidental (slightly different values that don't matter). PCA conflates the two. Factor Analysis attributes the incidental variation to noise, producing cleaner factors that better represent the intentional design dimensions.

**When to use:** When you suspect some parameters have noisy or arbitrary values across presets — common with ML-generated presets or presets ported from different plugin versions. Also useful when PCA produces components that are hard to interpret (loading on many params with similar small weights) — the noise might be muddying the structure.

**Tradeoff:** Slightly more complex to fit (iterative estimation, need to choose the number of factors). Results are similar to PCA when noise is low. Most useful when your preset bank is messy.

**Implementation:** `sklearn.decomposition.FactorAnalysis`. Compare the factor loadings to PCA loadings — if they're similar, the data is clean and PCA is fine. If they differ significantly, the Factor Analysis loadings are more trustworthy for macro design.

### Autoencoders — Nonlinear Parameter Compression

**What it does:** A neural network with a narrow bottleneck layer. The encoder compresses full parameters down to a few latent values; the decoder reconstructs full parameters from those values. Unlike PCA, both the compression and reconstruction can be nonlinear — the "macro space" can be a curved surface through parameter space.

**Why it matters for audio:** Audio effect parameters often have nonlinear relationships. Short delays + high feedback = metallic resonance; long delays + high feedback = lush reverb. PCA can't represent this as a single "character" dimension because the relationship between delay and feedback changes direction. An autoencoder can learn a curved path through parameter space that smoothly transitions from metallic to lush.

**When to use:** When PCA clearly struggles — specifically, when the perceptual-space PCA (step 10) disagrees with parameter-space PCA, or when PCA needs many more components than expected. Also when you want preset *generation* — sample from the latent space to create new presets. A Variational Autoencoder (VAE) adds structure to the latent space so that random samples produce valid, musically coherent parameter sets.

**Tradeoff:** Needs more data (hundreds of presets or heavy augmentation). The latent dimensions aren't automatically interpretable — "latent dim 3" doesn't have a name. You can partially fix this with disentangled VAEs (β-VAE), which encourage each latent dimension to control one independent factor. But the resulting macros may still need manual labeling after listening. Also harder to implement the forward/reverse mapping in Rust — you'd need to export the trained network weights and re-implement the decoder as the forward mapping.

**Implementation:** PyTorch or JAX for training. Architecture: input (56) → encoder (dense layers) → bottleneck (8–12) → decoder (dense layers) → output (56). Train on presets + augmented data with MSE reconstruction loss. For the Rust mapping, export decoder weights and implement as matrix multiplications + activations — no ML framework needed at runtime.

### Perceptual Distance Optimization — Skip the Statistics, Optimize Directly

**What it does:** Define a function that measures how different two parameter sets *sound* (not how different they are numerically). Then use optimization to find the smallest set of macro controls whose forward mapping can reproduce every preset within a perceptual tolerance.

**Why it matters for audio:** Every other method works in parameter space and hopes the results transfer to perception. This method works in perceptual space directly. If you have a good perceptual distance function, the macro controls are guaranteed to capture the dimensions that matter to the ear.

**Method:**
1. Define a perceptual distance: render both parameter sets, compute spectral distance / mel-cepstral distance / RT60 difference / etc.
2. Start with a candidate macro structure (informed by PCA or domain knowledge)
3. Optimize the macro-to-full-params mapping to minimize the maximum perceptual distance across all presets
4. If the best achievable error exceeds tolerance, add a macro and repeat
5. If it's well within tolerance, try removing a macro and repeat

**When to use:** As the final refinement after PCA-based macro design. PCA gives you a good starting structure; perceptual optimization tunes the mapping to minimize audible error rather than parameter-space error. Especially valuable when step 17 (perceptual validation) reveals that parameter reconstruction error doesn't predict audible quality.

**Tradeoff:** Expensive — each optimization step requires DSP renders. The perceptual distance function is the critical design choice and there's no universal correct answer. Also, the optimization can get stuck in local minima, especially with many macros. Use PCA results to initialize the mapping rather than starting from scratch.

**Implementation:** Scipy minimize or Bayesian optimization (botorch). The objective function renders two parameter sets via PyO3 and returns the perceptual distance. Warm-start with the PCA-derived mapping.

### Picking a Method

| Situation | Recommended approach |
|---|---|
| First pass, unknown parameter structure | Standard PCA (fast, gives you a baseline) |
| PCA loadings are hard to interpret | Sparse PCA (drop-in, cleaner components) |
| Want to know which params matter to the *sound*, not just which vary | Sobol sensitivity analysis (complement to PCA) |
| Messy preset bank, ML-generated presets mixed in | Factor Analysis (separates signal from noise) |
| Strong nonlinear parameter interactions, PCA needs too many components | Autoencoder / VAE (captures curved manifolds) |
| PCA-based macros pass parameter validation but fail perceptual validation | Perceptual distance optimization (final refinement) |
| All of the above are overkill for your effect | PCA is fine. Most effects with <30 params and well-designed presets don't need anything fancier. |

In practice, the most common workflow is: **Sparse PCA** as the primary analysis (replaces standard PCA), **Sobol** to validate dead parameter detection, and **perceptual optimization** as a final tuning pass if needed. The others are for when something isn't working.

## Applying to Other Plugins

### General Checklist

**Analysis (Python — throwaway scripts):**
1. Read "Specializing to Your Effect" — answer all questions for your plugin
2. Identify and remove dead parameters (step 1)
3. Write PCA script: `scripts/<plugin>/<plugin>_pca.py` — load presets, flatten, transform, standardize, PCA
4. Compute array CV stats, variance per feature, scree plot, loadings
5. Augment with random samples + perturbed presets, compare explained variance
6. Run separately on hand-designed vs. ML-generated subsets
7. Run perceptual-space PCA (render presets, extract audio features, compare to parameter-space PCA)
8. Cluster analysis — identify preset families, outliers
9. Identify macro controls from top PCs

**Implementation (Rust):**
10. Implement `SimplifiedParams` in `rust/crates/<plugin>-dsp/src/simplified.rs`
11. Forward mapping: `to_full_params()` with safety constraints baked in
12. Reverse mapping: `from_full_params()` (lossy best-fit)
13. `#[serde(default)]` for sparse JSON, unit tests for roundtrip, extremes, constraints

**Validation:**
14. Parameter reconstruction: <10% relative error for hand-designed presets
15. Audio comparison: render original vs. reconstructed through DSP, compare with effect-appropriate metrics
16. Macro continuity: sweep each macro, check for discontinuities
17. Unmapped space exploration: listen for musically interesting sounds in locked-out dimensions
18. `cargo test -p <plugin>-dsp` passes — DSP unchanged

## Shared Tooling

The PCA analysis is Python (sklearn), the DSP and simplified params are Rust. For the audio validation step, the Rust DSP is called from Python via PyO3 bindings (maturin). This is already how the project works — each plugin has a `<plugin>-python` crate with PyO3 bindings.

```
scripts/
  pca_common.py              — shared helpers: load presets, flatten, array stats, PCA runner
  <plugin>/
    <plugin>_pca.py           — plugin-specific feature extraction + analysis
    <plugin>_macro_validate.py — reconstruction + audio comparison
```

The `pca_common.py` module would provide:
- `load_presets(preset_dir)` — load all JSON presets
- `flatten_arrays(preset, defaults, array_keys, scalar_keys, categorical_keys)` — generic flattening
- `array_cv_stats(presets, defaults, array_keys)` — CV analysis
- `run_pca_analysis(X, labels, feature_names, title)` — PCA + scree + loadings + clusters
- `compare_audio(render_fn, original_params, reconstructed_params, test_signal)` — render both, compute spectral distance

The Rust `simplified.rs` modules are per-plugin — no shared crate needed since each plugin's mapping logic is unique.
