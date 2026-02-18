# ChordSpace synthesis engine design and timbre–dissonance integration

**A chord exploration tool that adapts consonance scoring to timbre requires a synthesis engine where the spectral content of every voice is analytically known at all times.** This report provides the mathematical foundations, data structures, algorithms, and architectural patterns needed to build ChordSpace's Rust-based backend — from oscillator equations through roughness computation to browser playback. The central design principle: the synthesis engine must produce not only audio but also a structured partial list per voice, enabling the dissonance engine to operate on exact spectral data rather than FFT estimates. Every synthesis method, instrument preset, and effect in the pipeline is evaluated through this lens.

---

## 1. Seven synthesis methods and their spectral predictability

ChordSpace's dissonance engine requires knowing the exact frequencies and amplitudes of every partial in every sounding voice. This constraint makes **spectral predictability** the decisive criterion when choosing synthesis methods. Each method below is evaluated on its mathematical formulation, computational cost per sample per voice at 48 kHz, polyphony headroom for 4–6 simultaneous chord tones, and — critically — how precisely the resulting overtone series can be predicted without analyzing the audio output.

### Subtractive synthesis

The workhorse for analog-style pads, bass, and brass. A band-limited oscillator generates a waveform with a known Fourier series, then a resonant filter reshapes the harmonic amplitudes.

**Waveform equations (Fourier series):**

- Sawtooth: `x(t) = (2/π) Σₙ (-1)^(n+1) sin(2πnf₀t) / n` — all harmonics at **1/n** amplitude
- Square: `x(t) = (4/π) Σ_{n odd} sin(2πnf₀t) / n` — odd harmonics only at 1/n
- Triangle: `x(t) = (8/π²) Σ_{n odd} (-1)^((n-1)/2) sin(2πnf₀t) / n²` — odd harmonics at **1/n²** (much softer)
- Pulse (duty cycle d): harmonic envelope modulated by `sinc(nπd)`, so varying d changes which harmonics are present

Anti-aliasing uses the **PolyBLEP** algorithm: a polynomial correction applied within ±1 sample of each waveform discontinuity. For normalized phase `t` with increment `dt = f₀/Fs`, the correction is `polyblep(t, dt) = t' + t' - t'·t' - 1` when `t < dt`, achieving roughly **−60 to −80 dB** alias suppression at ~4 operations per discontinuity.

**State Variable Filter (SVF)** provides simultaneous LP/BP/HP outputs from two integrators:

```
yh[n] = x[n] - yl[n] - (1/Q)·yb[n]
yb[n+1] = yb[n] + g·yh[n]
yl[n+1] = yl[n] + g·yb[n+1]
```

where `g = 2·sin(π·fc/Fs)`. The 2-pole low-pass response attenuates partial n above cutoff by `|H(nf₀)| = 1/√((1-(nf₀/fc)²)² + (nf₀/(Q·fc))²)`, giving **−12 dB/octave** rolloff.

**Cost:** ~25–30 ops/sample/voice. **Polyphony:** excellent — 6 voices ≈ 180 ops/sample. **Spectral predictability:** indirect but fully computable. Output partial amplitudes equal `Aₙ(output) = Aₙ(waveform) × |H(nf₀)|`, where both the waveform spectrum and filter transfer function are analytically defined. The dissonance engine can compute this in closed form.

### FM synthesis

The mathematical engine behind electric piano, bells, and metallic tones. A modulator oscillator perturbs the phase of a carrier:

```
y(t) = A·sin(2πfc·t + I·sin(2πfm·t))
```

The **Bessel function expansion** gives the exact spectrum: `y(t) = A·Σₙ Jₙ(I)·cos(2π(fc + n·fm)t)`. Sideband frequencies appear at `fc ± k·fm` with amplitudes `Jₖ(I)`. **Carson's rule** estimates significant sideband count at approximately **I + 1**. Integer C:M ratios produce harmonic spectra; non-integer ratios produce inharmonic/metallic timbres. The DX7 architecture uses 6 sine operators in **32 algorithm topologies** — from simple 2-operator stacks to complex 6-operator feedback networks.

**Feedback FM** (single operator feeding back) produces sawtooth-like spectra: `y[n] = sin(2πfc·n/Fs + β·y[n-1])`.

**Cost:** ~10–15 ops per operator. A 4-operator voice costs **40–60 ops/sample**; 6-operator costs 60–90. **Polyphony:** good — the DX7 ran 16 voices on 1983 hardware. **Spectral predictability:** moderate. Exact partial frequencies and amplitudes are computable via Bessel functions for any given parameters, but Bessel functions are non-intuitive (oscillatory with sign changes), and multi-operator stacks create combinatorially complex spectra. For the dissonance engine: compute `Jₖ(I)` for all significant k, map to frequencies `fc ± k·fm`.

### Additive synthesis

The gold standard for spectral predictability — every partial is a direct parameter:

```
y(t) = Σₙ Aₙ(t)·sin(2πfₙ(t)·t + φₙ)
```

Each of N partials has independently controllable frequency, amplitude, and phase. For large partial counts (N > 64), **iFFT synthesis** amortizes cost: render all partials as spectral bins, apply inverse FFT, overlap-add for continuity.

**Cost:** ~3–4 ops per partial (sine lookup + multiply + accumulate). 16 partials ≈ 50 ops, **32 partials ≈ 100–128 ops/sample/voice**. iFFT approach: ~8 ops/sample amortized for 128 partials. **Polyphony:** moderate at high partial counts — 32 partials × 6 voices ≈ 768 ops/sample, well within budget. **Spectral predictability: perfect.** The spectrum *is* the parameter set. The dissonance engine reads synthesis parameters directly with zero additional computation.

### Wavetable synthesis

Pre-computed single-cycle waveforms stored in lookup tables, read by a phase accumulator with interpolation. Band-limiting via **mipmap levels** (one table per octave, generated by zeroing FFT bins above the Nyquist limit for that octave). **Wavetable position modulation** interpolates between adjacent frames: `y[n] = (1-α)·frameₖ[φ·L] + α·frameₖ₊₁[φ·L]`.

Wavetables can be generated algorithmically from spectral descriptions: specify harmonic amplitudes `{A₁...Aₙ}`, set `X[k] = Aₖ·e^(jφₖ)`, compute IFFT to get a single-cycle waveform.

**Cost: ~8–15 ops/sample/voice** — the cheapest method (table lookup + cubic interpolation + mipmap selection). **Polyphony:** excellent. **Spectral predictability:** high at design time, fixed at runtime. Each wavetable frame's spectrum is precisely defined. For the dissonance engine: analyze each frame offline via FFT to extract partial amplitudes; these become known constants per frame. As position modulates, linearly interpolate the known spectra.

### Physical modeling synthesis

Models the physics of vibrating objects. **Karplus-Strong** (plucked strings) uses a delay line with a loop filter:

```
Initialize: delay line y[0..L-1] with noise burst
Loop: y[n] = g · (y[n-L] + y[n-L-1]) / 2
```

where `L = ⌊Fs/f₀⌋` sets pitch. The loop filter `Hₗₚ(z) = (1+z⁻¹)/2` causes higher harmonics to decay faster, with time constant `τₙ = -L/(Fs·ln(g·cos(πnf₀/Fs)))`. **Fractional delay** for precise tuning uses an allpass interpolator: `Hₐₚ(z) = (z⁻¹+a)/(1+a·z⁻¹)`. **Stiffness** (piano inharmonicity) is modeled by an allpass dispersion filter in the loop, stretching higher partials according to `fₙ = n·f₀·√(1+B·n²)`.

**Digital waveguide** synthesis extends this with bidirectional traveling waves and scattering junctions for coupled strings. Full piano models add nonlinear hammer interaction, 3 coupled strings per note, and soundboard resonance.

**Cost:** Basic KS: **~15–20 ops/sample/voice**. Extended KS: 25–40. Full piano model: **500–2000+**. **Polyphony:** excellent for basic KS (6 voices ≈ 120 ops); challenging for full piano. **Spectral predictability:** indirect — the overtone series emerges from physical parameters. However, partial frequencies (`fₙ ≈ n·f₀·√(1+B·n²)`) and decay rates (`τₙ` from loop filter analysis) are analytically derivable. For the dissonance engine: extract expected spectrum from physical parameters at preset load time.

### Modal synthesis

Models resonant bodies as banks of exponentially decaying sinusoids:

```
y(t) = Σₖ Aₖ·exp(-Rₖt)·sin(2πfₖt + φₖ)
```

Each mode has explicit frequency `fₖ`, amplitude `Aₖ`, and decay rate `Rₖ`. Implemented efficiently as **biquad resonant filters** (2 multiplies + 2 adds per mode per sample). Modes need not be harmonically related — natural for bells, bars, plates. Common modal frequency patterns include free-free beams (xylophone: fₖ ∝ {1.000, 2.756, 5.404, 8.933, ...}) and circular membranes (drums: Bessel-function zeros).

**Cost:** ~4–6 ops per mode. **32 modes ≈ 128–192 ops/sample/voice.** **Polyphony:** good. **Spectral predictability: direct** — second only to additive synthesis. Every mode's frequency and amplitude is an explicit parameter. The dissonance engine knows the exact spectrum at any time t, with amplitudes decaying as `Aₖ·exp(-Rₖt)`.

### Phase distortion and waveshaping

Nonlinear transfer functions applied to simple waveforms. **Chebyshev polynomials** provide exact harmonic control: `Tₙ(cos θ) = cos(nθ)`, so applying `Tₙ` to a cosine input produces precisely the nth harmonic. To produce a target spectrum `{α₁...αₙ}`, use `f(x) = Σₙ αₙ·Tₙ(x)`. **tanh() waveshaping** (`y = tanh(k·x)`) adds odd harmonics with approximately 1/n amplitude — warm, tube-like saturation.

**Phase distortion** (Casio CZ method) warps the phase ramp of a cosine, producing linear spectra (1/n rolloff) that sound like filtered analog waveforms.

**Critical caveat for chords:** Chebyshev waveshaping produces clean harmonics only for single sinusoidal inputs. For complex inputs (mixed chord tones), intermodulation distortion creates spectral "mess." **Waveshaping must be applied per-voice, before mixing.**

**Cost: ~5–15 ops/sample/voice** — very cheap. **Polyphony:** excellent. **Spectral predictability:** moderate-to-high for per-voice application. With unit-amplitude cosine input, `f(x) = Σ αₙTₙ(x)` produces exactly `{αₙ}`. At lower input amplitudes, harmonics change non-linearly (natural "brightness follows loudness").

### Synthesis method selection matrix for the dissonance engine

| Method | Cost (ops/sample/voice) | Spectral Predictability | Best For |
|---|---|---|---|
| Additive | 50–128 | **Perfect** (direct parameters) | Organ, flutes, custom timbres |
| Modal | 64–192 | **Direct** (explicit modes) | Bells, marimbas, metallic |
| Wavetable | 8–15 | High (known offline) | Versatile, evolving textures |
| Subtractive | 25–30 | Computable (waveform × filter) | Pads, bass, brass |
| FM | 40–60 | Computable (Bessel functions) | Electric piano, bells, metallic |
| PD/Waveshaping | 5–15 | Moderate-high (per-voice) | Aggressive, organ-like |
| Physical modeling | 15–2000 | Derivable (from physics) | Plucked strings, piano |

---

## 2. Overtone profiles for ten instrument families

The dissonance engine's accuracy depends on having precise spectral fingerprints for each built-in timbre. Below are measured and analytically derived partial amplitude profiles, inharmonicity data, and temporal evolution characteristics.

### Rhodes-type electric piano

The Rhodes mechanism — hammer striking a tine coupled to a tonebar, sensed by an electromagnetic pickup — produces an **asymmetric waveform** with velocity-dependent harmonic content. The pickup's nonlinear response causes the 2nd harmonic to rise dramatically as the tine approaches the pickup axis, sometimes exceeding the fundamental.

**Partial amplitudes (relative to fundamental = 1.0):**

| Partial | Soft Strike | Hard Strike |
|---|---|---|
| 1 | 1.00 | 1.00 |
| 2 | 0.50–0.65 | **0.70–0.85** |
| 3 | 0.20–0.35 | 0.35–0.50 |
| 4 | 0.10–0.20 | 0.20–0.35 |
| 5 | 0.05–0.10 | 0.10–0.20 |
| 6–8 | 0.01–0.05 | 0.03–0.15 |
| 9–16 | <0.01 | 0.02–0.05 (attack only) |

**Harmonicity:** Mostly harmonic during sustain. Significant **inharmonic modes during the attack transient** from tonebar bending (confirmed by JASA 2020 research). **Spectral evolution:** Bright "bark" in the first 50 ms with all harmonics plus inharmonic tonebar modes; by 300 ms, the spectrum collapses to a warm fundamental-heavy sustain where primarily partials 1–3 remain. Higher harmonics decay approximately proportional to n².

**Best synthesis:** FM with 2–3 operator pairs. Classic DX7 approach: C:M = 1:1 with velocity-dependent modulation index **I ≈ 1.5–3.5** for the body tone, plus a high-ratio pair (C:M = 1:14) with a short percussive modulator envelope for the bell attack transient.

### Wurlitzer-type electric piano

Reed-based mechanism with electrostatic (capacitive) pickup at 170V DC. **Spectrally richer** than Rhodes — described as "closer to a sawtooth wave" versus the Rhodes' "closer to a sine wave." Both even and odd harmonics are present with non-monotonic amplitude oscillations. At high velocity, the reed's large displacement through the electrostatic pickup field creates significant harmonic distortion — the characteristic "growl."

**Key difference from Rhodes:** Stronger upper harmonics (partials 3–8 at 0.25–0.65 hard), shorter sustain, and dramatic overdrive behavior that adds predominantly odd harmonics from tube amplifier saturation. Partials 2–4 remain strong even at soft dynamics.

**Harmonicity:** Primarily harmonic. Less inharmonicity than Rhodes due to simpler reed geometry. **Best synthesis:** Waveshaping (sine through velocity-dependent nonlinear transfer function) or FM with lower index than Rhodes (I ≈ 1.0–2.5).

### Hammond-style drawbar organ

The most transparently additive instrument in existence. **Nine drawbars** directly control nine harmonically related sine waves:

| Drawbar | Footage | Harmonic | Interval |
|---|---|---|---|
| 1 | 16' | Sub-fundamental | Sub-octave |
| 2 | 5⅓' | Sub-third | Sub-fifth |
| 3 | 8' | 1st (fundamental) | Unison |
| 4 | 4' | 2nd | Octave |
| 5 | 2⅔' | 3rd | Twelfth |
| 6 | 2' | 4th | Fifteenth |
| 7 | 1⅗' | 5th | Seventeenth |
| 8 | 1⅓' | 6th | Nineteenth |
| 9 | 1' | 8th | Twenty-second |

Notably, the **7th harmonic is absent** from the drawbar set (it would sound out of tune). Each drawbar setting (0–8) corresponds to approximately **3 dB increments**: setting 8 = 0 dB, setting 7 = −6 dB, setting 6 = −9 dB, down to setting 1 = −24 dB. This gives 9⁹ ≈ **387 million** possible registrations.

**Harmonicity:** Pseudo-harmonic. Octave harmonics are perfect, but non-octave harmonics use the nearest equal-tempered tonewheel frequency, introducing small deviations (up to ~14 cents for the 5th harmonic). This "imperfect" harmonicity is part of the distinctive Hammond character.

**Spectral evolution:** Essentially static (tonewheels produce continuous sine waves). **Key click** adds a 2–5 ms broadband transient at onset. Optional percussion circuit adds a fast-decaying 2nd or 3rd harmonic. **Best synthesis:** Additive — drawbars literally are additive synthesis.

### Analog-style synth pad

Built from a sawtooth waveform (partials at 1/n: 1.0, 0.50, 0.33, 0.25, 0.20, 0.167, 0.143, 0.125...) filtered through a **4-pole low-pass at ~2–4 kHz** with moderate resonance. A typical pad at A3 (220 Hz) retains partials 1–8 nearly unaffected (all below 2 kHz) with partials 11+ rapidly attenuated by the −24 dB/octave rolloff.

**Detuning** (2–3 oscillators at ±5–15 cents) transforms each sharp partial peak into a cluster of beating frequencies. At fundamental 220 Hz with ±10-cent detune, each partial beats at approximately `n × 1.27 Hz`. Higher partials beat faster, creating the characteristic thick pad sound with internal spectral motion.

**Harmonicity:** Each oscillator is perfectly harmonic; the composite has "spread" spectral peaks. **Best synthesis:** Subtractive — multiple detuned sawtooth oscillators through LPF with slow ADSR envelope.

### Plucked string (guitar)

The pluck position determines the initial spectral shape via `Aₙ ∝ sin(nπd/L)/n²`, where d/L is the fractional pluck position. For typical guitar plucking at d/L ≈ 0.23, partial 4 falls near a node and is strongly suppressed, while partial 2 is prominent at approximately 0.375 relative amplitude.

**Inharmonicity** follows `fₙ = n·f₀·√(1 + B·n²)`, where the inharmonicity coefficient B depends on string construction:

- Nylon (classical guitar): **B ≈ 10⁻⁵** — deviation <1% up to partial 17
- Steel-string acoustic: **B ≈ 3×10⁻⁵**
- Electric guitar: **B ≈ 10⁻⁴**

**Spectral evolution:** Bright attack (all excited harmonics present plus broadband pluck noise), then higher harmonics decay faster with damping τₙ ∝ 1/nᵧ. By ~500 ms, partials 8+ are largely gone; sustain is dominated by partials 1–4. **Best synthesis:** Karplus-Strong physical modeling.

### Clavinet

A struck/plucked string with magnetic pickups creates an exceptionally **harmonically rich** timbre. Pickup position introduces a **comb filter** — notches appear at harmonics where the pickup sits at a vibrational node (approximately every 5th partial for typical settings). Partials 1–4 are all strong (0.55–1.0 relative amplitude), with characteristic comb-filter dips at partials 5, 10, and 15. Bridge pickup emphasizes treble; neck pickup produces warmer tone. Anti-phase combination cancels the fundamental for a thin, nasal sound.

**Harmonicity:** Primarily harmonic with measurable inharmonicity from steel string stiffness. **Best synthesis:** Karplus-Strong with bright noise excitation and a comb filter modeling pickup position.

### Sub bass and synth bass

The simplest spectral profiles in ChordSpace's preset library. **Pure sine sub** has only the fundamental. **Square sub** has odd harmonics at 1/n. **808-style bass** uses a filtered saw/triangle with rapid partial rolloff (partial 2 at 0.3–0.5, partial 3 at 0.1–0.2, partial 4+ below 0.08). **Reese bass** (two detuned saws at ±7 cents) produces the standard 1/n sawtooth spectrum but with each partial as a beating pair at rate ≈ n × 1.8 Hz for a 50 Hz fundamental. All types are perfectly harmonic per oscillator.

### Brass stab and string ensemble

**Brass stab** is a sawtooth source (1/n harmonics) with a **time-varying low-pass filter**. During the 5–20 ms attack, the filter snaps open to ~8–10 kHz, exposing the full spectrum — the characteristic "bwah." The filter then settles to a sustain cutoff of ~2–4 kHz with moderate resonance (Q ≈ 2–4) adding a nasal peak. The spectrum is perfectly harmonic (sawtooth source).

**String ensemble** stacks 4–8 slightly detuned sawtooth oscillators (±5–15 cents) with slow attack envelopes (200–500 ms). Each individual oscillator has the standard 1/n spectrum, but the composite shows spread partial clusters. At 220 Hz with ±15-cent spread, the fundamental cluster spans ±1.9 Hz of beating, scaling linearly with partial number — higher partials merge into a smooth chorus wash.

---

## 3. Timbre–dissonance interaction: how overtones reshape chord consonance

This section is the theoretical core bridging synthesis and the consonance engine. **The same chord can sound consonant with one timbre and dissonant with another** — a fact that most music tools ignore.

### The Plomp-Levelt foundation

Plomp and Levelt's 1965 experiments established that for two pure sine tones, roughness (sensory dissonance) peaks at approximately **25% of the critical bandwidth** and drops to zero beyond it. Crucially, for pure sines, **there are no special consonance points at musical intervals** — the octave, fifth, and third have no privileged status. The critical bandwidth CB depends on center frequency:

- **Hutchinson-Knopoff:** `CB(f) = 1.72·f^0.65`
- **Moore-Glasberg (ERB):** `ERB(f) = 0.108·f + 24.7`

Roughly, CB ≈ 100 Hz for frequencies below 500 Hz and widens proportionally above ~500 Hz, corresponding to approximately ⅓ octave in the mid range.

### Sethares' computational model with exact constants

William Sethares extended Plomp-Levelt to complex tones (tones with multiple partials). For two sine components with frequencies f₁ < f₂ and amplitudes a₁, a₂:

```
d(f₁, f₂, a₁, a₂) = min(a₁, a₂) × [5·exp(-3.51·s·Δf) - 5·exp(-5.75·s·Δf)]
```

where:

```
Δf = f₂ - f₁
s = 0.24 / (0.0207·f₁ + 18.96)
```

**The exact numerical constants** from Sethares' published code:

| Parameter | Value | Meaning |
|---|---|---|
| d* | **0.24** | Point of maximum dissonance (fraction of CB) |
| s₁ | **0.0207** | Frequency-dependent scaling |
| s₂ | **18.96** | Frequency-dependent offset |
| b₁ | **3.51** | Dissonance curve rise rate |
| b₂ | **5.75** | Dissonance curve fall rate |

The `s` parameter stretches the Plomp-Levelt curve to account for the frequency-dependence of critical bandwidth — maximum dissonance shifts to higher frequency separations for higher-pitched tones.

**Amplitude weighting** uses the minimum model (`min(a₁, a₂)`) per Sethares' 2005 revision: roughness is proportional to the loudness of the beating, which equals the minimum of the two amplitudes.

### Vassilakis' improved model

Vassilakis (2001/2005) refined amplitude handling, achieving **r = 0.98 correlation** with perceptual roughness ratings (versus 0.87 for Hutchinson-Knopoff and 0.73 for Helmholtz):

```
R = (A_min·A_max)^0.1 × 0.5 × [2·A_min/(A_min+A_max)]^3.11 × Z
Z = exp(-3.5·s·Δf) - exp(-5.75·s·Δf)
```

The key innovations: the **SPL term** `(A_min·A_max)^0.1` compresses absolute amplitude dependence (exponent 0.1 versus linear), while the **AF-degree term** `[2·A_min/(A_min+A_max)]^3.11` dramatically reduces roughness when amplitudes are unequal — matching the perceptual finding that a loud partial near a quiet one sounds less rough than two equally loud partials. The steep exponent 3.11 makes this term very sensitive to amplitude imbalance.

### Extending to full chords

For a chord of K notes, each with N partials, **total dissonance is the sum of roughness across all unique partial pairs**:

```
D_total = Σᵢ<ⱼ d(fᵢ, fⱼ, aᵢ, aⱼ)
```

over all partials from all notes combined. The complexity for K notes with N partials each is **O(K²·N²)**: specifically, K(K-1)/2 note pairs × N² partial pairs per note pair, plus within-note self-dissonance. For a **6-note chord with 16 partials**: 96 total partials, 4,560 unique pairs. This is computed in under a microsecond on modern hardware — even without optimization, the dissonance engine is not the bottleneck.

**Optimization strategies** for larger partial counts or real-time interaction:

- **Prune low-amplitude partials:** Skip pairs where min(aᵢ, aⱼ) < 0.01 — many higher partials contribute negligibly
- **Frequency-proximity pruning:** Skip pairs where |fᵢ − fⱼ| > CB(f_mean), since roughness is effectively zero beyond the critical bandwidth
- **Spatial hashing:** Bin partials into critical-band-width buckets; compute pairs only within same or adjacent bins
- **Precompute interval tables:** For a fixed timbre, precompute roughness at 1-cent resolution across one octave (1,200 entries) × 12–24 base pitch points. This reduces real-time chord evaluation from O(K²·N²) to **O(K²)** — 15 table lookups for a 6-note chord instead of 3,840 partial-pair calculations

### How specific timbres shift the dissonance landscape

The magnitude of this effect depends on how different two timbres' overtone structures are:

**Sine wave** — the dissonance curve matches Plomp-Levelt exactly: a single roughness hump rising from unison, peaking at ~25% of CB, falling to zero. **No consonance valleys exist at musical intervals.** A minor second sounds only slightly dissonant; the octave has no special status.

**Sawtooth wave** (harmonics at 1/n) — deep consonance valleys emerge at just-intonation ratios: 2:1 (octave), 3:2 (fifth), 4:3 (fourth), 5:4 (major third), 5:3 (major sixth), 6:5 (minor third). The consonance hierarchy matches Western musical practice. **More harmonics create deeper and more numerous valleys.** This is why standard music theory "works" for harmonic timbres.

**Odd-harmonic timbres** (clarinet-like: square wave) — the octave (2:1) drops in relative consonance because the 2nd harmonic is absent, leaving no partial coincidence to reinforce it. The **twelfth (3:1) can become more consonant than the octave.** This is a dramatic example of timbre-dependent consonance ranking.

**Inharmonic timbres** (bells, metallophones, FM with non-integer C:M) — consonance valleys shift to **completely different locations**. Standard 12-TET intervals may all be dissonant. Sethares demonstrated that gamelan scales (pelog and slendro) correspond to dissonance minima of gamelan metallophone timbres — the "right" scale for a timbre is determined by its partials.

**Piano** (stretched partials: fₙ = n·f₀·√(1+B·n²), B ≈ 10⁻³) — consonance valleys shift slightly sharp relative to integer ratios. This is the physical origin of the **Railsback curve** in piano tuning: stretching the tuning minimizes total roughness for the inharmonic piano spectrum. Differences from harmonic-timbre dissonance are **~5–15%** depending on register.

### Quantifying the visualization impact

Between harmonic timbres varying in brightness (e.g., sawtooth vs. filtered sawtooth), consonance **rankings are preserved** but magnitudes change by ~15–25% — the visualization shifts colors but not topology. Between harmonic and inharmonic timbres, the **entire topology changes**: a minor second can go from "very dissonant" to "moderately consonant"; the octave can become dissonant. These changes require full recomputation and produce dramatic visual updates.

**Perceptual thresholds:** Changes below 5% in dissonance score are imperceptible. Changes of 5–15% are noticeable by trained musicians. Changes above 15% are clearly audible. Changes in **consonance ranking** (one interval surpassing another) are always musically significant.

### Masking effects in dense chords

Standard roughness models assume additivity across all partial pairs. However, in dense sonorities, a loud partial can **mask** nearby quiet partials, reducing perceived roughness. Dense sonorities like Ligeti's *Atmosphères* (composed entirely of minor seconds) don't sound as rough as additive models predict. For simple chords (2–4 notes), masking effects are small and can be neglected. For dense chords (6+ notes with bright timbres), masking can reduce perceived roughness by **20–40%** relative to the additive model.

**Simple masking implementation:** Before computing pairwise roughness, sort all partials by frequency, compute masking thresholds from louder neighbors within 1 critical bandwidth, and attenuate or remove partials that fall below the masking threshold.

---

## 4. Multi-voice polyphonic synthesis architecture

ChordSpace's engine must render full chords (4–6 notes) across multiple timbres simultaneously for voice splitting (bass on one synth, keys on another, pad on a third). The architecture must be **real-time safe**: no allocation, no locks, no syscalls in the audio render path.

### Voice allocation and management

A fixed-size pool of pre-allocated voice structs per instrument type — typically **8–16 voices per instrument, 32–64 total** — handles polyphony with headroom for release tails when new chords are struck. Voice states cycle through idle → attack → sustain → release → idle, with a "stolen" state for rapid reassignment.

**Voice stealing priority** (recommended order for ChordSpace):

1. Steal the oldest voice currently in Release state
2. If none releasing, steal the oldest active voice excluding the bass note (the root of a chord is perceptually most important — stealing it is very audible)
3. Apply a rapid crossfade (~1–5 ms, 48–240 samples at 48 kHz) to the stolen voice before resetting it

**Sample-accurate timing** prevents the 2–5 ms jitter that buffer-boundary quantization would introduce. When a note-on arrives at sample 137 within a 256-sample buffer, the engine renders samples 0–136 without the new note, processes the note-on event, then renders 137–255 with the new voice active. Optional humanization adds deterministic random offsets of 1–10 ms per voice for natural feel.

### CPU budget reality check

At 48 kHz on a modern x86 core (~8–16 billion FLOPS with SIMD), the synthesis budget is generous:

| Configuration | Ops/second | Core Utilization |
|---|---|---|
| 24 voices × subtractive (30 ops) | 34.6M | ~0.4% |
| 24 voices × FM 4-op (60 ops) | 69.1M | ~0.9% |
| 24 voices × additive 32-partial (128 ops) | 147.5M | ~1.8% |
| 24 voices × modal 32-mode (192 ops) | 221.2M | ~2.8% |

Even the most expensive realistic configuration uses **under 3% of a single core.** The architectural challenge is not raw throughput but maintaining real-time guarantees: no allocation, no locks, deterministic execution time.

### Lock-free communication and memory layout

Parameter changes and note events flow from the UI thread to the audio thread via a **lock-free SPSC (single-producer, single-consumer) ring buffer** carrying tagged messages (`NoteOn`, `NoteOff`, `ParamChange` with instrument ID, parameter ID, and value). Simple continuous parameters (knob values) use `AtomicF32` for direct sharing.

For voice data, **Structure of Arrays (SoA)** layout dramatically outperforms Array of Structures (AoS) for SIMD processing. All phases in one contiguous array, all frequencies in another, all amplitudes in another — enabling a single AVX instruction to process 8 voice phases simultaneously.

**Parameter smoothing** prevents audible stepping ("zipper noise") when knobs change: `current += coeff × (target - current)` with `coeff = 1 - exp(-2π/(smooth_time × Fs))`. Typical smoothing: 5–20 ms for user knobs, 1–2 ms for internal modulators.

### Mixing and gain staging

When summing N simultaneous voices, apply **1/√N amplitude scaling** to maintain approximately equal perceived loudness. For a 6-note chord: scale by 1/√6 ≈ 0.408. The signal chain proceeds: per-voice output × (velocity × envelope × 1/√N) → instrument channel bus × channel gain → master bus × master gain → peak limiter (brickwall at −0.3 dBFS) → output. Target **−6 to −3 dB headroom** on the master bus before limiting.

### SIMD optimization strategies

SIMD delivers the single biggest performance improvement for polyphonic synthesis. Process **4 voices simultaneously** (SSE: 4×f32) or **8 voices** (AVX: 8×f32) by packing homogeneous data from different voices into SIMD registers. Vectorized sine computation uses polynomial approximation: `sin(x) ≈ x - x³/6 + x⁵/120 - x⁷/5040` (~10 ops, fully SIMD-friendly, ~120 dB SNR with 7th-order polynomial). Alternative: 4096-point sine lookup table with linear interpolation gives ~90 dB SNR; 8192-point with cubic interpolation gives ~120 dB SNR.

Delay lines for physical modeling use **power-of-2 buffer sizes** for efficient modular indexing: `index = position & (buffer_size - 1)` (bitwise AND replaces modulo division). All audio buffers are pre-allocated at initialization — no allocation occurs during rendering.

---

## 5. Timbre parameterization, morphing, and preset architecture

### Macro parameter mapping

Five user-facing knobs map to underlying synthesis parameters through perceptually scaled curves:

**Brightness (0.0–1.0)** → Filter cutoff via exponential mapping `fc = f_min × (f_max/f_min)^brightness` (range: 200 Hz → 16 kHz), FM modulation index (0 → 5.0), partial amplitude high-frequency rolloff exponent (3.0 → 0.5). Brightness correlates strongly with **spectral centroid** — the amplitude-weighted mean frequency of the spectrum.

**Warmth (0.0–1.0)** → Gentle 6 dB/oct low-pass with cutoff 2–16 kHz, even harmonic boost of up to 6 dB (even harmonics are perceptually associated with warmth), attenuation of partials above the 8th by up to 12 dB.

**Attack (0.0–1.0)** → Amplitude envelope attack time via logarithmic mapping from 0.5 ms (percussive) to 2000 ms (slow pad). Logarithmic scaling ensures fine control at the fast end where small changes are most perceptually significant.

**Body (0.0–1.0)** → Mid-frequency partial amplitudes (partials 3–8, boosted by up to 8 dB), filter resonance Q from 0.5 to 4.5.

**Air (0.0–1.0)** → High-frequency noise component via high-pass filtered noise mixed at up to −12 dB relative to tonal signal, reverb send (0–40% wet), emphasis on partials above the 12th.

**Mapping functions** follow perceptual principles: frequency parameters use logarithmic mapping (matching cochlear perception), amplitude/gain uses exponential mapping (matching dB perception), time parameters use logarithmic mapping (matching temporal sensitivity where small differences at short times matter more).

### Timbre morphing between presets

Three approaches, chosen based on context:

**Parameter-space interpolation** (`param_morphed[i] = (1-α)·A[i] + α·B[i]`) works well when both presets use the same synthesis method. For filter parameters, use Log-Area Ratio coefficients to guarantee filter stability during interpolation. Fails for cross-method morphing (e.g., subtractive → FM) where intermediate parameter values may be meaningless.

**Spectral-space interpolation** aligns partials between two timbres (by harmonic order for harmonic sounds), then interpolates frequencies and amplitudes: `freq_morph = (1-α)·freq_A + α·freq_B`. More musically coherent — produces a single intermediate timbre rather than two overlapping sounds. Unmatched partials fade in/out with α. Can use Line Spectral Frequencies (LSFs) for the most perceptually linear spectral envelope morphs.

**Equal-power crossfade** (`gain_A = cos(α·π/2)`, `gain_B = sin(α·π/2)`) is simplest but produces a blend rather than a morph — two overlapping sounds, not one intermediate timbre.

**Recommended approach for ChordSpace:** Parameter-space interpolation for same-method presets (fast, predictable); spectral fingerprint interpolation for cross-method presets; equal-power crossfade as fallback.

### Preset data structure

A timbre preset fully describes the synthesis state and its interface with the dissonance engine:

```
TimbrePreset {
    meta: { name, category, version }
    synth_method: enum { Subtractive, FM, Additive, Wavetable, PhysicalModel, Modal, Waveshaping }
    synth_params: method-specific parameters
    overtone_profile: [(frequency_ratio: f32, amplitude: f32, decay_rate: f32)] × 16-32
    effects: { chorus, reverb, EQ parameters }
    macro_maps: [{ targets, ranges, curves }] × 5
    amp_envelope: ADSR
    filter_envelope: ADSR
}
```

The **overtone_profile** is the critical bridge to the dissonance engine: an array of **16–32 partial descriptors** (frequency ratio, relative amplitude, per-partial decay rate) computed at preset load time, not during synthesis. Size: 16 partials × 3 floats × 4 bytes = **192 bytes per timbre** — trivially compact. For time-varying timbres, store two fingerprints (attack and sustain) and use the sustain fingerprint for chord exploration since users hear sustained chords.

Ship **12–20 presets** covering the main instrument families: Rhodes EP, Wurlitzer, Hammond organ, analog pad, digital pad, sub bass, plucked string, clavinet, brass stab, string ensemble, plus 2–3 custom timbres designed for interesting dissonance properties.

---

## 6. Audio signal chain for polished chord presentation

### Chorus and ensemble thickening

Chorus uses N copies (typically 2–4) of the signal with independently modulated short delay lines:

```
delay_time[i](t) = base_delay + depth × LFO_i(t)
```

where base delay is **5–30 ms**, modulation depth ±0.5–3.0 ms (~5–20 cents detuning), LFO rate 0.1–5 Hz, with LFO phases evenly distributed: `φᵢ = 2π·i/N`. A classic Juno-style stereo chorus uses two delay lines with quadrature LFO (90° phase offset), panned left and right. Cost: ~12 multiply-adds per sample for 4 voices — negligible.

**Spectral impact on the dissonance engine:** Chorus smears each partial into a narrow beating cluster. A 10-cent detune on a 440 Hz fundamental creates ~5 Hz beating. The consonance engine should model this as a fixed roughness penalty proportional to chorus depth, or widen the effective partial bandwidth in the spectral fingerprint.

### Algorithmic reverb via Feedback Delay Network

The recommended reverb architecture is a **Feedback Delay Network (FDN)** with N delay lines (typically 4–8), a unitary mixing matrix, and per-line absorption filters:

```
s[n+m] = A·s[n] + b·x[n]    (feedback)
y[n] = cᵀ·s[n] + d·x[n]      (output)
```

where A is an N×N **Hadamard matrix** (for maximum diffusion via butterfly operations at O(N log N) cost instead of O(N²)):

```
H₄ = (1/2)·[[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,-1]]
```

Delay line lengths should be mutually prime, in the 20–100 ms range (e.g., for 8 lines at 44100 Hz: {1087, 1283, 1447, 1553, 1699, 1823, 1979, 2113} samples). **Absorption** (high-frequency damping) uses one-pole lowpass filters in the feedback loop: `g(z) = (1-d)/(1-d·z⁻¹)`, with per-delay gain set to `gain = 10^(-3·delay_length/(RT60·Fs))` for target RT60 decay time.

Key parameters: RT60 from 0.2 s (small room) to 5.0 s (cathedral), pre-delay 0–100 ms, damping coefficient, wet/dry mix (15–30% typical for chord presentation). Cost: ~40 ops/sample for an 8-line FDN.

**Reverb and consonance:** Late reverb creates a spectral wash that can perceptually mask some dissonance, but it does not change the fundamental interval relationships. **The consonance engine should operate on the dry signal's spectral fingerprint**, not the reverberant output.

### Per-instrument EQ using biquad filters

Three-band parametric EQ per instrument carves frequency space when multiple timbres play simultaneously. Implementation uses Robert Bristow-Johnson's Audio EQ Cookbook biquad coefficients (direct form I, ~5 multiply-adds per sample per band × 3 bands = 15 ops/sample/instrument). Typical settings: roll off sub-bass below 80 Hz for non-bass instruments, cut mud at 200–400 Hz, add presence at 2–5 kHz.

**EQ changes the effective overtone profile.** If significant EQ (>6 dB boost/cut) is applied, the spectral fingerprint should be adjusted by evaluating the EQ's magnitude response at each partial frequency: `adjusted_amplitude[i] = original_amplitude[i] × |H(freq_ratio[i] × f₀)|`.

### Master bus limiting and stereo imaging

A **look-ahead peak limiter** (5–10 ms look-ahead, 100 ms release) on the master bus prevents clipping on dense chords. The look-ahead delay allows the limiter to anticipate peaks and apply transparent gain reduction using a moving-minimum + cascaded box-filter smoothing algorithm. For lower CPU cost, `tanh(x)` soft clipping provides output bounded to ±1.0 with warm odd-harmonic distortion.

**Constant-power panning** distributes chord voices across the stereo field: `L = cos(θ)`, `R = sin(θ)`, guaranteeing L² + R² = 1 at all positions. For an N-note chord with spread parameter s ∈ [0,1]: `pan[i] = -s + 2s·i/(N-1)`. **Mid-side processing** provides width control: encode L/R → M/S, adjust widths independently, decode back.

### Effects and consonance engine: the dual-path principle

The consonance engine operates on the **pre-effects spectral fingerprint data** (dry signal path), not the post-effects audio. Reverb tail is irrelevant for instantaneous consonance. EQ changes are relevant and should be reflected. Chorus adds intra-note roughness that can be modeled as a fixed penalty. The "known spectral data" path (fingerprint → consonance) is architecturally separate from the "audio rendering" path (synthesis → effects → output).

---

## 7. Server-to-client audio streaming with sub-100ms latency

### Protocol selection and tradeoffs

**Raw PCM over WebSocket** sends uncompressed samples (stereo 48 kHz 16-bit = **192 KB/s = 1.54 Mbps**; 32-bit float = 3.07 Mbps). Zero encoding/decoding latency, lossless quality, but high bandwidth. Implementation: server fills N-sample buffers, sends as binary WebSocket messages.

**Opus over WebSocket** compresses audio before sending. Opus achieves **transparent music quality at 128 kbit/s stereo** (16 KB/s) — a 12:1 compression ratio over 16-bit PCM. The algorithmic delay in CELT-only mode (best for music) with 10 ms frames is **~12.5 ms**. Encoding/decoding costs are negligible on modern hardware. Decode on the client using a WebAssembly Opus decoder running in a Web Worker.

**WebRTC** offers built-in jitter buffering, congestion control, and UDP transport, achieving sub-50 ms end-to-end on same-network connections. However, it requires STUN/TURN servers for NAT traversal, SDP negotiation, and ICE candidate exchange — **complexity not justified** for ChordSpace's unidirectional server→client audio stream.

### Latency budget breakdown

| Stage | Optimistic | Typical | Pessimistic |
|---|---|---|---|
| User input → server (RTT/2) | 2 ms | 15 ms | 50 ms |
| Server synthesis (128 samples) | 2.67 ms | 5.33 ms | 10.67 ms |
| Opus encoding (10 ms frame) | 0 ms (raw) | 10 ms | 20 ms |
| Network to client (RTT/2) | 2 ms | 15 ms | 50 ms |
| Client decode | <0.5 ms | <1 ms | 1 ms |
| Jitter buffer | 5 ms | 20 ms | 60 ms |
| AudioWorklet buffer (128 samples) | 2.67 ms | 5.33 ms | 10.67 ms |
| **Total** | **~15 ms** | **~72 ms** | **~202 ms** |

**Target:** <100 ms is "playable" for interactive chord exploration. <50 ms feels "instant." On good connections with co-located servers, <50 ms is achievable with raw PCM. General internet with Opus: **~70 ms is realistic.** Above 200 ms feels laggy and unmusical.

### Client-side audio playback architecture

The canonical pattern uses three cooperating threads:

1. **Web Worker** receives WebSocket data, decodes Opus frames, writes decoded PCM to a SharedArrayBuffer-based SPSC ring buffer
2. **AudioWorkletProcessor** reads 128 samples per process() call from the ring buffer, outputs to speakers
3. **Main thread** handles UI, visualization, and user input

The AudioWorklet runs on a **dedicated high-priority rendering thread** with a strict ~2.67 ms timing budget at 48 kHz. It must not use postMessage, allocate memory, or trigger garbage collection in its process() callback.

**Adaptive jitter buffering:** Track arrival timestamps of the last N packets, compute jitter as the standard deviation of inter-arrival times, set buffer depth to `target_delay = mean_delay + 2·jitter_stddev` (covers ~95% of variance). Grow quickly on underrun (double), shrink slowly (1 sample per buffer period). Starting depth: ~30 ms, range 15–80 ms.

### Hybrid client-server synthesis for perceived-zero latency

Client-side synthesis generates a basic approximation instantly using Web Audio API oscillators or a simple wavetable when the user selects a chord. The server renders full-quality multi-timbral audio that arrives ~50–100 ms later. The client crossfades from local to server audio over ~20 ms. This approach is **particularly effective for ChordSpace** because chord tones are relatively simple to approximate with a few sine waves, and the crossfade is less critical for sustained tones than for transients.

### Synchronization and clock drift

Each WebSocket audio message includes a **monotonic sequence number** (cumulative sample count since stream start). The client computes playback position as `total_samples_written - ring_buffer_fill_level`. For visual sync, subtract audio buffer latency: `visual_time = server_time - jitter_buffer_delay - AudioWorklet_buffer_delay`.

Clock drift between server and client (~1–100 ppm) accumulates to 360 ms over a 1-hour session at 100 ppm. Solution: monitor ring buffer fill level — if growing, client clock is slower (slightly increase playback rate via resampling); if shrinking, decrease. This is the same approach WebRTC's NetEQ uses.

---

## 8. Feeding spectral data to the visualization pipeline

### Direct spectral export eliminates FFT

ChordSpace's greatest architectural advantage: **the synthesis engine knows exactly what partials it's generating.** For additive, modal, and wavetable synthesis, the partials ARE the synthesis parameters — extracting them costs zero additional computation. For FM synthesis, sideband frequencies and amplitudes are analytically calculable via Bessel functions. For subtractive synthesis, apply the filter's transfer function to the known waveform spectrum: `amplitude_n = base_amplitude_n × |H(n·f₀)|`. Only physical modeling requires FFT analysis of the output.

This is vastly superior to FFT-based analysis: direct spectral data gives **exact frequencies to floating-point precision** versus FFT's frequency resolution of only Fs/N Hz (23.4 Hz for 2048-sample FFT at 48 kHz — too coarse to distinguish semitones below ~100 Hz).

### Dual data path architecture

**Path 1 — Structured spectral data** (synthesis parameters → consonance engine → visualization): an array of `(frequency: f32, amplitude: f32)` tuples per active voice. Size: 6 voices × 16 partials × 8 bytes = **768 bytes per update**. Sent only on chord or timbre change (event-driven, typically <10 updates/second). Transported as JSON or binary-packed messages over a dedicated metadata WebSocket channel.

**Path 2 — Raw audio waveform** (synthesis → encode → AudioWorklet → speakers + optional client-side FFT for spectrogram/oscilloscope visualization): the continuous audio stream at 48 kHz.

Both paths originate from the synthesis engine but serve different consumers with fundamentally different timing requirements.

### Metadata channel protocol

Alongside the audio stream, the server sends structured messages:

- **TIMBRE_PROFILE** (on timbre change): the overtone fingerprint — `[(ratio, amplitude)]` × 16–32 partials. ~200–500 bytes per event.
- **CHORD_CONSONANCE** (on chord change): overall dissonance score, per-interval roughness breakdown, interval ratio identifications. ~200–1000 bytes per event.
- **PARTIAL_SNAPSHOT** (on chord/timbre change): complete per-voice partial list for detailed visualization. 768 bytes typical.

These travel over a **separate WebSocket channel** from audio to prevent metadata processing from blocking audio delivery.

### Update rates for visualization components

| Visualization | Update Rate | Data Source |
|---|---|---|
| Consonance score display | On chord/timbre change | Server (Path 1) |
| Interval roughness breakdown | On chord/timbre change | Server (Path 1) |
| Partial frequency bars | On chord/timbre change | Server (Path 1) |
| Waveform oscilloscope | **60 fps** (16.7 ms) | Client audio buffer |
| Spectrogram | **30 fps** (33.3 ms) | Client-side FFT |
| Level meter | 30 fps | Client audio buffer |

For spectrogram display, use the client-side **AnalyserNode** or a custom FFT in the AudioWorklet/Web Worker. FFT size 2048 with Hann window and 50–75% overlap provides 23.4 Hz frequency resolution with smooth updates. **No server-side FFT is needed** — consonance calculations use direct spectral data (Path 1), and spectrogram visualization uses client-side FFT on the decoded audio (Path 2).

---

## Conclusion: architectural coherence and the timbre–consonance bridge

ChordSpace's technical viability rests on one architectural decision that enables everything else: **the synthesis engine exports structured partial data alongside audio.** This dual-output design means the dissonance engine never runs FFT, the visualization receives exact spectral information, and timbre changes trigger instant consonance recalculation from compact fingerprints rather than expensive audio analysis.

The synthesis method selection should be guided by spectral predictability first, computational cost second. **Additive and modal synthesis** provide perfect spectral transparency for the dissonance engine. **Subtractive and FM synthesis** offer computable spectra with richer timbral character. **Wavetable** provides the cheapest rendering with offline-analyzable spectra. Physical modeling should be reserved for plucked-string presets where organic decay matters, accepting the need for FFT-based spectral extraction.

The Vassilakis roughness model (with its amplitude-sensitive AF-degree term at exponent 3.11) is the recommended dissonance calculation for its superior correlation with perception. For real-time performance, precomputing interval dissonance tables at 1-cent resolution per timbre reduces chord evaluation to O(K²) lookups — effectively instantaneous even at 60 fps visualization rates. Recomputation on timbre change requires ~1–5 ms for a full 1,200-entry table with 16 partials.

The streaming architecture targets **sub-100 ms latency** via Opus-encoded audio over WebSocket at 96–128 kbit/s, decoded client-side via WebAssembly in a Web Worker feeding an AudioWorklet through a SharedArrayBuffer ring buffer. The optional hybrid approach — instant client-side oscillator approximation crossfading to server-rendered audio — can push perceived latency to near-zero for a genuinely playable instrument.

The total CPU budget for synthesis is remarkably modest: even the most expensive configuration (24 voices of 32-partial additive synthesis across 4 timbres) uses under 3% of a single modern core. The real engineering challenge is architectural discipline — lock-free communication, zero-allocation rendering, SoA memory layout for SIMD, and maintaining the dual data path that keeps the consonance engine synchronized with the audio output.