# Fractal audio tiling: DSP theory, parameter interactions, and 28 musically optimized configurations

The fractal tiling algorithm is a genuine self-similar audio transform that creates rich texture by layering compressed-and-repeated copies of a signal at geometrically decreasing timescales — placing it at the intersection of granular synthesis, waveset distortion, comb filtering, and iterated function systems. **The most musically critical parameters are `scale_ratio` and `amplitude_decay`**, which together determine the spectral slope, harmonic/inharmonic character, and perceived "fractal dimension" of the output. This report provides the complete DSP theory behind the algorithm, analyzes all major parameter interactions, identifies mathematically optimal ratios, and proposes 28 specific configurations with full JSON parameter sets and DSP reasoning for each.

---

## What happens when you tile compressed copies of a signal

The core operation — downsample a signal to length `N × scale_ratio^s`, then repeat it to fill `N` samples — has a precise spectral interpretation rooted in the **periodization-discretization duality**: periodic repetition in time creates discrete spectral lines in the frequency domain.

For scale level `s`, the compressed chunk has length `L_s = N × scale_ratio^s` samples. When tiled, it creates a periodic signal with fundamental frequency **f₀(s) = fs / L_s = fs / (N × scale_ratio^s)**. The resulting spectrum consists entirely of harmonics at `k × f₀(s)`, with amplitudes determined by the spectral envelope of the compressed chunk. This is mathematically identical to a **comb filter** with teeth spaced at `f₀(s)` Hz.

As `s` increases, `L_s` shrinks and the comb teeth spread further apart. With `scale_ratio = 0.5` and `num_scales = 6`, the highest scale produces tiles only **1/64th** the original length — if the input is 1 second of audio at 44.1 kHz, that's ~689 samples per tile, creating a tiling fundamental around **64 Hz** with widely-spaced harmonics. At `num_scales = 8`, the last layer is **1/256th** the original length (~172 samples), pushing the tiling fundamental to ~256 Hz — an audible pitched buzz whose timbre is colored by the original signal's spectral envelope.

**Nearest-neighbor vs. linear interpolation** creates the most dramatic timbral fork in the algorithm. Nearest-neighbor (zero-order hold) has a frequency response of `sinc(fT)` with only **-3.9 dB** attenuation at Nyquist and sidelobes decaying as `1/f` — meaning spectral images are suppressed by only ~13 dB. The result is aggressive aliasing that folds frequencies back into the audible band as **inharmonic, metallic artifacts**. Linear interpolation's `sinc²(fT)` response provides **-7.8 dB** at Nyquist with `1/f²` sidelobe decay (~26 dB suppression), producing a noticeably smoother, darker sound with far less aliasing. For the fractalizer, nearest-neighbor is a feature: at high scale levels, aliasing dominates the compressed copy's spectrum, creating noise-like energy that contributes gritty texture when tiled.

The tile boundaries themselves matter. Unlike windowed granular synthesis (which applies Hanning/Gaussian envelopes to avoid discontinuities), the fractalizer tiles without crossfading. Each boundary creates an **impulsive discontinuity** that adds broadband click energy at the tiling rate — reinforcing the comb filter character and adding transient bite. This is structurally identical to **waveset distortion** as developed by Trevor Wishart for CDP, where repeating signal segments between zero-crossings generates pitch artifacts from any source material, even noise.

---

## Scale ratio sweet spots and the mathematics of self-similarity

Not all scale ratios are created equal. The mathematical properties of the ratio determine whether the layered tile patterns create periodic, quasi-periodic, or chaotic-feeling structures.

**`scale_ratio = 0.5` (octave)** is the "purest" choice. Every layer's period divides evenly into every other layer's period — tiles at scales 1/2, 1/4, 1/8, 1/16 all align at the original signal's boundaries. The spectral content is entirely harmonic: all comb filter peaks fall on harmonics of a common fundamental. This creates maximum spectral regularity and consonance with harmonic source material. The perceptual effect exploits **octave equivalence** — each layer is heard as the "same" pitch class at different registers, fusing into a unified but spectrally enriched texture.

**`scale_ratio = 0.618` (1/φ, golden ratio)** is the mathematically optimal choice for structured complexity. The golden ratio is provably the **most irrational number** — its continued fraction `[0; 1, 1, 1, ...]` converges more slowly than any other irrational, meaning tile periods are *maximally non-commensurate*. They never align, creating quasi-periodic patterns that exhibit near-repeats at Fibonacci-number intervals. The Three-Gap Theorem guarantees that the tile boundaries distribute with at most three distinct gap sizes, all related by φ — the most uniform possible non-periodic distribution. This is the audio equivalent of a **quasicrystal**: long-range order without translational periodicity.

**`scale_ratio = 0.667` (2/3, inverse perfect fifth)** offers an excellent middle ground. The ratio is simple enough to create consonant relationships between layers (related by musical fifths), but since `log₂(3)` is irrational, layers never perfectly align at octave boundaries. This produces the same **Pythagorean comma** micro-beating that gives stacked fifths their characteristic richness. Layers at 2/3, 4/9, 8/27, 16/81 of original length create a harmonically rich but subtly quasi-periodic structure.

**`scale_ratio = 0.75` (3/4, inverse perfect fourth)** and **`scale_ratio = 0.8` (4/5, inverse major third)** extend this logic into other just-intonation relationships, each with progressively more complex harmonic interactions. The key insight from William Sethares' research is that consonance depends on the relationship between the source signal's spectrum and the scale ratio — **harmonic source material benefits from simple rational ratios; inharmonic sources (metallic percussion, noise) can benefit from irrational ratios** where the lack of harmonic alignment doesn't create clashing.

**`scale_ratio = 0.333` (1/3)** deserves special mention: it creates a **Cantor-set-like** structure (layers at 1/3, 1/9, 1/27, 1/81), directly connecting the algorithm to fractal set theory. The rapid scale contraction means extreme compression at higher levels, producing buzzy, pitch-artifact-rich textures very quickly.

### Amplitude decay and spectral slope

The `amplitude_decay` parameter controls the relative weight of each scale layer and directly determines the output's **spectral slope**. For a power spectral density proportional to `1/f^β`:

- **`amplitude_decay = √(scale_ratio)` → β = 1 (pink noise / 1/f)**. For `scale_ratio = 0.5`, this means `amplitude_decay ≈ 0.707`. This is the sweet spot: Voss and Clarke (1975) showed that music itself exhibits 1/f spectral characteristics, and pink noise represents the perceptually optimal balance between predictability and surprise. **This is the single most important formula for the fractalizer**: `amplitude_decay = √(scale_ratio)` produces the most naturally "musical" spectral balance.
- **`amplitude_decay = scale_ratio` → β = 2 (brown noise / 1/f²)**. For `scale_ratio = 0.5`, `amplitude_decay = 0.5`. Higher layers are heavily attenuated — smooth, bass-heavy, with minimal high-frequency fractal detail.
- **`amplitude_decay = 1.0` → β = 0 (white noise)**. All layers at equal amplitude — maximum roughness and spectral density, often harsh and noisy.

---

## Iteration dynamics: from warm saturation to spectral explosion

The iteration system (1–4 passes through the fractalizer with tanh saturation between passes) creates an **iterated function system (IFS)** in the formal mathematical sense. IFS theory via the Banach Fixed-Point Theorem guarantees that if the composite mapping is contractive (which it is when `iter_decay < 1`), repeated application converges to a unique **audio attractor** — a self-similar signal that is a fixed point of the transform, independent of the input. With only 1–4 iterations and reasonable decay values, the input still strongly shapes the output, but the fractal structure progressively dominates.

**tanh waveshaping** adds exclusively **odd harmonics** (3rd, 5th, 7th...) due to its odd symmetry. At low drive (saturation < 0.3), the Taylor expansion `tanh(x) ≈ x - x³/3` means only a gentle third harmonic appears — near-transparent processing. At moderate drive (0.3–0.7), the spectrum thickens progressively as higher odd harmonics emerge. At high drive (>0.8), the output approaches a square wave with harmonics decaying as `1/n`. Among common waveshapers, tanh is the "hardest" — it transitions most abruptly from linear to saturated, creating a more aggressive character than arctan or algebraic soft-clipping.

The critical interaction is between **saturation and spectral density**. The fractalizer's output contains components at many non-harmonically-related frequencies (from tiling at different scale levels). When this passes through tanh, **intermodulation distortion** generates sum and difference products between every pair of frequency components. The number of IMD products scales as `O(N²)` for `N` input frequencies per iteration. Over `K` iterations, total spectral components grow as roughly `O(N^(2^K))` — a **spectral explosion** that rapidly fills the spectrum. At 2 iterations with moderate saturation, this creates warmth and density. At 4 iterations with high saturation, it produces a noise-like spectral floor — the intermodulation products become so dense they approximate broadband noise colored by the fractal structure.

**Stability regimes** map cleanly to musical character:

- **`iter_decay` 0.3–0.5, saturation < 0.3**: Strongly contractive. Each iteration refines the fractal texture without significant harmonic addition. Clean, architectural self-similarity.
- **`iter_decay` 0.5–0.7, saturation 0.3–0.6**: The productive sweet spot. Rich harmonic enrichment with controlled intermodulation. Warm, dense, evolving texture.
- **`iter_decay` 0.7–0.9, saturation 0.6–0.8**: Approaching criticality. Dense spectral content with prominent intermodulation artifacts. Aggressive, saturated, industrial character.
- **`iter_decay` 0.9–1.0, saturation > 0.8**: Near-critical. The Lipschitz constant of the composite mapping approaches 1. Behavior becomes unpredictable — quasi-periodic or chaotic-like textures. The tanh acts as a "safety valve" (output always bounded in [-1, 1]), preventing true divergence, but the *character* transitions from structured to chaotic.

---

## The spectral mode: STFT fractalization vs. time-domain

When the compress-and-tile operation is applied to **STFT magnitude frames** rather than raw samples, the fractal structure is created in the frequency domain rather than the time domain. Compressing and tiling the magnitude vector creates **spectral periodicity** — the spectrum becomes self-similar across frequency, which is qualitatively different from temporal self-similarity.

Time-domain fractalization creates **temporal self-similarity**: the waveform contains nested copies of itself at different timescales, producing audible comb filtering, pitched tiling artifacts, and rhythmic structure. Spectral fractalization creates **spectral self-similarity**: the frequency spectrum contains nested copies of its own shape at different frequency scales, producing timbral transformation without the pitched-buzz artifacts of time-domain tiling.

**Perceptual differences** are dramatic. Time-domain processing produces crunchy, rhythmic, pitched textures with clear distortion character. STFT processing produces smoother, more "smeared" results akin to spectral freezing or multiband compression — quieter spectral components are effectively boosted relative to louder ones (the tiling fills in spectral gaps), creating a denser, more filled-in spectrum without intermodulation artifacts between frequency bins. The phase coherence issues inherent in STFT manipulation add a subtle **metallic, phasey coloration** that can range from ethereal shimmer (at low blend values) to robotic artifacts (at high values).

**Window size** controls the time-frequency tradeoff via the Heisenberg-Gabor limit (`Δt × Δf ≥ 1/4π`). Small windows (256–512 samples) preserve transients but blur frequency resolution — useful for percussive material. Large windows (4096–8192) provide sharp frequency resolution but smear transients — ideal for sustained/ambient material. The Paulstretch algorithm demonstrates the extreme case: windows of 65536+ samples with randomized phases create the characteristic "frozen, ethereal drone" from any source material.

**Blending** (`spectral` parameter between 0 and 1) allows continuous morphing between time-domain character (gritty, rhythmic, pitched) and spectral character (smooth, timbral, smeared). A 50/50 blend creates an unusual hybrid: temporal fractal structure with spectral smoothing, producing a texture that has the density of time-domain processing but the smoothness of spectral processing.

---

## Pre-filter and post-filter strategies that actually matter

**Pre-highpass filtering (80–200 Hz)** is the single most impactful filter choice. Sub-bass content passing through the fractalizer creates massive intermodulation distortion when it encounters tanh saturation — every bass frequency generates sum/difference products with every other frequency component, producing what mixing engineers call "mud." A HP at 100–150 Hz before fractalization eliminates this problem at the source. This mirrors standard practice in distortion pedal design and the Aphex Aural Exciter principle (highpass → distort → blend with clean signal to add harmonics without muddying the fundamental).

**Pre-bandpass filtering** enables targeted fractalization of specific frequency ranges. BP at 500–3000 Hz isolates the midrange "presence" region, producing focused fractal texture without bass mud or harsh treble multiplication. BP at 2000–8000 Hz creates an "exciter" effect where only upper harmonics are fractalized. Narrow Q values (5–10) create resonant coloration that the fractalizer then multiplies into self-similar harmonic structures.

**Pre-lowpass filtering (2000–5000 Hz)** prevents high-frequency content from multiplying through iterations into ear-fatiguing harshness. This mirrors how guitar speaker cabinets naturally roll off above ~4 kHz, which is why distorted guitar sounds musical rather than abrasive. Essential for vocal processing and any application where the fractalizer will be run at high iteration counts.

**Post-lowpass filtering** tames aliasing artifacts from nearest-neighbor interpolation and harmonic content from saturation. A gentle LP at 8–12 kHz preserves "air" while removing grating artifacts. A steeper LP at 3–5 kHz creates vintage/lo-fi character. When bitcrushing and sample rate reduction are active, the post-LP becomes critical for shaping how much digital grit to retain — essentially functioning as a virtual speaker cabinet.

**Post-highpass filtering** removes sub-frequency rumble that can accumulate through iterations (especially with higher `num_scales` where very short tiles create artifacts below the intended frequency range).

---

## 28 parameter configurations organized by musical category

### Subtle enhancement

**1. Harmonic Exciter**
Adds shimmering upper harmonics to vocals or acoustic guitar using the Aphex principle: highpass pre-filter isolates upper-mid content, minimal fractalization adds harmonics, low wet mix blends them back. The octave scale ratio ensures all added content is harmonically related. Two scales at 0.5 means copies at 1/2 and 1/4 original length — creating octave-spaced harmonic reinforcement. Linear interpolation keeps it smooth.

```json
{
  "name": "Harmonic Exciter",
  "num_scales": 2,
  "scale_ratio": 0.5,
  "amplitude_decay": 0.7,
  "interp": "linear",
  "reverse_scales": false,
  "scale_offset": 0.0,
  "iterations": 1,
  "iter_decay": 0.8,
  "saturation": 0.1,
  "spectral": 0.0,
  "window_size": 2048,
  "filter_type": "HP",
  "filter_freq": 3000,
  "filter_q": 0.7,
  "post_filter_type": "LP",
  "post_filter_freq": 14000,
  "crush": 0.0,
  "decimate": 0.0,
  "gate": 0.0,
  "wet_dry": 0.2,
  "output_gain": 0.5
}
```

**2. Warm Thickener**
Adds body and warmth to thin synth patches. The golden ratio scale ratio creates quasi-periodic reinforcement that thickens without obvious comb filtering. Amplitude decay at √0.618 ≈ 0.786 produces pink-noise spectral balance across layers. Gentle saturation adds even more warmth through odd harmonics. Spectral blend at 0.3 softens the time-domain tiling artifacts.

```json
{
  "name": "Warm Thickener",
  "num_scales": 3,
  "scale_ratio": 0.618,
  "amplitude_decay": 0.786,
  "interp": "linear",
  "reverse_scales": false,
  "scale_offset": 0.618,
  "iterations": 2,
  "iter_decay": 0.6,
  "saturation": 0.25,
  "spectral": 0.3,
  "window_size": 2048,
  "filter_type": "HP",
  "filter_freq": 80,
  "filter_q": 0.7,
  "post_filter_type": "LP",
  "post_filter_freq": 10000,
  "crush": 0.0,
  "decimate": 0.0,
  "gate": 0.0,
  "wet_dry": 0.35,
  "output_gain": 0.5
}
```

**3. Vocal Shimmer**
Creates an airy doubling effect on vocals. Pure spectral mode avoids the pitched-buzz artifacts that time-domain tiling would create on monophonic vocal content. The large STFT window captures vocal formant structure with high frequency resolution. Two scales with gentle decay add subtle spectral self-similarity — reinforcing formant peaks without adding distortion. Post-LP at 12 kHz prevents sibilance buildup.

```json
{
  "name": "Vocal Shimmer",
  "num_scales": 2,
  "scale_ratio": 0.75,
  "amplitude_decay": 0.6,
  "interp": "linear",
  "reverse_scales": false,
  "scale_offset": 0.25,
  "iterations": 1,
  "iter_decay": 0.7,
  "saturation": 0.05,
  "spectral": 1.0,
  "window_size": 4096,
  "filter_type": "HP",
  "filter_freq": 200,
  "filter_q": 0.7,
  "post_filter_type": "LP",
  "post_filter_freq": 12000,
  "crush": 0.0,
  "decimate": 0.0,
  "gate": 0.0,
  "wet_dry": 0.25,
  "output_gain": 0.5
}
```

**4. Guitar Presence**
Adds midrange presence and harmonic complexity to electric guitar. The bandpass pre-filter at 800 Hz isolates the "body" frequencies. Scale ratio of 2/3 creates perfect-fifth harmonic relationships that complement guitar's natural overtone series. Nearest-neighbor interpolation adds subtle grit appropriate for rock/blues tones. Post-LP at 6 kHz mimics speaker cabinet rolloff.

```json
{
  "name": "Guitar Presence",
  "num_scales": 3,
  "scale_ratio": 0.667,
  "amplitude_decay": 0.75,
  "interp": "nearest",
  "reverse_scales": false,
  "scale_offset": 0.0,
  "iterations": 1,
  "iter_decay": 0.7,
  "saturation": 0.3,
  "spectral": 0.0,
  "window_size": 1024,
  "filter_type": "BP",
  "filter_freq": 800,
  "filter_q": 1.5,
  "post_filter_type": "LP",
  "post_filter_freq": 6000,
  "crush": 0.0,
  "decimate": 0.0,
  "gate": 0.0,
  "wet_dry": 0.3,
  "output_gain": 0.5
}
```

### Rhythmic and percussive textures

**5. Fractal Stutter**
Creates rhythmic stuttering from sustained input. High `num_scales` with small scale ratio means the upper layers are extremely short tiles (at scale 6 with ratio 0.333: ~1/729th original length), creating rapid buzzing pitched artifacts. The noise gate with moderate threshold chops the output into rhythmic bursts, as only the loudest transient-aligned moments pass through. Nearest-neighbor maximizes the aliased grit.

```json
{
  "name": "Fractal Stutter",
  "num_scales": 6,
  "scale_ratio": 0.333,
  "amplitude_decay": 0.6,
  "interp": "nearest",
  "reverse_scales": false,
  "scale_offset": 0.0,
  "iterations": 2,
  "iter_decay": 0.5,
  "saturation": 0.5,
  "spectral": 0.0,
  "window_size": 512,
  "filter_type": "HP",
  "filter_freq": 150,
  "filter_q": 0.7,
  "post_filter_type": "LP",
  "post_filter_freq": 8000,
  "crush": 0.15,
  "decimate": 0.1,
  "gate": 0.4,
  "wet_dry": 0.7,
  "output_gain": 0.5
}
```

**6. Metallic Percussion**
Transforms drum hits into metallic, pitched percussion. The Cantor-like 1/3 ratio creates three copies per tile at scale 1, nine at scale 2 — generating inharmonic frequency relationships characteristic of metallic instruments (bells, gongs). High saturation between 3 iterations creates dense intermodulation products. The BP pre-filter at 1 kHz focuses the effect on the attack transient's spectral core.

```json
{
  "name": "Metallic Percussion",
  "num_scales": 4,
  "scale_ratio": 0.333,
  "amplitude_decay": 0.58,
  "interp": "nearest",
  "reverse_scales": false,
  "scale_offset": 0.333,
  "iterations": 3,
  "iter_decay": 0.5,
  "saturation": 0.7,
  "spectral": 0.2,
  "window_size": 512,
  "filter_type": "BP",
  "filter_freq": 1000,
  "filter_q": 2.0,
  "post_filter_type": "LP",
  "post_filter_freq": 10000,
  "crush": 0.0,
  "decimate": 0.0,
  "gate": 0.15,
  "wet_dry": 0.6,
  "output_gain": 0.5
}
```

**7. Polyrhythmic Ghost**
Creates polyrhythmic interference patterns from simple rhythmic input. The golden ratio ensures tile periods are maximally non-commensurate — the tiling patterns from different scale levels create quasi-periodic rhythmic interactions that never exactly repeat. Low saturation preserves the rhythmic transients. Reversed scales create backwards ghost echoes that interleave with the forward pattern.

```json
{
  "name": "Polyrhythmic Ghost",
  "num_scales": 5,
  "scale_ratio": 0.618,
  "amplitude_decay": 0.786,
  "interp": "linear",
  "reverse_scales": true,
  "scale_offset": 0.382,
  "iterations": 1,
  "iter_decay": 0.7,
  "saturation": 0.15,
  "spectral": 0.0,
  "window_size": 1024,
  "filter_type": "HP",
  "filter_freq": 100,
  "filter_q": 0.7,
  "post_filter_type": "bypass",
  "post_filter_freq": 20000,
  "crush": 0.0,
  "decimate": 0.0,
  "gate": 0.1,
  "wet_dry": 0.5,
  "output_gain": 0.5
}
```

**8. Drum Machine Crunch**
Lo-fi drum processing inspired by vintage hardware samplers (SP-1200, MPC60). Moderate fractalization adds harmonic density, then bitcrushing at 8-bit equivalent and sample rate reduction create the classic gritty, punchy character. Post-LP at 8 kHz tames the harshest aliasing while retaining the crunch. The octave ratio keeps everything harmonically anchored to preserve the kick's fundamental.

```json
{
  "name": "Drum Machine Crunch",
  "num_scales": 2,
  "scale_ratio": 0.5,
  "amplitude_decay": 0.5,
  "interp": "nearest",
  "reverse_scales": false,
  "scale_offset": 0.0,
  "iterations": 1,
  "iter_decay": 0.6,
  "saturation": 0.4,
  "spectral": 0.0,
  "window_size": 512,
  "filter_type": "HP",
  "filter_freq": 60,
  "filter_q": 0.7,
  "post_filter_type": "LP",
  "post_filter_freq": 8000,
  "crush": 0.5,
  "decimate": 0.35,
  "gate": 0.1,
  "wet_dry": 0.6,
  "output_gain": 0.55
}
```

### Ambient and pad textures

**9. Crystalline Freeze**
Creates frozen, shimmering drones from any source. Full spectral mode with a large 8192-sample window captures the spectral snapshot with high frequency resolution, then the compress-and-tile operation creates spectral self-similarity — the spectrum contains nested copies of itself. Four iterations with moderate decay converge toward the IFS attractor, producing a stable, self-similar spectral structure. Phase randomization from the STFT processing adds the characteristic Paulstretch-like diffusion.

```json
{
  "name": "Crystalline Freeze",
  "num_scales": 5,
  "scale_ratio": 0.618,
  "amplitude_decay": 0.786,
  "interp": "linear",
  "reverse_scales": false,
  "scale_offset": 0.618,
  "iterations": 4,
  "iter_decay": 0.5,
  "saturation": 0.2,
  "spectral": 1.0,
  "window_size": 8192,
  "filter_type": "bypass",
  "filter_freq": 1000,
  "filter_q": 0.7,
  "post_filter_type": "LP",
  "post_filter_freq": 12000,
  "crush": 0.0,
  "decimate": 0.0,
  "gate": 0.0,
  "wet_dry": 0.8,
  "output_gain": 0.45
}
```

**10. Deep Space Pad**
Warm, evolving ambient texture. Brown-noise spectral slope (amplitude_decay = scale_ratio = 0.5) heavily favors lower scale layers, creating a deep, bass-heavy fractal texture. The reversed scales add backward grain movement for a sense of "breathing." Low saturation and spectral blend soften everything. Post-LP at 6 kHz creates a warm, distant quality.

```json
{
  "name": "Deep Space Pad",
  "num_scales": 6,
  "scale_ratio": 0.5,
  "amplitude_decay": 0.5,
  "interp": "linear",
  "reverse_scales": true,
  "scale_offset": 0.25,
  "iterations": 3,
  "iter_decay": 0.45,
  "saturation": 0.15,
  "spectral": 0.6,
  "window_size": 4096,
  "filter_type": "LP",
  "filter_freq": 4000,
  "filter_q": 0.5,
  "post_filter_type": "LP",
  "post_filter_freq": 6000,
  "crush": 0.0,
  "decimate": 0.0,
  "gate": 0.0,
  "wet_dry": 0.7,
  "output_gain": 0.45
}
```

**11. Fibonacci Shimmer**
The golden ratio appears three times: as scale_ratio, amplitude_decay (√φ for pink noise), and scale_offset. This creates the most mathematically "pure" quasi-periodic fractal structure, where near-repeats occur at Fibonacci intervals. Full spectral mode prevents time-domain tiling buzz, producing a smooth, complex shimmer. The perfect fourth pre-filter Q creates a gentle resonant peak that the fractalizer multiplies into a self-similar harmonic stack.

```json
{
  "name": "Fibonacci Shimmer",
  "num_scales": 5,
  "scale_ratio": 0.618,
  "amplitude_decay": 0.786,
  "interp": "linear",
  "reverse_scales": false,
  "scale_offset": 0.618,
  "iterations": 3,
  "iter_decay": 0.55,
  "saturation": 0.1,
  "spectral": 0.85,
  "window_size": 4096,
  "filter_type": "BP",
  "filter_freq": 2000,
  "filter_q": 0.8,
  "post_filter_type": "LP",
  "post_filter_freq": 14000,
  "crush": 0.0,
  "decimate": 0.0,
  "gate": 0.0,
  "wet_dry": 0.55,
  "output_gain": 0.48
}
```

**12. Granular Cloud**
Emulates the dense, cloudy texture of Mutable Instruments Clouds/Beads at high density settings. Many scales with the major-third ratio (0.8) create closely-spaced tile layers that overlap densely. Spectral mode at 0.5 blends time and spectral fractal structures. The slow decay across layers and high iteration count build toward the IFS attractor — a stable, cloud-like texture.

```json
{
  "name": "Granular Cloud",
  "num_scales": 7,
  "scale_ratio": 0.8,
  "amplitude_decay": 0.894,
  "interp": "linear",
  "reverse_scales": false,
  "scale_offset": 0.5,
  "iterations": 3,
  "iter_decay": 0.5,
  "saturation": 0.2,
  "spectral": 0.5,
  "window_size": 2048,
  "filter_type": "HP",
  "filter_freq": 120,
  "filter_q": 0.7,
  "post_filter_type": "LP",
  "post_filter_freq": 11000,
  "crush": 0.0,
  "decimate": 0.0,
  "gate": 0.0,
  "wet_dry": 0.65,
  "output_gain": 0.47
}
```

### Aggressive and industrial

**13. Waveform Destroyer**
Maximum fractal aggression. Eight scales at 0.5 ratio means the last layer is 1/256th the original length — pure buzzing pitch artifact. Nearest-neighbor creates maximum aliasing at every scale. Three iterations with high saturation produce cascading intermodulation (spectral components grow as roughly `N^8` over 3 iterations). No pre-filter lets sub-bass create maximum IMD "mud." Post-HP removes the resulting subsonic garbage. The 6-bit crush adds quantization distortion on top of everything.

```json
{
  "name": "Waveform Destroyer",
  "num_scales": 8,
  "scale_ratio": 0.5,
  "amplitude_decay": 0.85,
  "interp": "nearest",
  "reverse_scales": false,
  "scale_offset": 0.0,
  "iterations": 3,
  "iter_decay": 0.75,
  "saturation": 0.85,
  "spectral": 0.0,
  "window_size": 512,
  "filter_type": "bypass",
  "filter_freq": 1000,
  "filter_q": 0.7,
  "post_filter_type": "LP",
  "post_filter_freq": 7000,
  "crush": 0.65,
  "decimate": 0.2,
  "gate": 0.05,
  "wet_dry": 0.85,
  "output_gain": 0.45
}
```

**14. Feedback Howl**
Pushes the system toward its stability boundary. Decay near 1.0 means each iteration barely attenuates — the system is near-critical. High saturation creates dense odd-harmonic content. Four iterations approach the limit of contractive convergence. The result is a howling, oscillating texture where the tanh saturation is the only thing preventing runaway feedback. The bandpass pre-filter constrains the chaos to a manageable frequency range.

```json
{
  "name": "Feedback Howl",
  "num_scales": 4,
  "scale_ratio": 0.5,
  "amplitude_decay": 0.7,
  "interp": "nearest",
  "reverse_scales": false,
  "scale_offset": 0.0,
  "iterations": 4,
  "iter_decay": 0.95,
  "saturation": 0.9,
  "spectral": 0.0,
  "window_size": 1024,
  "filter_type": "BP",
  "filter_freq": 1500,
  "filter_q": 3.0,
  "post_filter_type": "LP",
  "post_filter_freq": 8000,
  "crush": 0.0,
  "decimate": 0.0,
  "gate": 0.0,
  "wet_dry": 0.75,
  "output_gain": 0.4
}
```

**15. Industrial Grind**
Aggressive texture combining fractalization with heavy bitcrushing and sample rate reduction. The 1/3 scale ratio creates Cantor-set structures that are inherently harsh and complex. Nearest-neighbor aliasing, 4-bit crushing, and sample rate reduction at 0.5 (roughly 22 kHz → 11 kHz equivalent) produce severe frequency folding. The noise gate with high threshold creates rhythmic chopping — only the loudest peaks survive, producing a pulsing industrial grind.

```json
{
  "name": "Industrial Grind",
  "num_scales": 5,
  "scale_ratio": 0.333,
  "amplitude_decay": 0.7,
  "interp": "nearest",
  "reverse_scales": false,
  "scale_offset": 0.0,
  "iterations": 2,
  "iter_decay": 0.65,
  "saturation": 0.75,
  "spectral": 0.0,
  "window_size": 512,
  "filter_type": "HP",
  "filter_freq": 80,
  "filter_q": 0.7,
  "post_filter_type": "LP",
  "post_filter_freq": 5000,
  "crush": 0.8,
  "decimate": 0.5,
  "gate": 0.35,
  "wet_dry": 0.8,
  "output_gain": 0.5
}
```

**16. Pythagorean Fuzz**
A more "musical" aggression that uses the perfect-fifth ratio (2/3) to create harmonically related distortion. Four scales create layers at 2/3, 4/9, 8/27, 16/81 — Pythagorean stacking. The moderate saturation adds warmth (odd harmonics) without the chaotic intermodulation of extreme settings. The post-LP at 5 kHz mimics a guitar cabinet, keeping the fuzz musical. Pre-HP removes bass intermodulation.

```json
{
  "name": "Pythagorean Fuzz",
  "num_scales": 4,
  "scale_ratio": 0.667,
  "amplitude_decay": 0.75,
  "interp": "nearest",
  "reverse_scales": false,
  "scale_offset": 0.0,
  "iterations": 2,
  "iter_decay": 0.6,
  "saturation": 0.6,
  "spectral": 0.0,
  "window_size": 1024,
  "filter_type": "HP",
  "filter_freq": 120,
  "filter_q": 0.7,
  "post_filter_type": "LP",
  "post_filter_freq": 5000,
  "crush": 0.1,
  "decimate": 0.0,
  "gate": 0.05,
  "wet_dry": 0.7,
  "output_gain": 0.5
}
```

### Glitch and experimental

**17. Digital Decay**
Simulates data corruption and digital degradation. Extreme sample rate reduction creates metallic frequency folding. The reversed scales create temporally inverted micro-grains that sound like data being read backwards. High bitcrushing quantizes the fractal complexity into stepped, crystalline artifacts. The noise gate creates stuttering gaps that suggest data dropouts.

```json
{
  "name": "Digital Decay",
  "num_scales": 6,
  "scale_ratio": 0.4,
  "amplitude_decay": 0.65,
  "interp": "nearest",
  "reverse_scales": true,
  "scale_offset": 0.73,
  "iterations": 2,
  "iter_decay": 0.55,
  "saturation": 0.45,
  "spectral": 0.15,
  "window_size": 256,
  "filter_type": "bypass",
  "filter_freq": 1000,
  "filter_q": 0.7,
  "post_filter_type": "bypass",
  "post_filter_freq": 20000,
  "crush": 0.7,
  "decimate": 0.65,
  "gate": 0.25,
  "wet_dry": 0.75,
  "output_gain": 0.5
}
```

**18. Cantor Dust**
Named for the Cantor set, this uses scale_ratio = 1/3 to create a structure directly analogous to the iterated removal of middle thirds. Seven scales create layers down to 1/2187th of the original length — extremely short tiles producing pitched buzz artifacts in the high hundreds of Hz. The high scale count combined with equal amplitude weighting (decay near 1.0) creates a white-noise-like spectral slope — maximum fractal roughness. The result is a buzzing, insect-like texture.

```json
{
  "name": "Cantor Dust",
  "num_scales": 7,
  "scale_ratio": 0.333,
  "amplitude_decay": 0.95,
  "interp": "nearest",
  "reverse_scales": false,
  "scale_offset": 0.0,
  "iterations": 1,
  "iter_decay": 0.7,
  "saturation": 0.3,
  "spectral": 0.0,
  "window_size": 256,
  "filter_type": "HP",
  "filter_freq": 60,
  "filter_q": 0.7,
  "post_filter_type": "LP",
  "post_filter_freq": 16000,
  "crush": 0.0,
  "decimate": 0.0,
  "gate": 0.0,
  "wet_dry": 0.6,
  "output_gain": 0.45
}
```

**19. Spectral Scatter**
Applies the fractal operation purely in the STFT domain with a very small window (256 samples) to maximize temporal resolution at the expense of frequency resolution. This creates blurry, "scattered" spectral copies that sound like the audio is being diffracted through a prism. Three iterations build the spectral self-similarity. The 50/50 blend with reverse scales creates an eerie, disorienting texture where forward and backward spectral fragments interleave.

```json
{
  "name": "Spectral Scatter",
  "num_scales": 4,
  "scale_ratio": 0.618,
  "amplitude_decay": 0.786,
  "interp": "linear",
  "reverse_scales": true,
  "scale_offset": 0.382,
  "iterations": 3,
  "iter_decay": 0.5,
  "saturation": 0.35,
  "spectral": 0.7,
  "window_size": 256,
  "filter_type": "bypass",
  "filter_freq": 1000,
  "filter_q": 0.7,
  "post_filter_type": "bypass",
  "post_filter_freq": 20000,
  "crush": 0.0,
  "decimate": 0.0,
  "gate": 0.0,
  "wet_dry": 0.65,
  "output_gain": 0.5
}
```

**20. Glitch Cascade**
Rapid-fire glitch texture. Four iterations with moderate-high saturation create cascading intermodulation. The 0.4 scale ratio places it between the simple 1/3 and 1/2 ratios, creating irrational-like tile relationships that resist settling into periodic patterns. Nearest-neighbor aliasing and moderate bitcrushing add digital artifacts. The fast noise gate (threshold 0.3) creates rhythmic chopping synchronized to the fractal structure's amplitude envelope.

```json
{
  "name": "Glitch Cascade",
  "num_scales": 5,
  "scale_ratio": 0.4,
  "amplitude_decay": 0.7,
  "interp": "nearest",
  "reverse_scales": false,
  "scale_offset": 0.5,
  "iterations": 4,
  "iter_decay": 0.55,
  "saturation": 0.65,
  "spectral": 0.0,
  "window_size": 512,
  "filter_type": "HP",
  "filter_freq": 200,
  "filter_q": 1.0,
  "post_filter_type": "LP",
  "post_filter_freq": 9000,
  "crush": 0.35,
  "decimate": 0.15,
  "gate": 0.3,
  "wet_dry": 0.7,
  "output_gain": 0.48
}
```

### Sound design: foley, transitions, and risers

**21. Tension Riser**
Designed for cinematic tension builds. The extreme number of scales (8) with a moderate ratio creates a dense stack of increasingly buzzy layers. The reversed scales add upward-sweeping character as the reversed tiles create ascending pitch artifacts when the source material has dynamics. Spectral mode at 0.4 adds an ethereal, phasey quality. Progressive saturation across 3 iterations builds intensity. Use with a signal that increases in level over time for a natural "riser" effect.

```json
{
  "name": "Tension Riser",
  "num_scales": 8,
  "scale_ratio": 0.618,
  "amplitude_decay": 0.786,
  "interp": "linear",
  "reverse_scales": true,
  "scale_offset": 0.0,
  "iterations": 3,
  "iter_decay": 0.6,
  "saturation": 0.5,
  "spectral": 0.4,
  "window_size": 2048,
  "filter_type": "HP",
  "filter_freq": 100,
  "filter_q": 0.7,
  "post_filter_type": "LP",
  "post_filter_freq": 14000,
  "crush": 0.0,
  "decimate": 0.0,
  "gate": 0.0,
  "wet_dry": 0.75,
  "output_gain": 0.5
}
```

**22. Foley Transformer**
Transforms everyday foley sounds into alien textures. The small window STFT processing (512) preserves transient character while the spectral fractalization creates formant-like resonances from any source material. The 2/3 scale ratio creates musical-fifth relationships between spectral copies — adding pitched, almost vocal quality to mechanical sounds. Moderate bitcrushing adds grit. Effective on: water drops, paper crumpling, footsteps, metal clanking.

```json
{
  "name": "Foley Transformer",
  "num_scales": 4,
  "scale_ratio": 0.667,
  "amplitude_decay": 0.7,
  "interp": "nearest",
  "reverse_scales": false,
  "scale_offset": 0.333,
  "iterations": 2,
  "iter_decay": 0.55,
  "saturation": 0.4,
  "spectral": 0.6,
  "window_size": 512,
  "filter_type": "bypass",
  "filter_freq": 1000,
  "filter_q": 0.7,
  "post_filter_type": "LP",
  "post_filter_freq": 10000,
  "crush": 0.25,
  "decimate": 0.1,
  "gate": 0.05,
  "wet_dry": 0.6,
  "output_gain": 0.5
}
```

**23. Impact Tail**
Designed to extend and texturize the tail of impact sounds (hits, explosions, drops). The fractal tiling multiplies the decay tail into self-similar repetitions that extend the perceived duration. Full spectral mode at a large window size smears the impact's transient into a resonant "ring" while preserving the spectral character. High iteration count converges toward a stable resonance. The noise gate can be adjusted to shape the tail length by cutting off the processed signal when it decays below threshold.

```json
{
  "name": "Impact Tail",
  "num_scales": 5,
  "scale_ratio": 0.5,
  "amplitude_decay": 0.707,
  "interp": "linear",
  "reverse_scales": false,
  "scale_offset": 0.0,
  "iterations": 4,
  "iter_decay": 0.5,
  "saturation": 0.3,
  "spectral": 0.8,
  "window_size": 4096,
  "filter_type": "HP",
  "filter_freq": 60,
  "filter_q": 0.7,
  "post_filter_type": "LP",
  "post_filter_freq": 8000,
  "crush": 0.0,
  "decimate": 0.0,
  "gate": 0.15,
  "wet_dry": 0.65,
  "output_gain": 0.5
}
```

**24. Whoosh Texture**
Creates swooshing, whoosh-like textures from noise or broadband input. The reversed scales create a sense of directional movement as the backward tiling shifts the perceived "center of gravity" of each micro-grain. Spectral-dominant processing (0.7) smooths the result into a cohesive whoosh. LP pre-filtering at 3 kHz keeps the whoosh warm rather than hissy. Multiple iterations with gentle saturation build density.

```json
{
  "name": "Whoosh Texture",
  "num_scales": 6,
  "scale_ratio": 0.75,
  "amplitude_decay": 0.866,
  "interp": "linear",
  "reverse_scales": true,
  "scale_offset": 0.5,
  "iterations": 2,
  "iter_decay": 0.5,
  "saturation": 0.2,
  "spectral": 0.7,
  "window_size": 2048,
  "filter_type": "LP",
  "filter_freq": 3000,
  "filter_q": 0.5,
  "post_filter_type": "LP",
  "post_filter_freq": 8000,
  "crush": 0.0,
  "decimate": 0.0,
  "gate": 0.0,
  "wet_dry": 0.8,
  "output_gain": 0.5
}
```

### Filter, crush, and decimate combinations for novel timbres

**25. Telephone From Hell**
Bandpass pre-filter simulates a telephone frequency response (300–3000 Hz), then extreme fractalization, bitcrushing, and sample rate reduction destroy the signal. The post-LP at 3.5 kHz maintains the "telephone" character while the fractal processing and bit reduction create the impression of a transmission degrading in real-time. Effective on voice for horror/sci-fi sound design.

```json
{
  "name": "Telephone From Hell",
  "num_scales": 5,
  "scale_ratio": 0.5,
  "amplitude_decay": 0.6,
  "interp": "nearest",
  "reverse_scales": false,
  "scale_offset": 0.0,
  "iterations": 3,
  "iter_decay": 0.6,
  "saturation": 0.7,
  "spectral": 0.0,
  "window_size": 512,
  "filter_type": "BP",
  "filter_freq": 800,
  "filter_q": 2.5,
  "post_filter_type": "LP",
  "post_filter_freq": 3500,
  "crush": 0.6,
  "decimate": 0.55,
  "gate": 0.15,
  "wet_dry": 0.85,
  "output_gain": 0.5
}
```

**26. Resonant Fractal Bass**
Creates growling bass textures from any low-frequency input. The high-Q bandpass pre-filter at 150 Hz creates a resonant peak that the fractalizer multiplies into a self-similar stack of resonances. The octave ratio keeps the bass content harmonically related. Moderate saturation adds sub-harmonic warmth through odd-harmonic generation. No bitcrushing preserves low-end clarity. Post-LP removes harsh upper harmonics that would distract from the bass focus.

```json
{
  "name": "Resonant Fractal Bass",
  "num_scales": 3,
  "scale_ratio": 0.5,
  "amplitude_decay": 0.707,
  "interp": "linear",
  "reverse_scales": false,
  "scale_offset": 0.0,
  "iterations": 2,
  "iter_decay": 0.65,
  "saturation": 0.5,
  "spectral": 0.0,
  "window_size": 1024,
  "filter_type": "BP",
  "filter_freq": 150,
  "filter_q": 5.0,
  "post_filter_type": "LP",
  "post_filter_freq": 4000,
  "crush": 0.0,
  "decimate": 0.0,
  "gate": 0.0,
  "wet_dry": 0.65,
  "output_gain": 0.55
}
```

**27. Chiptune Fractal**
8-bit aesthetic through the fractal lens. Heavy bitcrushing (≈5-bit) and aggressive sample rate reduction create the quantized, aliased character of NES/Game Boy audio. The fractalizer operating in time domain with nearest-neighbor interpolation creates additional aliasing that stacks with the bitcrusher's artifacts. The octave ratio keeps the result pitched and "game-like." Post-LP at 6 kHz mimics the limited bandwidth of retro hardware DACs.

```json
{
  "name": "Chiptune Fractal",
  "num_scales": 3,
  "scale_ratio": 0.5,
  "amplitude_decay": 0.6,
  "interp": "nearest",
  "reverse_scales": false,
  "scale_offset": 0.0,
  "iterations": 1,
  "iter_decay": 0.7,
  "saturation": 0.5,
  "spectral": 0.0,
  "window_size": 256,
  "filter_type": "HP",
  "filter_freq": 100,
  "filter_q": 0.7,
  "post_filter_type": "LP",
  "post_filter_freq": 6000,
  "crush": 0.75,
  "decimate": 0.6,
  "gate": 0.0,
  "wet_dry": 0.9,
  "output_gain": 0.5
}
```

**28. The Attractor**
Pushes every parameter toward the IFS attractor state. Maximum iterations at near-unity decay drive the system toward convergence on its mathematical fixed point — the unique self-similar signal that is invariant under the fractal transform. The golden ratio scale creates maximum quasi-periodic complexity. Heavy saturation shapes the attractor through nonlinear dynamics. The result is increasingly independent of the input signal, converging on a unique "sound of the fractalizer itself" — a buzzing, complex, self-similar texture that is the audio equivalent of a Sierpiński triangle.

```json
{
  "name": "The Attractor",
  "num_scales": 6,
  "scale_ratio": 0.618,
  "amplitude_decay": 0.786,
  "interp": "nearest",
  "reverse_scales": false,
  "scale_offset": 0.618,
  "iterations": 4,
  "iter_decay": 0.9,
  "saturation": 0.8,
  "spectral": 0.0,
  "window_size": 1024,
  "filter_type": "bypass",
  "filter_freq": 1000,
  "filter_q": 0.7,
  "post_filter_type": "LP",
  "post_filter_freq": 10000,
  "crush": 0.0,
  "decimate": 0.0,
  "gate": 0.0,
  "wet_dry": 0.85,
  "output_gain": 0.42
}
```

---

## The three formulas that govern everything

The research converges on three mathematical relationships that determine the fractalizer's character more than any individual parameter:

**First: spectral slope.** The ratio `amplitude_decay = scale_ratio^(β/2)` determines the power spectral density slope `1/f^β`. Setting `amplitude_decay = √(scale_ratio)` produces **β = 1 (pink noise)** — the spectral signature found in music, heartbeats, and natural phenomena. This is the default "musical" setting. Values above √(scale_ratio) push toward white noise (harsher, rougher); values below push toward brown noise (smoother, bass-heavy).

**Second: periodicity vs. quasi-periodicity.** Rational `scale_ratio` values (0.5, 0.333, 0.667, 0.75) create periodic composite structures — predictable, rhythmic, harmonically locked. Irrational values (0.618, 0.707, 0.382) create quasi-periodic structures — never repeating, always structured, with near-repeats at mathematically determined intervals. The golden ratio (0.618) is the extreme case: maximally non-repeating with the most uniform distribution of near-repeats.

**Third: contraction bound.** The system converges to its IFS attractor at rate `q^n` where `q ≈ iter_decay × saturation_drive`. When `q < 0.5`, convergence is fast — 4 iterations bring the signal within 6% of the attractor, meaning the input still dominates. When `q > 0.8`, convergence is slow — 4 iterations barely begin approaching the attractor, and the system operates in a regime where small input changes create large output differences (sensitivity to initial conditions). The musically productive zone is **`q` between 0.3 and 0.7**: enough iteration to develop fractal structure, not so much that the input is lost.

These three relationships form a complete framework for navigating the parameter space. Every preset above can be understood as a specific position in the three-dimensional space defined by spectral slope, periodicity class, and contraction rate — and new presets can be designed by choosing a target position in this space and deriving the parameters algebraically.