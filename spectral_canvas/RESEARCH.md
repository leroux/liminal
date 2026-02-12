# Drawing sound: the theoretical foundations of spectral audio painting

A spectral audio painting application rests on a single powerful idea: a two-dimensional image where the horizontal axis represents time, the vertical axis represents frequency, and pixel brightness represents amplitude **is simultaneously a picture and a piece of music**. This document lays out the complete mathematical, acoustic, and psychoacoustic theory behind that idea — everything a musician needs to understand what the application does sonically before a single line of code is written.

The concept has an 80-year lineage, from Evgeny Murzin's photoelectric ANS synthesizer to Iannis Xenakis's UPIC drawing system to Aphex Twin's face hidden inside a spectrogram. What unites these experiments is the recognition that the spectrogram — the time-frequency-energy representation of sound — is the natural bridge between the eye and the ear. The same canvas can drive twelve fundamentally different synthesis algorithms, each interpreting identical brushstrokes through different mathematics to produce radically different music. Understanding those twelve interpretations, the psychoacoustic principles that determine what the listener actually hears, and the mathematical operations that connect drawing to sound — that is the purpose of this report.

---

## Eighty years of drawing sound, from glass plates to neural networks

The idea of converting images directly into sound began with **Evgeny Murzin**, a Russian optical engineer who conceived the ANS synthesizer in 1937–1938 and spent twenty years building it. Named after Alexander Nikolayevich Scriabin, the ANS used five rotating glass discs, each carrying 144 optically printed sine-wave tracks, for a total of **720 discrete microtonal frequencies** spanning ten octaves at a resolution of 1/72 octave (16.67 cents). The composer scratched marks into opaque black mastic coating a glass plate; light passed through the scratched areas onto photoelectric cells, converting the drawn pattern directly into a bank of sine waves whose amplitudes were set by the transparency at each point. Eduard Artemyev used the ANS for the soundtracks of Tarkovsky's *Solaris* (1972), *The Mirror* (1975), and *Stalker* (1979). Alfred Schnittke and Sofia Gubaidulina also composed on it. Only one surviving unit exists, housed at the Glinka State Central Museum of Musical Culture in Moscow.

Xenakis — trained as an architect under Le Corbusier — conceived music as spatial-temporal architecture and built the **UPIC system** (Unité Polyagogique Informatique du CEMAMu) in 1977 at his Paris research center. An electromagnetic stylus on a large digitizing tablet allowed composers to draw pitch trajectories, waveform shapes, and amplitude envelopes directly. The horizontal axis represented time, the vertical axis pitch. Crucially, the same drawn line could function as a compositional score element, a wavetable defining timbre, or a control signal — an early instance of the "one canvas, many interpretations" principle. Xenakis composed *Mycenae Alpha* (1978), the first work made entirely of computer-generated sounds, and wrote: "With UPIC, music becomes a game for children. They draw. They hear. They can immediately devote themselves to composition." Approximately **130 composers** worked with UPIC before the CCMIX center ceased operations in 2007.

The modern commercial incarnation is **MetaSynth** (U&I Software, active since ~1999), which maps a bitmap image to sound: horizontal position equals time, vertical position equals pitch, brightness equals amplitude, and color channels map to stereo position (red = left, green = right). MetaSynth offers multiple synthesis engines interpreting the same image — additive, wavetable, FM, subtractive, granular, and multisampler — plus an Image Filter mode where the painting acts as a time-varying spectral mask on imported audio. It was MetaSynth that Aphex Twin (Richard D. James) used to embed his grinning face into the spectrogram of the track formally titled "ΔMi−1 = −αΣn=1NDi[n][Σj∈C[i]Fji[n − 1] + Fexti[n−1]]" on the *Windowlicker* EP (1999). The face appears starting at approximately 5:27 and is visible only on a **logarithmic frequency scale** spectrogram. Other artists who have embedded spectrogram art include Venetian Snares (cat photographs in *Songs About My Cats*), Plaid (streams of threes in "3recurring"), and Nine Inch Nails (a hand image in "My Violent Heart").

The broader community of spectral tools includes Photosounder (bidirectional image-sound conversion), SpectraLayers (Sony's spectral editing environment), Virtual ANS (Alexander Zolotov's software simulation of the ANS), Coagula, HighC, iZotope RX (spectral repair), and UPISketch. Each approaches the same core premise: **the spectrogram is a canvas, and painting on it is composing**.

---

## The Fourier transform decomposes sound into visible components

Every spectral painting tool rests on a single mathematical foundation: the **Fourier transform**, which decomposes any signal into a sum of sinusoids at every frequency. The continuous form is X(f) = ∫ x(t) e^{−j2πft} dt, where the complex exponential kernel e^{−j2πft} = cos(2πft) − j·sin(2πft) acts as a probe correlating the signal with each possible frequency. For discrete signals of length N, the Discrete Fourier Transform computes X[k] = Σ x[n] e^{−j2πkn/N}, where frequency bin k corresponds to physical frequency f_k = k · f_s / N.

Each DFT coefficient is a complex number with two components that carry fundamentally different information. The **magnitude** |X[k]| = √(Re² + Im²) represents the amplitude of each frequency component — how loud that frequency is. This is what you see in a spectrogram and what the eye interprets as brightness. The **phase** φ[k] = atan2(Im, Re) represents the time offset of each sinusoidal component — where in its cycle it begins. Phase is invisible in a standard spectrogram but critical for sound quality because it determines how sinusoids align in time. When many components are phase-aligned, they create sharp transients (the snap of a pluck, the attack of a consonant). When phase is random, energy spreads out into smooth, diffuse textures.

### The spectrogram captures energy over time and frequency

To analyze how a sound's spectrum evolves, the **Short-Time Fourier Transform** applies a sliding window before computing the DFT for each time segment: STFT[m,k] = Σ x[n] · w[n − mH] · e^{−j2πkn/N}, where m is the frame index, H is the hop size between successive windows, and w[n] is a window function that tapers the signal smoothly to zero at its edges. The **spectrogram** is the squared magnitude |STFT[m,k]|² — a two-dimensional image of energy distributed over time and frequency. This image is exactly what a spectral painting canvas represents.

The choice of window function controls a fundamental tradeoff. The Hann window (w[n] = 0.5 − 0.5·cos(2πn/N)) offers a good balance between frequency resolution and spectral leakage with **−31 dB peak sidelobes**. The Hamming window achieves −43 dB sidelobes but doesn't taper to zero. The Blackman window suppresses sidelobes to −58 dB but broadens the mainlobe, blurring frequency precision.

### You cannot know exactly when and at what frequency simultaneously

The **Gabor uncertainty principle** — the signal-processing analogue of Heisenberg's principle — states that σ_t · σ_f ≥ 1/(4π). A sound event cannot be perfectly localized in both time and frequency simultaneously. In concrete terms at a 44,100 Hz sample rate: a 4096-sample analysis window gives **~10.8 Hz frequency resolution** (fine enough to distinguish notes a semitone apart above 180 Hz) but **~93 ms time resolution** (smearing transients over nearly a tenth of a second). A 256-sample window gives excellent **~5.8 ms time resolution** for percussive attacks but only **~172 Hz frequency resolution**, unable to separate notes in the bass register. Every spectral painting canvas embodies this tradeoff — its pixels can represent precise frequencies or precise moments, but not both.

Multi-resolution approaches address this limitation. The **Constant-Q Transform** (Judith C. Brown, 1991) uses logarithmically spaced frequency bins where Q = f/Δf is constant. For 12 bins per octave, Q ≈ 17, and the window length varies inversely with frequency — longer windows for bass, shorter for treble. This matches both musical pitch spacing and the cochlea's own frequency analysis. Wavelet transforms operate on the same principle, and reassigned spectrograms sharpen the display by moving each energy point to its true center of gravity.

### The Nyquist theorem sets the canvas ceiling

The Nyquist-Shannon sampling theorem states that a signal bandlimited to B Hz requires sampling at ≥ 2B samples per second. The Nyquist frequency f_s/2 is the absolute ceiling of representable frequency. At CD quality (44,100 Hz), the maximum frequency is **22,050 Hz** — covering the full range of human hearing. Any spectral content painted above this limit would alias, folding back into lower frequencies and creating spurious artifacts. This defines the top edge of the canvas.

---

## The phase problem is the central challenge of spectral painting

A spectrogram contains only magnitude — the phase is discarded. But a signal's STFT has N/2 + 1 phase values per frame alongside its N/2 + 1 magnitude values. **Discarding phase loses half the information.** This is the phase problem, and every spectral painting tool must confront it.

Phase matters most for transients and temporal structure. Randomizing phase while preserving magnitude destroys all attacks, clicks, and rhythmic events, turning them into smooth, diffuse textures. For slowly evolving, sustained, noise-like sounds, the ear is relatively phase-insensitive, and random phase can sound acceptable. Six mathematical approaches exist to reconstruct or circumvent phase:

**Random phase** assigns φ[m,k] drawn uniformly from [0, 2π] for every time-frequency bin. This is the simplest approach and the basis of the PaulStretch algorithm (created by Nasca Octavian Paul). The result sounds ethereal and dreamy — all transients dissolve into smooth, evolving textures. The famous viral audio of Justin Bieber's "U Smile" slowed 800% (posted by Nick Pittsinger in August 2010, downloaded over a million times) demonstrated this aesthetic. Random phase is ideal for ambient drone but fundamentally incapable of rhythm.

**Griffin-Lim** (Griffin and Lim, 1984) iteratively alternates projections between two constraint sets: the set of valid STFT representations (obtained by doing ISTFT then re-STFT) and the set of magnitude-consistent spectrograms (obtained by replacing the magnitude with the target while keeping the estimated phase). Each iteration reduces ‖|STFT(x̂)| − |S_target|‖². Convergence is monotonic but reaches only a local minimum because the constraint sets are non-convex. Typically **30–100 iterations** yield acceptable quality, though the Fast Griffin-Lim variant (Perraudin et al., 2013) accelerates convergence via momentum. The resulting audio retains a characteristic metallic, phasey quality — recognizable but haunted by artifacts.

**Phase Gradient Heap Integration (PGHI)** exploits a direct mathematical relationship between phase and magnitude. For a Gaussian analysis window, the partial derivatives of phase can be computed entirely from the log-magnitude gradients: ∂φ/∂t = −(∂ log|STFT|/∂ω)/λ and ∂φ/∂ω = λ·(∂ log|STFT|/∂t). Phase is then reconstructed by integrating these gradients from high-energy bins outward using a priority queue (heap), ensuring the most reliable estimates propagate first. PGHI is **non-iterative** — a single pass — and produces quality comparable to many Griffin-Lim iterations.

**Single Pass Spectrogram Inversion (SPSI)** propagates phase frame-by-frame using the expected phase advance for a stationary sinusoid: φ[m+1,k] = φ[m,k] + 2πkH/N. Spectral peaks are identified and tracked; their phases are propagated while non-peak bins receive interpolated values. This is fast and deterministic but works best for tonal signals.

**Neural vocoders** (WaveNet, WaveGlow, HiFi-GAN) bypass explicit phase estimation entirely. They learn to generate waveforms directly from spectral representations by modeling the conditional probability p(x[n] | spectrogram features). HiFi-GAN's generator uses adversarial training against multi-period and multi-scale discriminators, achieving audio quality nearly indistinguishable from recorded sound. The limitation is data dependency — models trained on speech may struggle with novel hand-drawn spectrograms that deviate from natural audio statistics.

**Additive synthesis bypass** sidesteps the entire STFT framework. Sound is generated as y(t) = Σ Aₖ(t) · sin(∫ 2πfₖ(τ)dτ + φₖ(0)), where each sinusoidal partial has explicit, continuous phase control maintained by integrating instantaneous frequency. Phase coherence is guaranteed by construction. This is the basis of Spectral Modeling Synthesis.

---

## Twelve engines interpret the same canvas through different mathematics

The central design insight of a spectral painting application is that the **same drawing produces radically different music** depending on which synthesis algorithm interprets it. A diagonal bright line becomes a glissando in additive mode, a filter sweep in subtractive mode, a grain-density ramp in granular mode, and a sequence of pluck events in physical modeling mode. The visual gesture stays the same; only the mathematical interpretation of pixel values changes.

### Additive synthesis builds sound from individual sinusoids

The output is y(t) = Σ Aₖ(t) · sin(2πfₖt + φₖ), where each pixel row controls one sinusoidal oscillator's amplitude over time. The sound is **crystalline, precise, and microscopically detailed** — like glass organ tones or the resonances of a struck bell in a cathedral. With few partials it sounds pure and electronic; with thousands, it can approximate any timbre. Additive synthesis suits ambient, drone, spectral music (in the Grisey/Murail tradition), and academic electroacoustic composition. Its limitation is that pure sinusoids lack the noise and stochastic texture of real instruments — without a supplementary noise model, everything sounds artificially clean.

### Griffin-Lim reconstruction treats the canvas as a target spectrogram

The algorithm iteratively finds a signal whose STFT magnitude best matches the painted image. The result sounds **metallic, watery, and ghostly** — as if the music were playing through a glass of water. Notes shimmer and wobble from imperfect phase reconstruction. It suits experimental, glitch, and lo-fi aesthetics. Quality degrades for complex spectral content, and low frequencies reconstruct particularly poorly due to fewer overlap-add constraints per cycle.

### Random phase ISTFT creates PaulStretch-style ambient textures

Each frame's magnitude comes from the canvas brightness; phase is independently randomized. The result is **ethereal, weightless, and hauntingly beautiful** — infinite reverb tails where all transients dissolve into hovering spectral essence. It is fundamentally incapable of producing sharp attacks, clicks, or any rhythmic content. Everything becomes a pad. This is the core technique behind the extreme time-stretching that inspired the slo-mo theme in the film *Dredd* (2012).

### Subtractive synthesis paints the filter, not the sound

The canvas becomes a time-varying spectral filter H(f,t) applied to a harmonically rich source signal: Y(f,t) = X(f,t) · H(f,t). Bright regions pass energy; dark regions attenuate. This is the **quintessential analog synth sound** — the fat, growling basses and squelchy acid lines of Moog, Roland TB-303, and classic techno. Drawing a bright diagonal line sweeping upward creates a resonant filter opening. The mathematical operation is pointwise multiplication in the frequency domain (convolution in time). The fundamental limitation: a subtractive filter can only remove energy, never add frequencies that weren't present in the source.

### FM synthesis maps brightness to modulation index

The equation y(t) = A sin(2πf_c t + β sin(2πf_m t)) produces sidebands at f_c ± n·f_m with amplitudes governed by **Bessel functions** J_n(β). As modulation index β increases (brighter pixels), more sidebands appear, and the spectrum grows brighter and more complex. Integer carrier-to-modulator ratios produce harmonic spectra; non-integer ratios produce metallic, bell-like inharmonic tones. This is the sound of the **Yamaha DX7** — glassy electric pianos, crystalline bells, slap basses. The limitation is that the relationship between canvas brightness and audible spectrum is highly nonlinear and difficult to intuit: small changes in β cause dramatic timbral reorganization as individual Bessel function values cross zero.

### Wavetable synthesis converts canvas columns into morphing waveforms

Each vertical slice of the canvas is treated as a spectral snapshot, inverse-FFT'd into a single-cycle waveform. These waveforms are stacked into a wavetable and scanned during playback with interpolation between adjacent frames: y(t) = (1−α)·w_⌊p(t)⌋[φ(t)] + α·w_⌈p(t)⌉[φ(t)]. The sound is **evolving, shapeshifting, and versatile** — the modern workhorse of electronic sound design as heard in instruments like Serum and Vital. Each canvas column defines a timbre; moving left to right designs a timbral evolution. The limitation is that single-cycle waveforms are perfectly periodic — no noise, no natural variation within a cycle.

### Formant synthesis models the human vocal tract

Bright spots at specific formant frequencies create vowel-like sounds. The core model is source-filter: an excitation (pulse train or noise) passes through resonant bandpass filters H_k(z) = 1/(1 − 2R_k cos(θ_k)z^{−1} + R_k² z^{−2}), where θ_k = 2πF_k/f_s sets the formant center frequency and R_k controls bandwidth. Drawing bright spots at **800 Hz and 1200 Hz** produces an open "aah" (/a/); moving them to **300 Hz and 2300 Hz** shifts to "ee" (/i/). Sweeping formant positions across the canvas creates a ghostly choir morphing through vowels. Without careful coarticulation modeling, transitions sound robotic.

### Karplus-Strong turns bright pixels into plucked strings

Each bright pixel triggers a noise burst into a feedback delay line with a lowpass filter: y(n) = ½[y(n−P) + y(n−P−1)], where P = f_s/f₀ sets the pitch. The averaging filter attenuates higher harmonics more per round trip — exactly mimicking frequency-dependent string losses. The result is **strikingly natural plucked-string sound** from absurdly simple mathematics. Dense clusters of bright pixels create cascading harp glissandi. The limitation is that only decaying, resonant tones are possible — no sustained notes, no bowed textures.

### Granular synthesis spawns clouds of micro-sound from painted points

Each bright pixel spawns overlapping grains: g_i(t) = w(t−t_i) · sin(2πf_i(t−t_i) + φ_i), typically 1–100 ms long with Gaussian or Hann envelopes. The output y(t) = Σ A_i · g_i(t) is a pointillist texture. Dense bright regions produce **thick, shimmering clouds**; sparse regions create rain-on-tin stochastic textures. The time-frequency uncertainty principle is inescapable: short grains (5 ms) have imprecise pitch (Δf ≈ 200 Hz, noise-like); long grains (50 ms) have clear pitch but poor time localization. Granular synthesis suits ambient, electroacoustic, and Xenakis-style stochastic music.

### The phase vocoder is the most literal spectrogram instrument

The canvas is treated as a magnitude spectrogram and resynthesized with phase propagated using instantaneous frequency: ω_inst(m,k) = ω_k + Δφ_unwrapped(m,k)/H. Time-stretching changes only the synthesis hop size while preserving the spectral content. The quality is crystalline for moderate transformations but develops characteristic **"phasiness"** at extreme settings — a metallic, underwater quality from phase coherence breakdown, with transients smeared into ghostly pre-echoes.

### SMS decomposes drawn content into tones, noise, and attacks

Spectral Modeling Synthesis (Xavier Serra, Stanford, 1989) is the most sophisticated approach. Clear horizontal lines on the canvas become **sinusoidal partials** with tracked frequency, amplitude, and phase trajectories. Diffuse, foggy regions become the **stochastic noise component** — white noise filtered through the drawn spectral envelope. Sharp vertical strokes become **transients**. The reconstructed sound can be startlingly realistic: a drawn clarinet that breathes, a drawn piano with a percussive hammer attack. The three-layer decomposition is detailed fully in the next section.

### Neural vocoders learn natural phase from data

A trained neural network accepts the painted canvas as a mel-spectrogram and generates a waveform directly, implicitly hallucinating plausible phase relationships from patterns learned during training. HiFi-GAN generates at **1186× real-time on a GPU**. Quality is highest among all methods for inputs resembling the training distribution, but novel hand-drawn spectrograms may produce artifacts or gibberish. The network is a black box — no interpretable parameters for artistic fine-tuning.

---

## Why raw spectral painting sounds like noise, and how SMS fixes it

Paint arbitrary brightness values across a spectrogram canvas, assign random phase, and invert. The result is invariably some variety of noise. Three deficiencies explain why, and Spectral Modeling Synthesis addresses all three.

**Phase incoherence** is the first problem. A real sound at frequency f₀ produces phase that advances predictably between frames: φ[m+1,k] = φ[m,k] + 2πf_k H/f_s. Random phase destroys this coherence, causing destructive interference between overlapping frames. The ear perceives granular buzz with no repeating structure. **Absence of harmonic structure** is the second problem. A pitched sound concentrates energy at integer multiples of a fundamental: f_n = n·f₀. Arbitrary drawings distribute energy across all frequencies including the gaps between harmonics, producing colored noise rather than pitch. **Missing temporal envelope** is the third: natural sounds have attack, decay, sustain, and release. Static spectral frames that switch on and off abruptly lack any gestural identity.

### The harmonic series is the lattice of pitched sound

A periodic waveform with period T has a Fourier series x(t) = Σ A_n cos(2πnf₀t + φ_n) where f₀ = 1/T. Because every partial frequency is an integer multiple of f₀, the waveform repeats exactly every T seconds, and the ear fuses the components into a single perceived pitch. The relative amplitudes {A_n} determine timbre. Real strings deviate from perfect harmonicity due to stiffness: f_n = n·f₁·√(1 + B·n²), where B is the inharmonicity coefficient (approximately **0.0002 for piano bass strings** to **0.04 for treble strings**). This inharmonicity produces the characteristic metallic shimmer of piano tone and explains why piano octaves are "stretched" — tuned slightly wider than 2:1 to align the displaced partials.

### Three layers reconstruct natural sound from painted marks

Xavier Serra and Julius O. Smith III developed SMS at Stanford's CCRMA, published as a landmark 1990 paper in the *Computer Music Journal*. The model decomposes any sound into:

The **deterministic component** captures everything pitched as a sum of time-varying sinusoids: x_det(t) = Σ A_r(t)·cos(θ_r(t)), where θ_r(t) = ∫ 2πf_r(τ)dτ. Analysis detects spectral peaks via local maxima in magnitude spectra, refines their locations through quadratic interpolation, and tracks peaks across frames by minimizing frequency-distance costs. On the canvas, these appear as clear, narrow horizontal strokes — the skeleton of the sound.

The **stochastic component** is computed by spectrally subtracting the deterministic resynthesis from the original: R[m,k] = X_orig[m,k] − X_det[m,k]. This residual is modeled as white noise filtered through a time-varying spectral envelope, approximated by piecewise-linear segments. On the canvas, it appears as diffuse, foggy regions. This layer captures bow friction, breath noise, room ambience — everything that makes a sound feel physically present rather than artificially sterile.

The **transient component** (added in later extensions) captures short bursts — hammer strikes, consonant plosives, pluck attacks — that are neither periodic nor noise-like. On the canvas, these appear as sharp vertical lines. Transient phase must be preserved exactly, because transients require temporal coherence for their "snap" quality.

### Noise is not a flaw — it is the acoustic signature of physical processes

A purely deterministic resynthesis sounds dead because it is maximally predictable. **Stochastic variation prevents the auditory system from fully predicting the next moment**, and this unpredictability is perceived as "aliveness." Bow friction on a violin string produces broadband turbulence from rosin stick-slip mechanics, centered in the 1–8 kHz range. Breath noise in wind instruments adds aspiration colored by the bore resonances. Analog circuit warmth comes from thermal noise (flat spectrum), shot noise, and 1/f (pink) noise that introduce subtle random amplitude and frequency fluctuations.

Different noise colors serve different sonic roles. **White noise** (flat power spectral density) sounds like rain or TV static. **Pink noise** (S(f) ∝ 1/f) sounds like rushing wind and, critically, distributes equal energy per octave, matching human frequency perception. **Brown noise** (S(f) ∝ 1/f²) sounds like distant thunder or ocean surf. Filtered noise bands create specific textures: highpass pink noise at 2–8 kHz replicates vocal breathiness; narrow mid-band noise at 2–5 kHz replicates bow scrape. In SMS, the stochastic component's time-varying spectral envelope shapes white noise into whatever color is needed frame by frame.

---

## Spectral editing operates on magnitude and phase as separate domains

When existing audio is imported onto the canvas, the STFT analysis produces a complex-valued time-frequency matrix X[m,k] = |X[m,k]| · e^{jφ[m,k]}. The magnitude is displayed visually; the phase is stored invisibly. The artist edits the magnitude spectrogram — erasing regions, painting new content, adjusting brightness — then the modified magnitude is recombined with appropriate phase and inverted back to audio via ISTFT with overlap-add.

**Phase preservation** is appropriate for gentle edits: EQ adjustments, partial component removal, subtle spectral shaping. The original phase maintains temporal coherence and transient quality. **Phase reconstruction** becomes necessary when substantial new spectral content is painted that has no corresponding original phase data. A hybrid approach preserves phase near original content and reconstructs it for newly drawn regions, blending at boundaries.

The three fundamental spectral operations each have direct physical meaning. **Multiplication** Y[m,k] = X[m,k] · G[m,k] is filtering — an infinitely precise EQ that can vary with time. **Addition** Y[m,k] = X₁[m,k] + X₂[m,k] is layering, where phase relationships between sources determine constructive or destructive interference. **Subtraction** |Y[m,k]| = max(|X_mix[m,k]| − |X_noise[m,k]|, 0) is removal, the basis of classic noise reduction, though it introduces "musical noise" artifacts from isolated surviving spectral peaks.

### Cross-synthesis imposes one sound's shape onto another's harmonics

The classic vocoder effect extracts the spectral envelope of a modulator (typically voice) and applies it to the flattened spectrum of a carrier (typically a synthesizer): Y[m,k] = (C[m,k] / E_C[m,k]) · E_M[m,k]. The carrier provides the harmonic fine structure (pitch, partials); the modulator provides the broad spectral shape (formants, vowel identity). The result is a "talking synthesizer" — the carrier's pitch articulated by the modulator's speech gestures. Spectral envelope extraction uses cepstral smoothing, LPC analysis, or true-envelope estimation.

### Optimal transport morphing moves spectral energy rather than crossfading it

Linear interpolation between two spectra S(α) = (1−α)S₁ + αS₂ at α = 0.5 produces a ghostly superposition — both sounds playing simultaneously at half volume. This is crossfading, not morphing. **Optimal transport displacement interpolation** (Henderson and Solomon, DAFx 2019) solves this by computing the minimum-cost plan to move spectral mass from one distribution to another. Spectral peaks physically slide from source frequencies to target frequencies: f_α = (1−α)f_A + α·f_B. At the midpoint, a partial at 440 Hz in source A and 660 Hz in source B becomes a single partial at **~550 Hz** rather than two simultaneous tones. The Wasserstein distance W₂ serves as the metric, and for 1D spectral distributions, the optimal transport map has a closed-form solution via quantile functions. Recent work (Renaud et al., 2025) extends this to global spectrogram interpolation using Wasserstein barycenters.

### Time-frequency masks separate sources by controlling transparency

Masking applies a 2D gain function to the complex STFT: Ŝ_target[m,k] = M[m,k] · X_mix[m,k]. **Binary masks** (M ∈ {0,1}) assign each time-frequency bin entirely to target or interference — clean separation but with "musical noise" artifacts from bins switching on and off. **Wiener soft masks** (M = |S_target|²/Σ|S_i|²) distribute shared energy proportionally, producing smoother separation with more leakage. **Ratio masks** (M = |S_target|/|X_mix|) can exceed unity, allowing boosting. The exponent p controls hardness: p = 1 gives amplitude ratios, p = 2 gives Wiener power ratios, p → ∞ approaches the binary mask.

---

## Spectral effects replace time-domain pedals with frequency-domain surgery

Traditional audio effects (reverb, delay, chorus) operate on the waveform. Spectral effects operate on the frequency representation, enabling **per-frequency control that is impossible in the time domain**. Each effect is a mathematical operation on the STFT matrix.

**Spectral filtering** is multiplication: Y[m,k] = X[m,k] · G[m,k], where G is a gain surface drawn by the user. A time-invariant G(k) is a standard EQ; a time-varying G(m,k) is a dynamic spectral filter with potentially thousands of independently controllable bands. Drawing a deep notch at 1 kHz creates a hollow-tube resonance. Drawing a gentle high shelf adds "air" and "sparkle."

**Per-frequency delay** shifts each frequency bin independently in time: Y[m,k] = X[m−d(k),k]. If d(k) increases linearly with frequency, a percussive hit dissolves into a downward cascade — a "spectral waterfall" where sound unfolds across the spectrum like a prism splitting light, but for time. An inverse curve creates an upward spectral rise.

**Spectral freeze** captures one STFT frame's magnitude and holds it indefinitely with randomized phase: Y[m,k] = |X[m₀,k]| · e^{jθ_rand[m,k]}. Phase randomization is essential — holding original phase would produce a clicking periodic loop. The result is **infinite, shimmering sustain** of a single moment, an "audio photograph" that neither decays nor evolves.

**Spectral blur** applies 2D Gaussian convolution to the magnitude spectrogram: |Y[m,k]| = (|X| * G_σ)[m,k]. Temporal blurring (large σ_t) smears transients into soft swells — a snare hit becomes a gentle fade. Frequency blurring (large σ_f) merges neighboring harmonics into broad formant bands — a clear piano tone becomes an organ-like pad. Both together transform any sound into a warm ambient cloud, the sonic equivalent of photographing through frosted glass.

**Bin shuffling** randomly permutes frequency bins: Y[m,k] = X[m,π(k)]. This is pure spectral vandalism — all harmonic relationships are destroyed. A sung note becomes a dense inharmonic metallic cluster. Partial shuffling (permuting within local neighborhoods) produces degrees of "spectral drunkenness."

**Frequency warping** remaps bins via a drawn curve: Y[m,k] = X[m,w(k)]. Linear scaling w(k) = k·r shifts pitch uniformly. Power-law warping w(k) = k^β with β > 1 stretches higher harmonics progressively apart, creating bell-like inharmonicity. An S-curve that compresses midrange and expands extremes produces an eerie, scooped timbre.

---

## One canvas unifies all twelve engines under a single interaction model

The deepest conceptual contribution is not any individual synthesis method but the realization that a single 2D canvas — time, frequency, brightness — serves as universal input for all of them. As the KVR Audio review of MetaSynth observed: "There is no qualitative difference between a piano roll and a spectrograph; both are ways of mapping vertical space into pitch and horizontal space to time."

The canvas operates in three modes simultaneously. As a **blank canvas for synthesis**, each engine interprets drawn marks through its own mathematics: pixel brightness becomes sinusoidal amplitude (additive), filter gain (subtractive), modulation index (FM), grain density (granular), or pluck intensity (Karplus-Strong). As an **imported spectrogram for editing**, the canvas displays existing audio that can be modified with the same drawing tools — erasing regions removes frequency content, brightening adds energy. As a **filter/effect control surface**, the drawing acts as a time-frequency mask or parameter mapping applied to audio passing through the engine.

What changes between modes is the mathematical interpretation of pixel values. What stays the same is everything the artist interacts with: the drawing tools (brush, eraser, line, gradient, fill), the visual feedback (always a 2D image with time and frequency axes), the fundamental metaphor (bright = energy, dark = silence), and the musical constraint systems (scale snapping, harmonic auto-fill). This means a user can draw a shape once and hear it as additive, subtractive, granular, FM, and physical modeling in rapid succession — a **timbral palette from a single visual gesture**.

Multi-layer composition extends this further. Layer 1 might use additive synthesis for tonal harmonic content, Layer 2 might use granular synthesis for textural atmosphere, Layer 3 might use Karplus-Strong for plucked percussive events, and Layer 4 might use FM for metallic accents. The final output sums all layers — a complete multitrack composition created entirely through drawing.

This paradigm connects to a rich conceptual lineage: Xenakis's vision of music as architecture, Cornelius Cardew's *Treatise* (193 pages of abstract visual symbols interpreted as sound), Morton Feldman's graph pieces composed on graph paper, and the broader insight that visual art principles — composition, contrast, rhythm, texture — translate directly to musical principles when the canvas is a spectrogram.

---

## Musical constraints transform free drawing into coherent music

Without constraints, spectral painting produces arbitrary spectral content. Musical helpers impose the structures that make sound feel like music.

### Pitch quantization maps the frequency axis onto tuning systems

**Equal temperament** divides the octave into 12 equal semitones: f_n = f₀ · 2^{n/12}, where the semitone ratio is 2^{1/12} ≈ 1.05946. No interval except the octave is a perfect integer ratio — the fifth at 700 cents is 1.955 cents flat of the just 3:2 (701.955 cents), and the major third at 400 cents is **13.69 cents sharp** of the pure 5:4 (386.31 cents). This impurity is the cost of free modulation: every key sounds identical.

**Just intonation** uses small integer ratios — perfect fifth = 3:2, major third = 5:4, minor third = 6:5 — where harmonics align precisely and beating vanishes. The tradeoff is severe: the syntonic comma (81/80 ≈ 21.51 cents) means that stacking four just fifths does not equal a just major third plus two octaves, creating "wolf" intervals that make modulation impossible without retuning. For spectral painting, just intonation maximizes harmonic fusion — partials lock into the overtone series, creating a unified timbre rather than a cluster of separate tones.

**Pythagorean tuning** generates all intervals from stacked perfect fifths, but twelve fifths overshoot seven octaves by the **Pythagorean comma**: (3/2)^12/2^7 = 531441/524288 ≈ 23.46 cents. **Microtonal systems** extend beyond 12-TET: 24-TET provides quarter tones for Arabic maqam traditions; **53-TET** (known since Jing Fang, 78–37 BCE) approximates 5-limit just intonation with extraordinary precision — its fifth deviates by only 0.07 cents from pure. The Bohlen-Pierce scale replaces the octave with the tritave (3:1) divided into 13 steps, optimized for odd-harmonic timbres.

### Harmonic auto-fill generates overtone series from a single drawn fundamental

When a point is drawn at frequency f₁, the system can automatically place partials at 2f₁, 3f₁, 4f₁, and so on with amplitude rolloff following a chosen timbre envelope. A **sawtooth envelope** (all harmonics, A_n ∝ 1/n) produces bright, buzzy, brass-like tone. A **square-wave envelope** (odd harmonics only, A_n ∝ 1/n) produces hollow, clarinet-like quality. A **triangle envelope** (odd harmonics, A_n ∝ 1/n²) produces soft, flute-like mellowness. Inharmonic partial spacing creates metallic, bell-like timbres where pitch perception breaks down. The spectral rolloff exponent α in A_n ∝ 1/n^α functions as a "brightness knob" for the harmonic auto-fill brush.

### ADSR envelopes make painted tones sound performed rather than switched

A sudden amplitude onset produces a mathematical step function containing energy at all frequencies — a broadband click with bandwidth ≈ 1/(π·t_attack). Even a 1 ms attack produces ~318 Hz of spectral splatter. ADSR (Attack-Decay-Sustain-Release) envelopes shape painted spectral content into expressive notes. Short attack, short decay, low sustain, short release creates plucked/struck character. Long attack, no decay, high sustain, long release creates bowed/blown character. Critically, each partial in a spectral painting can have its **own ADSR envelope**, modeling the physical reality that higher harmonics decay faster in acoustic instruments.

### Keyframe interpolation determines how spectra evolve between painted states

Linear interpolation S(α) = (1−α)S₁ + αS₂ is simple but produces a ~3 dB loudness dip at the midpoint because amplitudes add but powers don't. **Equal-power crossfade** uses trigonometric gain curves: g₁ = cos(πα/2), g₂ = sin(πα/2), maintaining constant total power because cos² + sin² = 1. **Cubic spline interpolation** (Catmull-Rom) provides C² continuity for smooth parameter trajectories. And **optimal transport displacement interpolation** moves spectral energy along the frequency axis rather than blending it, producing true timbral morphs via Wasserstein barycenters: F_α^{−1} = (1−α)F₀^{−1} + αF₁^{−1}.

---

## Generative systems populate the canvas with emergent complexity

Beyond hand-drawing, mathematical systems can generate spectral content with organic, self-organizing properties impossible to create by hand.

**Reaction-diffusion systems** (the Gray-Scott model: ∂u/∂t = D_u∇²u − uv² + F(1−u), ∂v/∂t = D_v∇²v + uv² − (F+k)v) produce visual patterns — spots, stripes, spirals, labyrinthine structures — depending on the feed rate F and kill rate k. When the 2D simulation output is mapped as a spectrogram, **spots become isolated tonal pinging events**, horizontal stripes become drones with organic wobble, spirals become pitch sweeps that curl through frequency space, and labyrinthine patterns become dense, interwoven tonal masses. A 1996 paper in *Organised Sound* documented using reaction-diffusion patterns for algorithmic composition, creating naturalistic soundscapes. The patterns are non-repeating and self-organizing — "sonic organisms" that feel biological and alive.

**Cellular automata** produce spacetime diagrams that map naturally onto spectrograms (rows = time steps, columns = cells = frequency bands). Wolfram's Rule 90 generates the Sierpinski triangle fractal — as a spectrogram, this creates self-similar patterns at multiple timescales. Rule 110, proven Turing-complete, sits at the "edge of chaos" where structures are complex enough to be engaging yet ordered enough to have recognizable patterns. Stephen Wolfram's WolframTones system demonstrated that cellular automata music could pass a musical Turing test — listeners assumed it had human origins.

**L-systems** (Lindenmayer systems) are formal grammars that rewrite strings according to production rules. The classic example (axiom A, rules A→AB, B→A) produces strings whose length ratio converges to the **golden ratio φ**. When characters map to musical parameters (pitch steps, durations, dynamics), the recursive structure creates **self-similar patterns at multiple timescales** — melodies that contain smaller versions of themselves. This connects to the landmark finding by Voss and Clarke (1975/1978) that pitch and amplitude fluctuations in music follow a **1/f power law**, meaning music with fractal statistics (balanced between predictable and surprising) is consistently rated most musically pleasing.

**Xenakis's GENDYN** (GENération DYNamique) represents the most radical generative approach: algorithmic composition at the sample level. A waveform is defined by n breakpoints (typically 12) with amplitude and duration coordinates. At each cycle, breakpoints are perturbed by random walks controlled by probability distributions — Cauchy (delicate pitch movement), logistic (rough, nervous buzzing), Gaussian (moderate perturbation). Elastic barriers prevent divergence. The result sounds **raw, primitive, and alien** — waveforms that evolve continuously and unpredictably, unlike any traditional synthesis. GENDYN compositions include *Gendy3* (1991) and *S.709* (1994), Xenakis's late masterworks.

**Vector synthesis** positions a point within a 2D space where each corner holds a different spectral snapshot. Bilinear interpolation S(x,y) = (1−x)(1−y)S₁ + x(1−y)S₂ + (1−x)yS₃ + xyS₄ produces smooth timbral transitions as the point moves. Drawing a path through this space over time creates timbral trajectories — exactly the principle behind the Sequential Prophet VS (1986) and Korg Wavestation (1990).

---

## Psychoacoustics determines what the listener actually hears

Everything painted on a spectrogram passes through the listener's auditory system — a biological processor with its own filters, biases, and illusions. Understanding these determines whether a spectral painting sounds like music or noise.

### Critical bands set the resolution of human hearing

The basilar membrane acts as a bank of overlapping bandpass filters. The **critical band** is the frequency range within which two tones interact rather than being heard as separate pitches. The Bark scale divides hearing into **24 critical bands** from 20 Hz to 15,500 Hz. Below 500 Hz, critical bandwidth is roughly **100 Hz** (constant); above 500 Hz it grows to approximately 15–20% of center frequency. The ERB (Equivalent Rectangular Bandwidth) scale provides finer resolution: ERB(f) = 24.7 × (4.37f/1000 + 1) Hz. Two tones painted within one critical band will not be heard as distinct pitches — they fuse into beating (< 15 Hz separation), roughness (15–75 Hz), or a single merged percept.

### Auditory masking makes some drawn content inaudible

A louder sound can render a nearby quieter sound completely inaudible. Simultaneous masking spreads asymmetrically: a low-frequency masker effectively masks higher frequencies, with the high-frequency slope flattening dramatically at high intensities (shallowing to ~25 dB/octave at 80–100 dB). Forward masking persists for **100–200 ms** after a loud sound ends; backward masking can suppress sounds up to **10–20 ms before** a loud onset. The implication for spectral painting is sobering: not everything drawn will be heard. Quiet features near loud ones will be perceptually invisible. This is the principle behind MP3 compression — masked content is discarded because the listener cannot perceive it anyway.

### The missing fundamental means you don't need to draw the bass

When harmonics 2, 3, 4, 5 of a fundamental are present without the fundamental itself, the brain still perceives pitch at the missing fundamental frequency. This is not cochlear distortion — it persists even when masking noise eliminates any possible distortion product. The mechanism is **periodicity detection**: auditory nerve fibers phase-lock to individual harmonics, and the combined firing pattern has a periodicity of 1/f₀. This is why telephone bandwidth (300–3400 Hz) removes male voice fundamentals (~100 Hz) without affecting perceived pitch. For spectral painting, this means **drawing upper harmonics alone can establish low pitches** without requiring actual low-frequency energy, which demands enormous resolution and canvas space.

### The ear is a logarithmic analyzer, and the canvas should match

The basilar membrane maps frequency to position via an approximately exponential function (Greenwood, 1961): f = 165.4 × (10^{2.1x} − 0.88), where x is fractional distance from the apex. Each octave occupies roughly equal distance (~3.5–4 mm) along the **35 mm membrane**. This means equal frequency ratios map to equal distances — the defining characteristic of logarithmic perception. A linear frequency display wastes space catastrophically: the three octaves from A1 (55 Hz) to A4 (440 Hz) occupy only ~2% of a 0–20,000 Hz linear display, while the single octave from 10,000–20,000 Hz occupies 50%. A logarithmic or mel-scaled axis allocates equal space to equal perceptual intervals.

### Roughness and consonance arise from critical band interactions

The Plomp-Levelt curve (1965) established that maximum dissonance occurs at approximately **25% of the critical bandwidth** — the most unpleasant interval at any frequency. Full consonance returns beyond one critical bandwidth separation. For complex harmonic tones, total dissonance is the sum of roughness contributions from all partial pairs. This elegantly explains classical consonance rankings: unison (all partials coincide), octave (every partial of the upper tone coincides with an even partial of the lower), and perfect fifth (many partial coincidences) produce the least roughness. The Sethares parameterization extends this to predict that **different timbres prefer different intervals** — a profound insight for spectral painting, where timbre and harmony are drawn simultaneously.

### Equal-loudness contours mean brightness does not equal perceived loudness

The Fletcher-Munson curves (ISO 226:2003) show that the ear is most sensitive at **2–5 kHz** (due to ear canal resonance at ~3.5 kHz). At moderate listening levels (40 phons), 100 Hz requires **24 dB more energy** than 1 kHz to sound equally loud; 50 Hz requires 40+ dB more. Equal-brightness spectral painting will sound disproportionately prominent in the 2–5 kHz presence region and deficient in bass and extreme treble. The contours also flatten at high loudness levels — spectral paintings sound more spectrally balanced when played loud.

### Auditory scene analysis determines how many "objects" the listener hears

The brain parses incoming sound into perceptual objects following Gestalt principles. **Harmonicity** is the dominant grouping cue: partials in integer frequency ratios fuse into a single perceived timbre. A harmonic mistuned by more than ~3% begins to "pop out" as a separate object. **Common fate** is equally powerful: spectral components that change together (coherent vibrato, synchronized amplitude modulation) fuse into one source. **Common onset** promotes grouping; asynchronous onsets of even 30–50 ms promote segregation. For spectral painting, creating a single perceived voice requires drawing harmonics at exact integer ratios, with coherent amplitude envelopes, synchronized frequency modulation, and simultaneous onset. Violating any of these principles causes the brain to parse the drawing into multiple separate auditory objects.

---

## Conclusion: where mathematics meets musical intuition

This theoretical foundation reveals that a spectral audio painting application is not merely a drawing tool connected to a speaker — it is a system that translates between two complete sensory domains through precise mathematical transformations, constrained by the physics of acoustics and the biology of hearing.

The most consequential insight is that **the same visual gesture acquires entirely different musical meaning depending on the mathematical function that interprets it**. A bright diagonal line is a glissando, a filter sweep, a grain-density ramp, or a cascade of pluck events. This is not a limitation but a creative multiplier — twelve synthesis engines sharing one canvas means twelve sonic interpretations of every brushstroke.

The three-layer SMS decomposition resolves the fundamental problem of why naïve spectral painting sounds artificial. Separating deterministic partials from stochastic noise from transient attacks mirrors how the auditory system itself groups spectral energy into perceived objects, and it provides three distinct "drawing modes" — lines for pitch, fog for texture, spikes for attacks — that map intuitively to physical sound-production mechanisms.

Psychoacoustics imposes hard constraints that no amount of synthesis sophistication can override. Critical bands determine minimum resolvable frequency spacing. Masking makes some drawn content inaudible regardless of its visual prominence. The missing fundamental means implied pitch can substitute for actual low-frequency energy. Equal-loudness contours mean perceptually balanced spectral content requires physically unequal amplitudes across frequency. And auditory scene analysis means the listener will group or segregate drawn content based on harmonic ratios, common fate, and onset synchrony — not based on visual proximity on the canvas.

The generative methods — reaction-diffusion, cellular automata, L-systems, GENDYN — offer a path beyond what any human hand would draw, populating the canvas with emergent mathematical structures that exhibit the 1/f spectral statistics associated with the most musically pleasing compositions. And optimal transport displacement interpolation solves the morphing problem that linear crossfading cannot: spectral energy slides continuously between states rather than fading through ghostly superpositions.

Together, these theoretical foundations form a complete conceptual architecture. What remains is building it.