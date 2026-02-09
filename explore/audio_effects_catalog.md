# Audio Effects Algorithm Catalog
## DSP Math Reference for Parallel Code Generation

Each entry is a **standalone algorithm**. Each is independently implementable as a Python function using NumPy + Numba.

**Function signature convention:** `def effect_XXX(samples: np.ndarray, sr: int, **params) -> np.ndarray`

**Post-chain (applied to ALL outputs, implement once):**
- DC offset removal (subtract running mean)
- Gentle lowpass at 16kHz (2nd order Butterworth)
- Soft-knee limiter: `y = x * (threshold / max(threshold, |x|))` smoothed
- Peak normalize to -1dBFS
- Fade in/out (256 samples linear) to avoid clicks

---

# A. DELAY-BASED EFFECTS

## A001 — Simple Feedback Delay
```
y[n] = x[n] + feedback * buffer[(write_pos - delay_samples) % buf_len]
buffer[write_pos] = y[n]
```
- **Params:** delay_ms ∈ [50, 1000], feedback ∈ [0.0, 0.9]
- **Perceptual:** Discrete echoes, rhythmic repeats
- **Numba:** Yes, single sample loop with ring buffer

## A002 — Multi-Tap Delay
```
y[n] = x[n] + Σ_k gain[k] * buffer[(write - delay[k]) % len]
```
- **Params:** num_taps ∈ [2, 8], delays spaced by golden ratio × base_ms, gains decaying geometrically
- **Perceptual:** Complex rhythmic echoes, diffuse repeats
- **Numba:** Yes

## A003 — Ping-Pong Stereo Delay
```
L_buf[n] = x[n] + fb * R_buf[n - delay]
R_buf[n] = fb * L_buf[n - delay]
```
- **Params:** delay_ms ∈ [100, 800], feedback ∈ [0.2, 0.85]
- **Perceptual:** Echoes bouncing left to right
- **Numba:** Yes, two ring buffers
- **Note:** Output is stereo (2-channel)

## A004 — Reverse Delay
```
For each chunk of delay_ms length:
  buf = last delay_ms of input
  y += buf[::-1] * feedback
Feed reversed chunk back into delay line
```
- **Params:** delay_ms ∈ [100, 500], feedback ∈ [0.3, 0.8]
- **Perceptual:** Swelling, backwards-sounding echoes
- **Numba:** Yes, chunk-based

## A005 — Tape Delay Emulation
```
delay_mod[n] = delay_samples + depth * sin(2π * rate * n / sr)
y[n] = x[n] + feedback * lowpass(buffer[n - delay_mod[n]])
```
- **Params:** delay_ms ∈ [100, 600], feedback ∈ [0.3, 0.8], wow_rate_hz ∈ [0.3, 3.0], wow_depth_samples ∈ [1, 8], filter_cutoff ∈ [2000, 5000]
- **Perceptual:** Warm, wobbly, lo-fi echoes
- **Numba:** Yes, interpolated read from buffer

## A006 — Granular Delay
```
For each grain event (density-based scheduling):
  read_pos = delay_line_pos + random_offset(±scatter)
  grain = window(delay_buffer[read_pos : read_pos + grain_size])
  overlap-add grain into output
```
- **Params:** delay_ms ∈ [100, 1000], grain_size_ms ∈ [10, 100], scatter_ms ∈ [0, 200], density ∈ [5, 50 grains/sec], feedback ∈ [0.0, 0.7]
- **Perceptual:** Diffuse, cloudy echoes that smear in time
- **Numba:** Grain scheduling loop

## A007 — Allpass Delay Diffuser
```
Chain of N allpass filters:
  y[n] = -g*x[n] + x[n-d] + g*y[n-d]
with different delay lengths d per stage
```
- **Params:** num_stages ∈ [4, 12], delay_range_ms ∈ [1, 50], g (allpass coeff) ∈ [0.5, 0.7]
- **Perceptual:** Smeared transients, diffused sound without obvious echoes, reverb-like
- **Numba:** Yes, chained allpass loops

## A008 — Fibonacci Delay Network
```
Delay taps at Fibonacci numbers × base_ms: 1, 1, 2, 3, 5, 8, 13, 21, 34, 55...
y[n] = x[n] + Σ fib_gain[k] * buffer[n - fib[k]*base_samples]
```
- **Params:** base_ms ∈ [5, 50], num_fibs ∈ [5, 12], decay ∈ [0.6, 0.95]
- **Perceptual:** Non-uniform echo pattern with natural-feeling density buildup
- **Numba:** Yes

## A009 — Prime Number Delay Structure
```
Delay taps at prime numbers × base_ms: 2, 3, 5, 7, 11, 13, 17, 19, 23...
Primes ensure no tap is a multiple of another → no resonant modes
```
- **Params:** base_ms ∈ [1, 20], num_primes ∈ [5, 15], feedback ∈ [0.3, 0.8]
- **Perceptual:** Dense, non-metallic diffusion — primes avoid comb-filter buildup
- **Numba:** Yes

## A010 — Stutter / Retrigger
```
Capture window_ms of audio at trigger point
Repeat it N times with optional per-repeat pitch shift and decay
```
- **Params:** window_ms ∈ [20, 200], repeats ∈ [2, 32], decay ∈ [0.8, 1.0], pitch_drift ∈ [-0.1, 0.1] per repeat
- **Perceptual:** Glitchy, rhythmic repetition, DJ-style stutter
- **Numba:** Yes

## A011 — Buffer Shuffle
```
Divide signal into N chunks of chunk_ms
Apply random permutation to chunk order
Crossfade at boundaries (5ms)
```
- **Params:** chunk_ms ∈ [50, 500], seed
- **Perceptual:** Disorienting reordering, cut-up technique
- **NumPy:** Chunk-level operations

## A012 — Reverse Chunks
```
Divide into chunks, reverse every other chunk (or random subset)
Crossfade at boundaries
```
- **Params:** chunk_ms ∈ [50, 300], reverse_probability ∈ [0.3, 1.0]
- **Perceptual:** Partially backwards, disorienting but rhythmically connected

## A013 — Bouncing Ball Delay
```
Exponentially decreasing delay times simulating a bouncing object:
cumulative_delay += current_delay
output[n] += decay^bounce * input[n - cumulative_delay]
current_delay *= damping_coeff  (each bounce shorter)
```
- **Params:** initial_delay_ms ∈ [150, 1000], decay ∈ [0.5, 0.85], num_bounces ∈ [6, 25], damping ∈ [0.5, 0.75]
- **Perceptual:** Accelerating echoes like a ball bouncing to rest
- **Numba:** Yes

---

# B. REVERB ALGORITHMS

## B001 — Schroeder Reverb
```
4 parallel comb filters → 2 series allpass filters
Comb: y[n] = x[n-d] + g * y[n-d]
Allpass: y[n] = -g*x[n] + x[n-d] + g*y[n-d]
Comb delays: 29.7ms, 37.1ms, 41.1ms, 43.7ms (mutually prime sample counts)
Allpass delays: 5ms, 1.7ms
```
- **Params:** rt60 ∈ [0.5, 5.0] sec (determines g = 10^(-3*d/(sr*rt60))), wet_mix ∈ [0.2, 0.8]
- **Perceptual:** Classic metallic digital reverb
- **Numba:** Yes, all filters are sample-by-sample

## B002 — Moorer Reverb
```
6 comb filters with lowpass in feedback path → allpass chain
Comb feedback: y[n] = x[n-d] + g * lpf(y[n-d])
lpf: one-pole filter y = (1-damp)*x + damp*y_prev
```
- **Params:** rt60 ∈ [0.5, 8.0], damping ∈ [0.1, 0.9], wet_mix
- **Perceptual:** Warmer than Schroeder, high frequencies decay faster
- **Numba:** Yes

## B003 — Feedback Delay Network (FDN) Reverb
```
N delay lines with NxN feedback matrix (Hadamard or householder)
For N=4 Hadamard: [[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,-1]] / 2
Each delay line has lowpass in feedback path
Output = weighted sum of delay line outputs
```
- **Params:** N ∈ {4, 8, 16}, delay_lengths (mutually prime ms values), rt60 ∈ [0.5, 20.0], damping ∈ [0.0, 0.9]
- **Perceptual:** Dense, lush, smooth reverb tails
- **Numba:** Yes, matrix multiply per sample

## B004 — Plate Reverb Model
```
2D waveguide mesh approximation:
- Allpass chain simulating dispersion
- Multiple nested allpass + delay sections
- Modulated delay lengths for decorrelation
```
- **Params:** decay ∈ [0.5, 10.0], damping ∈ [0.1, 0.8], mod_rate ∈ [0.5, 2.0], mod_depth ∈ [1, 8] samples, pre_delay_ms ∈ [0, 100]
- **Perceptual:** Bright, dense, shimmery — the classic plate sound
- **Numba:** Yes

## B005 — Spring Reverb Model
```
Chirp dispersion: allpass chain with frequency-dependent delay
y[n] = Σ_k allpass_k(x[n]) with increasing delay per stage
Add nonlinearity in feedback for "boing" character
```
- **Params:** num_springs ∈ [1, 3], tension (controls dispersion), damping, chaos (feedback nonlinearity)
- **Perceptual:** Twangy, boinging, lo-fi character
- **Numba:** Yes

## B006 — Shimmer Reverb
```
Reverb with pitch-shifted signal fed back into reverb input
FDN reverb → pitch shift up by +12 semitones (or +7, +5) → feed back
pitch_shift via resampling or simple phase vocoder
```
- **Params:** rt60, pitch_shift_semitones ∈ [5, 12], shimmer_amount ∈ [0.1, 0.6]
- **Perceptual:** Ethereal, celestial, ever-ascending reverb tails
- **Numba:** Main loop yes, pitch shift may use FFT

## B007 — Convolution Reverb with Synthetic IR
```
Generate synthetic impulse response:
  ir[n] = noise * exp(-n * decay_rate) * random_early_reflection_spikes
Convolve: y = ifft(fft(x) * fft(ir))
```
- **Params:** ir_length_ms ∈ [200, 5000], decay_rate, num_early_reflections ∈ [3, 15], er_spacing_ms
- **Perceptual:** Realistic spatial reverb from synthetic space
- **NumPy:** FFT-based convolution

## B008 — Metallic Resonator
```
Bank of very short comb filters (1–5ms) in parallel
Each tuned to different pitch, high feedback (0.9–0.99)
Sum outputs with equal or weighted gain
```
- **Params:** num_resonators ∈ [4, 12], freq_hz_list (100–5000), feedback ∈ [0.9, 0.995], bandwidth
- **Perceptual:** Metallic, bell-like, ringing resonances added to input
- **Numba:** Yes

## B009 — Dattorro Plate Reverb
```
Distinct topology from B004: uses Dattorro's specific algorithm with
input bandwidth control → 4 input diffusion allpasses →
figure-eight tank (2 branches with allpass + delay + lowpass, cross-fed)
```
- **Params:** decay ∈ [0.3, 0.99], damping ∈ [0.2, 0.8], bandwidth ∈ [0.3, 0.9], pre_delay_ms ∈ [0, 100]
- **Perceptual:** Smooth, lush plate reverb — the gold standard topology
- **Numba:** Yes

## B010 — Freeverb (Jezar)
```
8 parallel Lowpass-Feedback-Comb filters → 4 series allpass filters
Comb delays: 1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617 samples (at 44.1k)
Allpass delays: 556, 441, 341, 225 samples
Damping via one-pole lowpass in each comb's feedback path
```
- **Params:** room_size ∈ [0.3, 0.99], damping ∈ [0.2, 0.9], wet_mix ∈ [0.3, 0.8]
- **Perceptual:** Classic digital reverb — bright, airy, instantly recognizable
- **Numba:** Yes

## B011 — Velvet Noise Reverb
```
Sparse random +1/-1 impulse sequence (velvet noise) as FIR impulse response
ir[n] = polarity * exp(-decay_rate * t) at pseudo-random positions
Convolve via FFT
Smoother than noise-based IRs due to temporal regularity of impulses
```
- **Params:** ir_length_ms ∈ [500, 4000], density ∈ [500, 4000] impulses/sec, decay_rate ∈ [1, 4], seed
- **Perceptual:** Smooth, artifact-free reverb tail without metallic coloring
- **NumPy:** FFT convolution

---

# C. MODULATION EFFECTS

## C001 — Chorus
```
y[n] = x[n] + wet * x[n - d(n)]
d(n) = base_delay + depth * sin(2π * rate * n / sr)
```
- **Params:** base_delay_ms ∈ [5, 30], depth_ms ∈ [1, 10], rate_hz ∈ [0.1, 5.0], voices ∈ [1, 4] (each with offset phase)
- **Perceptual:** Thickened, shimmering, detuned doubling
- **Numba:** Yes, interpolated delay read

## C002 — Flanger
```
Same as chorus but shorter delay and with feedback:
y[n] = x[n] + wet * buffer[n - d(n)]
buffer[n] = x[n] + feedback * buffer[n - d(n)]
d(n) = base + depth * sin(2π * rate * n / sr)
base_delay: 1–10ms (shorter than chorus)
```
- **Params:** base_delay_ms ∈ [0.5, 5], depth_ms ∈ [0.5, 5], rate_hz ∈ [0.05, 2.0], feedback ∈ [-0.95, 0.95]
- **Perceptual:** Jet-engine sweep, metallic comb filtering, negative feedback = hollow
- **Numba:** Yes

## C003 — Phaser
```
Chain of N allpass filters with swept cutoff:
allpass: y[n] = a*x[n] + x[n-1] - a*y[n-1]
where a = (1 - tan(π*f/sr)) / (1 + tan(π*f/sr))
f swept by LFO: f(n) = f_min * (f_max/f_min)^((1+sin(2π*rate*n/sr))/2)
Mix original + allpass output
```
- **Params:** num_stages ∈ {4, 6, 8, 12}, f_min ∈ [100, 400], f_max ∈ [1000, 8000], rate_hz ∈ [0.05, 2.0], feedback ∈ [0.0, 0.9], depth ∈ [0.5, 1.0]
- **Perceptual:** Sweeping notches, swooshing, spatial movement
- **Numba:** Yes

## C004 — Vibrato
```
Pure pitch modulation via modulated delay:
y[n] = x[n - d(n)]  (no dry mix)
d(n) = max_depth * sin(2π * rate * n / sr)
```
- **Params:** rate_hz ∈ [1, 8], depth_ms ∈ [1, 10]
- **Perceptual:** Pitch wobble, vocal-like expression
- **Numba:** Yes, interpolated read

## C005 — Tremolo
```
y[n] = x[n] * (1 - depth + depth * lfo(n))
lfo shapes: sin, triangle, square, sample-and-hold
```
- **Params:** rate_hz ∈ [1, 20], depth ∈ [0.3, 1.0], shape ∈ {sin, tri, square, s&h}
- **Perceptual:** Amplitude pulsing, rhythmic volume changes
- **Numba:** Yes

## C006 — Ring Modulation
```
y[n] = x[n] * carrier(n)
carrier: sin(2π * freq * n / sr)
```
- **Params:** carrier_freq_hz ∈ [20, 2000]
- **Perceptual:** Metallic, inharmonic, bell-like — creates sum and difference frequencies
- **Numba:** Yes

## C007 — Ring Mod with Noise Carrier
```
y[n] = x[n] * filtered_noise[n]
filtered_noise = bandpass(white_noise, center_freq, bandwidth)
```
- **Params:** center_freq ∈ [100, 5000], bandwidth_hz ∈ [50, 2000]
- **Perceptual:** Gritty, textural, unpredictable modulation
- **Numba:** Yes (generate noise, filter, multiply)

## C008 — Ring Mod with Chaos Carrier
```
y[n] = x[n] * chaos_osc[n]
chaos_osc: logistic map x_next = r * x * (1 - x), scaled to [-1, 1]
```
- **Params:** r ∈ [3.5, 4.0] (onset of chaos), chaos_speed (how often to step the map)
- **Perceptual:** Organic, evolving, unpredictable timbral shifts
- **Numba:** Yes

## C009 — Frequency Shifting (Hilbert)
```
Analytic signal: z[n] = x[n] + j*hilbert(x[n])
y[n] = Re(z[n] * exp(j*2π*shift_hz*n/sr))
Hilbert via FIR approximation or FFT
```
- **Params:** shift_hz ∈ [-500, 500]
- **Perceptual:** Inharmonic, metallic. Unlike pitch shift — partials shift by constant Hz not ratio
- **NumPy:** FFT for Hilbert, or Numba FIR

## C010 — Barber Pole Flanger (Infinite Rise)
```
Multiple flangers with staggered LFO phases (like Shepard tones)
Each flanger voice fades in/out as its delay sweeps through range
N voices with phase offsets 2πk/N
```
- **Params:** num_voices ∈ [3, 6], rate_hz ∈ [0.05, 0.5], depth_ms, feedback
- **Perceptual:** Continuously ascending or descending flange — never resolves
- **Numba:** Yes

## C011 — Stereo Auto-Pan
```
L_gain[n] = cos(π/4 + depth * lfo(n))
R_gain[n] = sin(π/4 + depth * lfo(n))
```
- **Params:** rate_hz ∈ [0.1, 10], depth ∈ [0.3, 1.0], lfo_shape
- **Perceptual:** Sound moves between left and right
- **Numba:** Yes

## C012 — Doppler Effect Simulation
```
Moving source: distance(t) varies sinusoidally or linearly
delay(t) = distance(t) / speed_of_sound
amplitude(t) = 1 / distance(t)
lowpass cutoff varies with approaching/receding
```
- **Params:** speed_mps ∈ [5, 100], closest_distance_m ∈ [1, 20], path (flyby, orbit, approach)
- **Perceptual:** Approaching/receding pitch shift + volume + filtering
- **Numba:** Yes

---

# D. DISTORTION & WAVESHAPING

## D001 — Hard Clipping
```
y[n] = clip(x[n], -threshold, threshold)
```
- **Params:** threshold ∈ [0.05, 0.9], pre_gain ∈ [1, 20]
- **Perceptual:** Harsh, buzzy, square-ish waveform
- **Numba:** Yes

## D002 — Soft Clipping (Tanh)
```
y[n] = tanh(drive * x[n])
```
- **Params:** drive ∈ [1, 20]
- **Perceptual:** Warm saturation, compressed dynamics
- **Numba:** Yes

## D003 — Tube Saturation Model
```
y[n] = (1 - exp(-drive * x[n])) for x >= 0
y[n] = -(1 - exp(drive * x[n])) for x < 0
Asymmetric version: different curves for positive/negative
```
- **Params:** drive ∈ [1, 10], asymmetry ∈ [0, 0.5]
- **Perceptual:** Even harmonics from asymmetry, warm breakup
- **Numba:** Yes

## D004 — Foldback Distortion
```
while |x[n]| > threshold:
  x[n] = 2*threshold - |x[n]|  (fold back)
Effectively: y = threshold - |threshold - |x| % (2*threshold) - threshold|
```
- **Params:** threshold ∈ [0.1, 0.8], pre_gain ∈ [2, 30]
- **Perceptual:** Complex harmonics, buzzy, synth-like, more interesting than clipping
- **Numba:** Yes

## D005 — Chebyshev Polynomial Waveshaper
```
T_0(x) = 1, T_1(x) = x
T_n(x) = 2x*T_{n-1}(x) - T_{n-2}(x)
y[n] = Σ a_k * T_k(x[n])
```
- **Params:** order ∈ [2, 8], coefficients a_k (controls which harmonics are added)
- **Perceptual:** Precise harmonic addition — T_k adds the k-th harmonic
- **Numba:** Yes

## D006 — Polynomial Waveshaper
```
y[n] = a1*x + a2*x² + a3*x³ + a4*x⁴ + a5*x⁵
```
- **Params:** coefficients a1–a5, each ∈ [-2, 2]
- **Perceptual:** Controllable harmonic content. Odd powers = odd harmonics, even = even
- **Numba:** Yes

## D007 — Sigmoid Family Waveshaping
```
atan:  y = (2/π) * atan(drive * x)
erf:   y = erf(drive * x / √2)
algebraic: y = x / sqrt(1 + drive * x²)
```
- **Params:** drive ∈ [1, 50], type ∈ {atan, erf, algebraic}
- **Perceptual:** Each has subtly different harmonic character
- **Numba:** Yes

## D008 — Bit Crusher
```
y[n] = round(x[n] * levels) / levels
levels = 2^bits
```
- **Params:** bits ∈ [1, 12]
- **Perceptual:** Quantization noise, gritty, retro digital
- **Numba:** Yes

## D009 — Sample Rate Reduction
```
hold_counter = sr / target_sr
y[n] = x[n - (n % hold_counter)]
```
- **Params:** target_sr ∈ [500, 16000]
- **Perceptual:** Aliasing, lo-fi, crunchy
- **Numba:** Yes

## D010 — Slew Rate Limiter
```
max_change = max_slew_per_sample
if y[n] - y[n-1] > max_change: y[n] = y[n-1] + max_change
if y[n] - y[n-1] < -max_change: y[n] = y[n-1] - max_change
```
- **Params:** max_slew ∈ [0.001, 0.3]
- **Perceptual:** Smoothed transients, rounded, low-pass-like but nonlinear
- **Numba:** Yes

## D011 — Diode Clipper Model
```
Shockley diode equation:
I = Is * (exp(V / (n*Vt)) - 1)
Approximate with cubic: y ≈ x - (x³/3) for |x| < 1
```
- **Params:** forward_voltage ∈ [0.2, 0.7], num_diodes ∈ [1, 4]
- **Perceptual:** Germanium warmth vs silicon edge
- **Numba:** Yes

## D012 — Rectification Distortion
```
Half-wave: y[n] = max(0, x[n])
Full-wave: y[n] = |x[n]|
Biased: y[n] = max(bias, x[n]) - bias
```
- **Params:** type ∈ {half, full, biased}, bias ∈ [-0.5, 0.5]
- **Perceptual:** Octave-up effect (full wave), buzzy (half wave)
- **Numba:** Yes

## D013 — Dynamic Waveshaping
```
Transfer curve changes based on input envelope:
env[n] = α * |x[n]| + (1-α) * env[n-1]
drive_dynamic[n] = base_drive + env_drive * env[n]
y[n] = tanh(drive_dynamic[n] * x[n])
```
- **Params:** base_drive ∈ [1, 5], env_drive ∈ [0, 20], env_speed α ∈ [0.001, 0.1]
- **Perceptual:** Louder parts get more distorted — dynamic, responsive character
- **Numba:** Yes

## D014 — XOR / Bitwise Distortion
```
Quantize to N bits (int16)
y_int = x_int XOR pattern  (or AND, OR)
Convert back to float
```
- **Params:** bit_depth ∈ [8, 16], operation ∈ {xor, and, or}, pattern (bitmask)
- **Perceptual:** Harsh digital artifacts, aliasing, unique timbres
- **Numba:** Yes

## D015 — Modular Arithmetic Distortion
```
y[n] = ((x[n] * scale + offset) % modulus) - modulus/2
Normalized to [-1, 1]
```
- **Params:** scale ∈ [1, 10], modulus ∈ [0.2, 2.0], offset ∈ [0, 1]
- **Perceptual:** Sawtooth-like wavefolding, creates complex harmonics
- **Numba:** Yes

## D016 — Serge-Style Wavefolder
```
Multi-stage wavefolder with shaped transfer functions:
  sine fold:     x = sin(π * x)   — smooth harmonic generation
  triangle fold: x = 2*|2*(x/2 - floor(x/2 + 0.5))| - 1  — sharp folds
  tanh fold:     x = tanh(sin(x * π/2))  — soft saturated folds
Each stage re-folds the output, adding harmonics multiplicatively.
Asymmetry parameter biases input for even harmonic content.
```
- **Params:** pre_gain ∈ [2, 10], fold_type ∈ {sine, triangle, tanh}, stages ∈ [1, 3], asymmetry ∈ [0, 0.4]
- **Perceptual:** Richer harmonics than D004's symmetric foldback — sine fold is smooth, triangle is aggressive, tanh is warm
- **Numba:** Yes

---

# E. FILTER EFFECTS

## E001 — Biquad Filter (Parametric EQ)
```
y[n] = (b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]) / a0
Coefficient formulas for LPF, HPF, BPF, Notch, Peak, LowShelf, HighShelf
from Audio EQ Cookbook (Robert Bristow-Johnson)
```
- **Params:** type, freq_hz ∈ [20, 20000], Q ∈ [0.1, 20], gain_db ∈ [-24, 24]
- **Perceptual:** Standard EQ — boost/cut frequency bands
- **Numba:** Yes, direct form II transposed

## E002 — State Variable Filter
```
hp = x[n] - lp - q*bp
bp += f * hp
lp += f * bp
where f = 2 * sin(π * cutoff / sr), q = 1/Q
Outputs LP, HP, BP, Notch (=HP+LP) simultaneously
```
- **Params:** cutoff_hz ∈ [20, 20000], Q ∈ [0.5, 30], output_type
- **Perceptual:** Resonant filter with selectable output
- **Numba:** Yes

## E003 — Moog Ladder Filter
```
4 cascaded one-pole filters with feedback:
stage[k] = stage[k] + tune * (tanh(input) - tanh(stage[k]))
input for each stage is previous stage output
Global feedback: input = x[n] - resonance * stage[3]
tune = 1 - exp(-2π * cutoff / sr)
```
- **Params:** cutoff_hz ∈ [20, 20000], resonance ∈ [0, 4] (self-oscillates > 3.8)
- **Perceptual:** Fat, squelchy, iconic analog lowpass
- **Numba:** Yes

## E004 — Comb Filter
```
Feedforward: y[n] = x[n] + g * x[n - d]
Feedback: y[n] = x[n] + g * y[n - d]
```
- **Params:** delay_samples (frequency = sr/delay), g ∈ [-0.99, 0.99]
- **Perceptual:** Metallic, tuned resonance at harmonics of fundamental
- **Numba:** Yes

## E005 — Formant Filter (Vowel Shaping)
```
3 parallel bandpass filters tuned to vowel formant frequencies:
  /a/: 800, 1200, 2500 Hz
  /e/: 350, 2000, 2800 Hz
  /i/: 270, 2300, 3000 Hz
  /o/: 500, 800, 2800 Hz
  /u/: 325, 700, 2500 Hz
Each BPF implemented as biquad with appropriate Q
```
- **Params:** vowel ∈ {a, e, i, o, u} or formant freqs directly, Q ∈ [5, 20]
- **Perceptual:** Makes any audio "speak" a vowel
- **Numba:** Yes, parallel biquads

## E006 — Vowel Morph Filter
```
Interpolate formant frequencies between two vowels over time:
f_k(t) = lerp(vowel_A_f[k], vowel_B_f[k], t/duration)
Sweep t from 0 to 1
```
- **Params:** vowel_from, vowel_to, morph_rate_hz ∈ [0.1, 5], Q
- **Perceptual:** Talking/morphing vowel sound, wah-like
- **Numba:** Yes

## E007 — Auto-Wah (Envelope Follower → Filter)
```
env[n] = α * |x[n]| + (1-α) * env[n-1]
cutoff[n] = min_freq + env[n] * (max_freq - min_freq)
Apply SVF or biquad LPF with cutoff[n]
```
- **Params:** min_freq ∈ [100, 500], max_freq ∈ [1000, 8000], sensitivity α ∈ [0.001, 0.05], Q ∈ [2, 15]
- **Perceptual:** Funky wah, filter opens with louder playing
- **Numba:** Yes

## E008 — Resonant Filter Sweep
```
Sweep cutoff frequency linearly or logarithmically over duration
LPF or BPF with high Q
```
- **Params:** start_freq, end_freq, Q ∈ [5, 30], filter_type ∈ {lpf, bpf}
- **Perceptual:** Classic EDM/synth filter sweep
- **Numba:** Yes

## E009 — Multi-Mode Filter Crossfade
```
Compute LP, HP, BP simultaneously
Crossfade between them based on LFO or envelope:
y = w_lp*lp + w_hp*hp + w_bp*bp where weights sum to 1
```
- **Params:** morph_rate_hz, Q, cutoff
- **Perceptual:** Evolving, morphing filter character
- **Numba:** Yes

## E010 — Cascade of Detuned Resonators
```
N bandpass filters with center frequencies spread around a base:
f_k = base_freq * (1 + k * detune)
Sum all outputs
```
- **Params:** base_freq ∈ [200, 2000], num_resonators ∈ [3, 8], detune ∈ [0.01, 0.1], Q ∈ [10, 50]
- **Perceptual:** Shimmering, chorus-like resonance, metallic shimmer
- **Numba:** Yes

## E011 — Allpass Lattice Filter
```
Cascade of first-order lattice allpass sections:
  y = k*x + state
  state_next = x - k*y
Each stage has a coupled coefficient k.
Mix original + allpass output to create notch patterns.
Different from phaser (C003): lattice coupling creates distinct harmonic relationships.
```
- **Params:** num_stages ∈ [4, 16], base_coeff ∈ [-0.5, 0.7], spread ∈ [0.2, 0.6], mix ∈ [0.6, 0.9]
- **Perceptual:** Deep, complex notch patterns — speech-coding-derived timbral coloring
- **Numba:** Yes

## E012 — Pitch-Tracking Resonator
```
1. Detect input pitch via autocorrelation
2. Tune a bank of biquad BPF resonators to harmonics of detected pitch
3. Filter input through resonator bank — sympathetic resonance follows input
```
- **Params:** num_harmonics ∈ [4, 16], Q ∈ [10, 50], wet ∈ [0.4, 0.8]
- **Perceptual:** Sympathetic resonance that follows the input's pitch — harp/sitar-like shimmer
- **Numba:** Yes

---

# F. DYNAMICS

## F001 — Compressor
```
env[n] = peak or RMS envelope with attack/release:
  if |x[n]| > env[n-1]: env[n] = attack_coeff * env[n-1] + (1-attack_coeff) * |x[n]|
  else: env[n] = release_coeff * env[n-1] + (1-release_coeff) * |x[n]|
gain_db = min(0, threshold_db + (env_db - threshold_db) / ratio) - env_db
y[n] = x[n] * 10^(gain_db/20)
```
- **Params:** threshold_db ∈ [-40, 0], ratio ∈ [2, 20], attack_ms ∈ [0.1, 100], release_ms ∈ [10, 1000]
- **Perceptual:** Reduced dynamic range, louder quiet parts
- **Numba:** Yes

## F002 — Expander / Gate
```
Same envelope as compressor but:
gain_db = min(0, (env_db - threshold_db) * (ratio - 1) / ratio)
When env < threshold, signal is attenuated
```
- **Params:** threshold_db, ratio, attack_ms, release_ms
- **Perceptual:** Quiet parts get quieter, noise gate removes low-level signals
- **Numba:** Yes

## F003 — Transient Shaper
```
Detect transients via difference of envelopes:
fast_env: attack 0.1ms, release 1ms
slow_env: attack 10ms, release 100ms
transient = fast_env - slow_env (positive = attack, negative = sustain)
y[n] = x[n] * (1 + attack_gain * max(transient, 0) + sustain_gain * min(transient, 0))
```
- **Params:** attack_gain ∈ [-1, 2], sustain_gain ∈ [-1, 2]
- **Perceptual:** Punchier attacks or smoother sustain, independent control
- **Numba:** Yes

## F004 — Multiband Dynamics (3-band)
```
Split into low/mid/high via Linkwitz-Riley crossovers (4th order)
Apply independent compressor to each band
Sum bands
```
- **Params:** low_xover ∈ [100, 400], high_xover ∈ [2000, 8000], per-band threshold/ratio/attack/release
- **Perceptual:** Frequency-dependent dynamics control, loudness maximizing
- **Numba:** Yes for compressors, filters

## F005 — Sidechain Ducker
```
Generate a synthetic sidechain signal (4-on-the-floor kick pattern, sine burst every beat_ms)
Use sidechain envelope to duck the input
y[n] = x[n] * (1 - duck_amount * sidechain_env[n])
```
- **Params:** beat_ms ∈ [200, 600], duck_amount ∈ [0.3, 1.0], attack_ms ∈ [1, 10], release_ms ∈ [50, 300]
- **Perceptual:** Rhythmic pumping, EDM sidechain effect
- **Numba:** Yes

---

# G. PITCH & TIME

## G001 — Phase Vocoder Pitch Shift
```
STFT → shift bins up/down by semitone ratio → phase correction → ISTFT
bin_shift = round(original_bin * 2^(semitones/12))
Phase accumulation: Δφ corrected for bin shift
```
- **Params:** semitones ∈ [-24, 24], window_size ∈ [1024, 4096], hop_size
- **Perceptual:** Pitch change without time change (with some artifacts)
- **NumPy:** FFT-based

## G002 — Granular Pitch Shift
```
Read grains from input at original speed
Play grains at resampled speed (ratio = 2^(semitones/12))
Overlap-add with Hann windows
```
- **Params:** semitones ∈ [-24, 24], grain_size_ms ∈ [20, 100]
- **Perceptual:** Pitch shift with granular artifacts, less phasey than vocoder
- **Numba:** Yes

## G003 — Harmonizer
```
Mix original with one or more pitch-shifted copies
shifts typically: +3, +5, +7, +12 semitones (musical intervals)
```
- **Params:** intervals_semitones (list), per-voice gain, wet_mix
- **Perceptual:** Harmony, chord-like thickening
- **NumPy:** Uses pitch shift internally

## G004 — Octave Up via Full-Wave Rectification + Filter
```
y = bandpass(|x|, original_fundamental * 2, Q=5)
Mix with original
```
- **Params:** fundamental_hz (or auto-detect), filter_Q, wet_mix
- **Perceptual:** Analog-style octave up, gritty
- **Numba:** Yes

## G005 — Time Stretch (WSOLA)
```
Waveform Similarity Overlap-Add:
For each output frame, search for best overlap position within tolerance
Cross-correlate to find alignment
Crossfade between frames
```
- **Params:** stretch_factor ∈ [0.25, 4.0], window_ms ∈ [20, 80], tolerance_ms ∈ [5, 20]
- **Perceptual:** Time stretch without pitch change, better transients than OLA
- **Numba:** Cross-correlation loop

## G006 — Paulstretch (Extreme Time Stretch)
```
For each frame:
  X = FFT(windowed_frame)
  magnitudes = |X|
  phases = uniform_random(0, 2π)
  frame_out = IFFT(magnitudes * exp(j * phases))
Overlap-add with large stretch factor
```
- **Params:** stretch_factor ∈ [2, 100], window_size ∈ [2048, 65536]
- **Perceptual:** Frozen, ambient, drone-like texture from any input
- **NumPy:** FFT-based

## G007 — Formant-Preserving Pitch Shift
```
1. Extract spectral envelope (LPC or cepstral method)
2. Pitch shift the fine structure
3. Re-apply original spectral envelope
```
- **Params:** semitones ∈ [-12, 12], lpc_order ∈ [10, 40]
- **Perceptual:** Pitch shift that keeps the "character" of voice/timbre
- **NumPy:** LPC + FFT

---

# H. SPECTRAL / FREQUENCY DOMAIN

## H001 — Spectral Freeze
```
Take one FFT frame's magnitudes at freeze_position
For all subsequent frames, keep those magnitudes, advance phase by hop_size * bin_freq
y = ISTFT with frozen magnitudes + advancing phases
```
- **Params:** freeze_position (0.0–1.0 through file), window_size
- **Perceptual:** Infinite sustain of a single moment, drone
- **NumPy:** STFT/ISTFT

## H002 — Spectral Blur / Smear
```
For each STFT frame:
  magnitudes = gaussian_filter_1d(magnitudes, sigma=blur_width)
Averaging neighboring bins blurs the spectrum
```
- **Params:** blur_width ∈ [1, 50] bins
- **Perceptual:** Smeared, dreamy, loss of pitch definition
- **NumPy:** STFT + scipy-style convolution

## H003 — Spectral Gate
```
For each frame:
  threshold = percentile(magnitudes, gate_percentile)
  magnitudes[magnitudes < threshold] = 0
  or *= smooth_gate_curve
```
- **Params:** gate_percentile ∈ [50, 99], mode ∈ {hard, soft}
- **Perceptual:** Only strongest components survive — whispered, ghostly at high thresholds
- **NumPy:** STFT

## H004 — Spectral Shift (Bin Shift)
```
For each frame:
  new_magnitudes[k + shift] = magnitudes[k]
  new_phases[k + shift] = phases[k]
Shift all bins up or down by N bins
```
- **Params:** shift_bins ∈ [-100, 100]
- **Perceptual:** Inharmonic, metallic frequency shift (NOT pitch shift)
- **NumPy:** STFT

## H005 — Phase Randomization (Whisperization)
```
For each frame:
  magnitudes unchanged
  phases = uniform_random(0, 2π)
```
- **Params:** amount ∈ [0, 1] (interpolate between original and random phase)
- **Perceptual:** Breathy, whispered, loss of transients. At 100% → noise-like with same spectrum
- **NumPy:** STFT

## H006 — Robotization (Zero Phase)
```
For each frame:
  magnitudes unchanged
  phases = 0 (or constant)
```
- **Params:** (minimal — just window_size affects character)
- **Perceptual:** Buzzy, robotic, metallic — all partials phase-aligned creates impulse train
- **NumPy:** STFT

## H007 — Spectral Bin Sorting
```
For each frame:
  Sort magnitude bins by value (ascending or descending)
  Remap phases accordingly
```
- **Params:** order ∈ {ascending, descending}, partial_sort (only sort top N%)
- **Perceptual:** Bizarre, alien redistribution of spectral energy
- **NumPy:** STFT

## H008 — Spectral Bin Permutation
```
For each frame:
  Apply a fixed random permutation to bin indices
  magnitudes_new[perm[k]] = magnitudes[k]
  phases_new[perm[k]] = phases[k]
```
- **Params:** seed, permutation_amount ∈ [0, 1] (blend between identity and random permutation)
- **Perceptual:** Scrambled spectrum, alien but preserves overall energy
- **NumPy:** STFT

## H009 — Spectral Cross-Synthesis
```
Take magnitudes from signal A, phases from signal B (the input)
Or: multiply magnitudes of A and B
Synthesized signal: A filtered by the spectrum of B
```
- **Params:** source_signal (can be synthetic: noise, sine sweep, etc.), blend ∈ [0, 1]
- **Perceptual:** Hybrid timbre, vocoder-like
- **NumPy:** STFT of both signals

## H010 — Classic Channel Vocoder
```
Analysis: bandpass filter bank on modulator (voice), extract envelopes
Synthesis: apply envelopes to carrier (synth/noise) via same filter bank
N bands, typically 16–64
```
- **Params:** num_bands ∈ [8, 64], freq_range ∈ [80, 8000], carrier_type ∈ {noise, saw, input_self}
- **Perceptual:** Talking robot, Kraftwerk-style
- **Numba:** Yes, parallel filters + envelope followers

## H011 — Harmonic / Percussive Separation
```
Compute STFT magnitude spectrogram
Harmonic: median filter along time axis (horizontal)
Percussive: median filter along frequency axis (vertical)
Create masks: H_mask = H / (H + P + eps), P_mask = P / (H + P + eps)
Output either component or recombined differently
```
- **Params:** filter_length ∈ [5, 31], output ∈ {harmonic, percussive, remix_with_weights}
- **Perceptual:** Isolate tonal vs transient content
- **NumPy:** Median filter on 2D array

## H012 — Spectral Mirror
```
For each frame:
  Mirror magnitudes around a center frequency
  mag_new[center + k] = mag[center - k]
```
- **Params:** mirror_center_hz ∈ [500, 5000]
- **Perceptual:** Reversed spectral content, eerie, alien
- **NumPy:** STFT

## H013 — Spectral Stretch / Compress
```
Resample the magnitude spectrum to stretch or compress it
stretch > 1: spread partials apart (inharmonic)
stretch < 1: compress partials together
```
- **Params:** spectral_stretch ∈ [0.5, 2.0]
- **Perceptual:** Inharmonic at non-1.0 values, bell-like, metallic
- **NumPy:** STFT + interpolation

## H014 — Cepstral Processing
```
Cepstrum = IFFT(log(|FFT(x)|))
Lifter: zero out high or low quefrency components
Low-pass lifter → spectral envelope only
High-pass lifter → fine pitch structure only
Modify one, recombine, resynthesize
```
- **Params:** lifter_cutoff_quefrency, operation ∈ {smooth_envelope, remove_pitch, modify_envelope}
- **Perceptual:** Separate and independently modify timbre vs pitch structure
- **NumPy:** Multiple FFTs

## H015 — Spectral Delay
```
Each frequency bin has its own delay time
delay_per_bin = base_delay + bin_index * delay_slope
Process in STFT domain, shifting each bin's phase appropriately
```
- **Params:** base_delay_ms ∈ [0, 100], delay_slope_ms_per_bin ∈ [0, 1]
- **Perceptual:** Low frequencies arrive before high (or vice versa) — chirp-like dispersion
- **NumPy:** STFT

## H016 — Spectral Compressor
```
For each frame, for each bin:
  if mag[k] > threshold: mag[k] = threshold + (mag[k] - threshold) / ratio
Compress dynamics within the spectrum
```
- **Params:** threshold_db ∈ [-60, -10], ratio ∈ [2, 20]
- **Perceptual:** Flattened spectrum, everything becomes equally loud, dense
- **NumPy:** STFT

## H017 — Spectral Reassignment
```
Compute instantaneous frequency and group delay for each STFT bin
Reassign energy to more accurate time-frequency positions
Sharper spectrogram → resynthesize
```
- **Params:** sharpening_amount ∈ [0, 1]
- **Perceptual:** Sharpened transients and spectral lines, hyper-real clarity
- **NumPy:** Multiple STFTs with different windows

## H018 — Spectral Subtraction
```
For each frame:
  noise_floor[k] = percentile(all_magnitudes[k], noise_percentile)
  cleaned[k] = max(0, mag[k] - subtraction_factor * noise_floor[k])
At moderate settings: denoiser. At extreme: "musical noise" artifacts.
```
- **Params:** subtraction_factor ∈ [1, 15], noise_percentile ∈ [10, 50]
- **Perceptual:** Gentle denoise → ghostly tonal residuals at extreme settings
- **NumPy:** STFT

## H019 — Spectral Transfer / Timbre Stamp
```
1. Extract spectral envelope of input via cepstral smoothing
2. Generate carrier (noise, chirp, or pulse train)
3. Apply input's envelope to carrier magnitudes
4. Blend with original
Unlike vocoder (H010): uses cepstral envelope for phase-coherent transfer.
```
- **Params:** carrier_type ∈ {noise, chirp, pulse}, envelope_order ∈ [10, 40], blend ∈ [0.5, 1.0]
- **Perceptual:** Input timbre stamped onto carrier — smoother than vocoder
- **NumPy:** STFT + cepstrum

---

# I. GRANULAR PROCESSING

## I001 — Granular Cloud
```
Schedule grains at random positions in the source
Each grain: extract, window (Hann), pitch shift (resample), amplitude scale
Grain scheduling: Poisson process at density grains/sec
Overlap-add all grains
```
- **Params:** grain_size_ms ∈ [10, 200], density ∈ [5, 100], position_spread ∈ [0, 1], pitch_spread_semitones ∈ [0, 12], amplitude_spread ∈ [0, 0.5]
- **Perceptual:** Cloudy, textural, from recognizable to completely abstract
- **Numba:** Grain loop

## I002 — Granular Freeze
```
Fix grain read position to one point in the source
Continually spawn grains from that position with slight random variation
```
- **Params:** freeze_position (0–1), position_jitter_ms ∈ [0, 50], pitch_jitter ∈ [0, 2] semitones, density, grain_size_ms
- **Perceptual:** Sustained texture from a single moment
- **Numba:** Yes

## I003 — Granular Time Stretch
```
Move grain read position slower than real time
read_speed = 1 / stretch_factor
Grains still played at original pitch
```
- **Params:** stretch_factor ∈ [1, 50], grain_size_ms, density, overlap
- **Perceptual:** Slow-motion audio with granular texture, less phasey than FFT methods
- **Numba:** Yes

## I004 — Granular Reverse Scatter
```
Each grain has 50% chance of being played forward or reversed
Position scanning through file normally
```
- **Params:** reverse_probability ∈ [0, 1], grain_size_ms, density
- **Perceptual:** Glitchy, stuttering, partially backwards
- **Numba:** Yes

## I005 — Granular Pitch Cloud
```
All grains from same position but each with different random pitch
pitch per grain drawn from uniform or normal distribution
```
- **Params:** center_semitones ∈ [-12, 12], spread_semitones ∈ [0, 24], grain_size_ms, density
- **Perceptual:** Thick pitch clusters, chord-like, from unison to chaos
- **Numba:** Yes

## I006 — Granular Density Ramp
```
Grain density ramps from sparse (1/sec) to dense (200/sec) over duration
At low density: distinct grains. At high density: smooth texture
```
- **Params:** start_density, end_density, grain_size_ms, ramp_curve ∈ {linear, exponential}
- **Perceptual:** Transformation from discrete events to continuous texture
- **Numba:** Yes

## I007 — Microsound Particles
```
Extremely short grains (1–10ms) scattered in time
Each particle: tiny burst of audio, possibly pitch-shifted
Ultra-high density creates dense textures
```
- **Params:** grain_size_ms ∈ [1, 10], density ∈ [100, 1000], pitch_range
- **Perceptual:** Buzzy, insect-like, textural, Curtis Roads microsound territory
- **Numba:** Yes

---

# J. CHAOS & MATHEMATICAL EFFECTS

## J001 — Logistic Map Modulator
```
Logistic map: x_{n+1} = r * x_n * (1 - x_n)
Use output sequence to modulate amplitude, filter cutoff, or delay time
Step the map once per N samples
```
- **Params:** r ∈ [3.5, 4.0], step_rate (samples per step), mod_target ∈ {amplitude, cutoff, delay}, mod_depth
- **Perceptual:** Quasi-periodic to chaotic modulation, evolving unpredictability
- **Numba:** Yes

## J002 — Logistic Map Waveshaper
```
Use logistic map as a transfer function:
Run map starting from x0 = |input_sample|
Take value after N iterations as output
```
- **Params:** r ∈ [3.5, 4.0], num_iterations ∈ [1, 20]
- **Perceptual:** Chaotic distortion that depends on input level
- **Numba:** Yes

## J003 — Lorenz Attractor Modulation
```
dx/dt = σ(y - x)
dy/dt = x(ρ - z) - y
dz/dt = xy - βz
Integrate with Euler method, normalize output to [-1, 1]
Use x, y, z to modulate three different parameters simultaneously
```
- **Params:** σ = 10, ρ ∈ [20, 35], β = 8/3, integration_speed, mod_targets, mod_depths
- **Perceptual:** Three coupled chaotic modulators — complex, evolving, never repeating
- **Numba:** Yes

## J004 — Lorenz Attractor as Audio-Rate Signal
```
Same equations but integrated at audio rate
Normalize x/y/z outputs to [-1, 1]
Ring-mod or mix with input signal
```
- **Params:** σ, ρ, β, mix_mode ∈ {ring_mod, add, am}
- **Perceptual:** Chaotic oscillator tones mixed with input
- **Numba:** Yes

## J005 — Henon Map Distortion
```
x_{n+1} = 1 - a * x_n² + y_n
y_{n+1} = b * x_n
Use x sequence as waveshaping LUT or direct modulator
```
- **Params:** a ∈ [1.0, 1.4], b ∈ [0.2, 0.4]
- **Perceptual:** Strange-attractor-shaped distortion character
- **Numba:** Yes

## J006 — Duffing Oscillator Resonator
```
x'' + δx' + αx + βx³ = γ*input(t)*cos(ωt)
Driven nonlinear resonator — can exhibit chaos
Euler or RK4 integration at audio rate
Drive it with the input signal
```
- **Params:** δ (damping) ∈ [0.1, 0.5], α ∈ [-1, 1], β ∈ [0.5, 2], γ ∈ [0.1, 1.5], ω_hz
- **Perceptual:** Nonlinear resonance — input excites chaotic oscillation
- **Numba:** Yes

## J007 — Double Pendulum Parameter Modulation
```
Full double pendulum equations of motion (θ1, θ2, ω1, ω2)
RK4 integration, normalize angles to [-1, 1]
Use θ1 → filter cutoff, θ2 → delay time (or any param pair)
```
- **Params:** m1, m2, l1, l2, initial_angles, integration_speed
- **Perceptual:** Physically-motivated chaotic modulation with visual intuition
- **Numba:** Yes

## J008 — Cellular Automaton Rhythm Gate
```
Run 1D cellular automaton (e.g., Rule 30, 90, 110)
Each cell state (0/1) controls whether a time slice of audio is gated on/off
Row = time steps, cells = audio chunks
```
- **Params:** rule ∈ [0, 255], num_cells ∈ [16, 128], cell_duration_ms ∈ [10, 100]
- **Perceptual:** Complex, emergent rhythmic gating patterns — more interesting than random
- **Numba:** Yes

## J009 — Cellular Automaton Spectral Mask
```
2D cellular automaton on spectrogram grid:
time → CA generations, frequency → cell positions
CA state (0/1) masks spectral bins
```
- **Params:** rule, initial_state, num_generations
- **Perceptual:** Evolving spectral patterns, emergent timbral structures
- **NumPy:** STFT + CA logic

## J010 — Reaction-Diffusion Spectrogram
```
Treat spectrogram magnitudes as concentration field
Run Gray-Scott reaction-diffusion:
  du/dt = Du∇²u - uv² + F(1-u)
  dv/dt = Dv∇²v + uv² - (F+k)v
Time axis = frequency, generations = new frames
Resynthesize modified spectrogram
```
- **Params:** F ∈ [0.02, 0.06], k ∈ [0.04, 0.07], Du, Dv, num_iterations ∈ [10, 200]
- **Perceptual:** Organic, evolving, pattern-forming spectral modification. Spots/stripes in spectrum
- **NumPy:** 2D convolution + STFT

## J011 — L-System Parameter Sequencer
```
L-system grammar generates string of symbols
Map symbols to parameter changes:
  F → increase cutoff, + → increase delay, - → decrease delay, [ → push state, ] → pop state
Use generated sequence to automate any effect's parameters over time
```
- **Params:** axiom, rules, iterations, symbol_mapping, step_duration_ms
- **Perceptual:** Self-similar, fractal parameter evolution
- **Numba:** String generation, then parameter application

## J012 — Iterated Function System Audio
```
IFS: apply contractive affine transforms to audio chunks
T_k(x) = a_k * x + b_k (compression + translation in time domain)
Select transform probabilistically, iterate
Like fractal image compression but for waveforms
```
- **Params:** num_transforms ∈ [2, 5], contraction_ratios, offsets, probabilities, iterations ∈ [3, 10]
- **Perceptual:** Self-similar, fractal-like waveform structures
- **Numba:** Yes

## J013 — Fibonacci / Golden Ratio Rhythmic Gate
```
Gate on/off pattern based on Fibonacci word:
Start: "1", "10", "101", "10110", "10110101", ...
Or: gate intervals at golden ratio multiples of base_ms
```
- **Params:** base_ms ∈ [20, 200], num_generations ∈ [5, 12]
- **Perceptual:** Quasi-periodic rhythm that feels organic and non-repetitive
- **Numba:** Yes

## J014 — Brownian Motion Parameter Walk
```
param[n] = param[n-1] + σ * randn()
Clamp to valid range, optionally low-pass filter the walk
Apply to filter cutoff, delay time, pitch, etc.
```
- **Params:** σ (step size), target_param, smoothing, range_min, range_max
- **Perceptual:** Drifting, slowly evolving parameter changes
- **Numba:** Yes

## J015 — Strange Attractor Trajectory as Spectral Curve
```
Run Lorenz/Rössler/Chen attractor, project 3D trajectory to 2D (time, frequency)
Use trajectory points as (time, frequency) pairs
Boost or cut spectral bins near the trajectory path
```
- **Params:** attractor_type, attractor_params, trajectory_width_bins, boost_db ∈ [3, 20]
- **Perceptual:** Spectral energy follows a chaotic, never-repeating path through the spectrum
- **NumPy:** STFT + attractor integration

## J016 — Möbius Transform on Spectrum
```
FFT → complex spectrum z[k] = mag[k] * exp(j*phase[k])
Apply Möbius transform: w = (a*z + b) / (c*z + d) where ad-bc ≠ 0
Convert back to mag/phase, IFFT
```
- **Params:** a, b, c, d ∈ ℂ (randomly sampled or parameterized)
- **Perceptual:** Conformal spectral warping — preserves some structure while distorting
- **NumPy:** STFT

## J017 — Fractal Delay Network
```
Delay times at self-similar ratios:
d, d/r, d/r², d/r³, ... for ratio r
Feedback from longest delay back to shortest
```
- **Params:** base_delay_ms ∈ [50, 500], ratio ∈ [1.5, 3.0], num_levels ∈ [3, 7], feedback ∈ [0.3, 0.8]
- **Perceptual:** Self-similar echo pattern, fractal temporal structure
- **Numba:** Yes

## J018 — Audio Boids (Spectral Flocking)
```
Treat top-N spectral peaks as "boids" (agents)
Each boid has position (frequency) and velocity
Apply flocking rules: separation, alignment, cohesion
Boids move through spectrum, carrying energy with them
```
- **Params:** num_boids ∈ [5, 30], separation_weight, alignment_weight, cohesion_weight, speed
- **Perceptual:** Spectral peaks that move in coordinated, organic patterns
- **NumPy:** STFT + boid simulation per frame

## J019 — Stochastic Resonance
```
Add calibrated noise to signal such that sub-threshold features become detectable
y[n] = threshold_detector(x[n] + σ * noise[n])
Or: soft version using noise to modulate a filter's sensitivity
```
- **Params:** noise_amplitude ∈ [0.01, 0.5], threshold ∈ [0.05, 0.3]
- **Perceptual:** Paradoxically, adding noise reveals hidden detail in quiet passages
- **Numba:** Yes

## J020 — Chua's Circuit Oscillator as Effect
```
dx/dt = α(y - x - f(x))    where f(x) is piecewise linear
dy/dt = x - y + z
dz/dt = -βy
f(x) = m1*x + 0.5*(m0-m1)*(|x+1| - |x-1|)
Drive with input signal or use as modulator
```
- **Params:** α ∈ [9, 16], β ∈ [14, 15], m0 = -1/7, m1 = 2/7, drive_mode
- **Perceptual:** Rich chaotic oscillation, double-scroll attractor character
- **Numba:** Yes

---

# K. NEURAL / LEARNED EFFECTS

## K001 — Random Neural Network Waveshaper
```
3-layer MLP with random weights (never trained):
h1 = tanh(W1 @ x_chunk + b1)  (input: chunk of N samples)
h2 = tanh(W2 @ h1 + b2)
y  = W3 @ h2 + b3
Different random seeds → completely different nonlinear transforms
```
- **Params:** hidden_size ∈ [16, 128], chunk_size ∈ [32, 256], seed
- **Perceptual:** Complex, unpredictable nonlinear transform. Each seed = different "character"
- **NumPy:** Matrix multiplies

## K002 — Tiny Autoencoder (Compress → Corrupt → Decompress)
```
Encoder: h = tanh(W_enc @ x_chunk + b_enc)  → latent (small dim)
Corrupt: h_corrupted = h + noise * σ, or zero random dims, or scale
Decoder: y = W_dec @ h_corrupted + b_dec
Train on input signal itself (overfit): MSE loss, simple gradient descent
Then corrupt and decode
```
- **Params:** latent_dim ∈ [4, 32], chunk_size ∈ [64, 512], corruption_type ∈ {noise, dropout, scale}, corruption_amount, training_epochs ∈ [50, 500]
- **Perceptual:** Lossy reconstruction, like audio through a bottleneck with imperfections
- **Hand-rolled or tinygrad:** Gradient descent needed

## K003 — Echo State Network (Reservoir Computing)
```
Reservoir: N recurrently connected nodes with fixed random weights
x_state[n] = tanh(W_in * input[n] + W_res @ x_state[n-1])
output[n] = W_out @ x_state[n]
W_out trained via linear regression on input signal (predict next sample or identity)
```
- **Params:** reservoir_size ∈ [50, 500], spectral_radius ∈ [0.8, 1.2], input_scaling, leak_rate ∈ [0.1, 1.0]
- **Perceptual:** Complex temporal filtering with memory, resonances emerge from topology
- **NumPy:** Matrix ops, linear regression for readout

## K004 — Neural ODE Audio Processor
```
Model: dx/dt = f(x, t) where f is a small neural network
Integrate forward with Euler: x[t+dt] = x[t] + dt * f(x[t], t)
f = single hidden layer MLP with random or trained weights
Process overlapping chunks of audio
```
- **Params:** hidden_size ∈ [16, 64], dt ∈ [0.01, 0.1], num_steps ∈ [5, 50], seed
- **Perceptual:** Smooth, flow-like transformation of audio chunks
- **NumPy:** Matrix ops

## K005 — Weight Space Interpolation
```
Generate two random MLPs (same architecture, different random seeds)
Interpolate all weights: W_mix = α*W_A + (1-α)*W_B
Process audio through the interpolated network
Sweep α from 0 to 1 for morphing
```
- **Params:** hidden_size, num_layers ∈ [2, 4], seed_a, seed_b, alpha ∈ [0, 1]
- **Perceptual:** Morphing between two nonlinear transforms
- **NumPy:** Matrix ops

## K006 — 1D Convnet Filter Bank
```
N random 1D convolutional kernels of various sizes
For each kernel: output = relu(conv1d(input, kernel))
Sum or interleave outputs
Kernels can be random or trained (overfit on input)
```
- **Params:** num_kernels ∈ [4, 32], kernel_sizes ∈ [3, 64], seed
- **Perceptual:** Random learned filter bank — each seed discovers different spectral features
- **NumPy:** np.convolve

## K007 — Tiny RNN Sample Processor
```
Single RNN cell processing sample-by-sample:
h[n] = tanh(w_ih * x[n] + w_hh * h[n-1] + b)
y[n] = w_ho * h[n]
Hidden state creates memory/context between samples
Random weights or train via BPTT on small chunks
```
- **Params:** hidden_size ∈ [4, 32], seed (for random) or train_epochs
- **Perceptual:** Nonlinear filtering with temporal memory, unique resonant character per seed
- **Numba:** Yes — single sample loop with small matrix

## K008 — Overfit-Then-Corrupt (tinygrad candidate)
```
Train a small MLP to perfectly reconstruct chunks of input (overfit)
Then systematically corrupt:
  - Add noise to weights
  - Quantize weights to fewer bits
  - Prune (zero out) random weights
  - Scale certain layers
Regenerate audio through corrupted model
```
- **Params:** model_size, corruption_type, corruption_amount, train_epochs
- **Perceptual:** Graceful degradation of learned representation — unique artifacts
- **tinygrad:** Autograd for training, then export weights for NumPy inference

## K009 — Random Projection Dimensionality Transform
```
Take chunks of N samples
Project to lower dim: h = W_down @ chunk  (random Gaussian matrix)
Project back up: y = W_up @ h  (another random matrix, or pseudoinverse)
Information loss creates spectral effects
```
- **Params:** chunk_size ∈ [32, 256], bottleneck_dim ∈ [2, chunk_size//2], seed
- **Perceptual:** Lossy, creates spectral ghosts and resonances based on random subspace
- **NumPy:** Matrix ops

---

# L. CONVOLUTION & MORPHING

## L001 — Convolve with Mathematical IR
```
Generate IR from mathematical function:
  exponential_decay: ir[n] = exp(-n * decay) * noise[n]
  sinc: ir[n] = sinc(2π * freq * n / sr)
  chirp: ir[n] = sin(2π * (f0 + rate*n) * n / sr) * exp(-n*decay)
  Gaussian: ir[n] = exp(-n²/(2σ²))
Convolve with input via FFT
```
- **Params:** ir_type, ir_params (per type), ir_length_ms
- **Perceptual:** Each mathematical shape creates different spatial/timbral character
- **NumPy:** FFT convolution

## L002 — Convolve with Self (Auto-Convolution)
```
y = ifft(fft(x) * fft(x))
Normalize output
Can iterate: convolve result with original again
```
- **Params:** num_iterations ∈ [1, 4]
- **Perceptual:** Smeared, elongated, progressively Gaussian envelope. Spectral squaring.
- **NumPy:** FFT

## L003 — Deconvolution Effect
```
y = ifft(fft(x) / (fft(ir) + epsilon))
Using a synthetic IR, this "removes" that IR's character
Epsilon prevents division by zero (Wiener-style regularization)
```
- **Params:** ir_type (same as L001), epsilon ∈ [0.001, 0.1]
- **Perceptual:** Sharpened transients, removed resonance, can be unstable = interesting artifacts
- **NumPy:** FFT

## L004 — Spectral Morphing
```
Morph between magnitude spectrum of input and a target:
mag_out = (1-α) * mag_input + α * mag_target
Phase from input
Target: synthetic spectrum (sawtooth, noise, formant shape, etc.)
```
- **Params:** target_type, alpha ∈ [0, 1]
- **Perceptual:** Gradual transformation from original timbre to target timbre
- **NumPy:** STFT

## L005 — Convolution with Chirp IR
```
IR = linear or exponential chirp from f_start to f_end
Convolving with chirp creates frequency-dependent delay (dispersion)
```
- **Params:** f_start ∈ [20, 1000], f_end ∈ [1000, 15000], chirp_duration_ms ∈ [10, 500]
- **Perceptual:** Dispersive, spring-reverb-like, different frequencies arrive at different times
- **NumPy:** FFT

## L006 — Morphological Audio Processing
```
Treat audio as 1D signal, apply morphological operations:
  Dilation: y[n] = max(x[n-k:n+k]) — expands peaks
  Erosion: y[n] = min(x[n-k:n+k]) — expands troughs
  Opening: erosion then dilation — removes narrow peaks
  Closing: dilation then erosion — fills narrow troughs
```
- **Params:** kernel_size ∈ [3, 51], operation ∈ {dilate, erode, open, close}
- **Perceptual:** Envelope shaping, transient modification, waveform smoothing
- **Numba:** Yes, sliding window

---

# M. PHYSICAL MODELING EFFECTS

## M001 — Karplus-Strong as Effect
```
Feed input into KS delay line:
  buffer initialized with input audio (not noise)
  y[n] = 0.5 * (buffer[n-N] + buffer[n-N-1])  (averaging filter)
  buffer[n] = y[n]
Delay length N determines pitch of resonance
```
- **Params:** freq_hz ∈ [50, 2000] (determines N = sr/freq), decay_factor ∈ [0.9, 0.999]
- **Perceptual:** Input audio resonates at specified pitch, plucked-string character
- **Numba:** Yes

## M002 — Waveguide String Resonator
```
Two delay lines (upper/lower rail) with reflections
Input injected at excitation point
Losses and dispersion filters at boundaries
More accurate than KS, models actual string physics
```
- **Params:** freq_hz, decay, brightness (filter at bridge), excitation_position (0–1 along string)
- **Perceptual:** Audio "played through" a vibrating string
- **Numba:** Yes

## M003 — Mass-Spring Damper Chain
```
N masses connected by springs with damping:
F_spring = -k * (x_i - x_{i-1})
F_damp = -c * v_i
x_i'' = (F_spring_left + F_spring_right + F_damp + F_input) / m
Euler integration at audio rate
Input signal drives first mass
Output from last mass (or any mass)
```
- **Params:** num_masses ∈ [3, 20], stiffness k, damping c, mass m, output_mass
- **Perceptual:** Mechanical resonance, propagating waves through a physical system
- **Numba:** Yes

## M004 — Membrane / Drum Resonator
```
2D waveguide mesh (simplified):
y[x,y,n] = c * (y[x+1,y,n-1] + y[x-1,y,n-1] + y[x,y+1,n-1] + y[x,y-1,n-1]) - y[x,y,n-2]
Drive with input signal at center point
Read output from edge
```
- **Params:** grid_size ∈ [5, 20], tension c, damping, drive_position
- **Perceptual:** Audio filtered through a vibrating membrane — drum-like resonance
- **Numba:** Yes, 2D grid update loop

## M005 — Tube Resonator (Acoustic Waveguide)
```
1D waveguide with varying cross-section (tube segments)
Scattering junctions between segments: partial reflection/transmission
Models vocal tract, brass instrument, etc.
Input is "breath" = audio signal
```
- **Params:** num_segments ∈ [3, 10], segment_radii (cross-sections), tube_length
- **Perceptual:** Audio "played through" a tube — formant-like resonances
- **Numba:** Yes

---

# N. LO-FI & TEXTURE

## N001 — Vinyl Crackle Overlay
```
Generate crackle: sparse random impulses with exponential decay
crackle[n] = A * delta[random_times] convolved with short decay
Add to signal at adjustable level
```
- **Params:** crackle_density ∈ [5, 100] per second, crackle_amplitude ∈ [0.01, 0.1], tone (filtered)
- **Perceptual:** Vintage record surface noise
- **Numba:** Yes

## N002 — Tape Hiss
```
Filtered noise: bandpass white noise (1kHz–8kHz), shaped with gentle rolloff
Modulate level slightly over time for realism
```
- **Params:** hiss_level ∈ [-40, -15] dBFS, color (bright vs warm)
- **Perceptual:** Analog tape background noise
- **NumPy:** Noise generation + filter

## N003 — Tape Wow and Flutter
```
Modulate playback speed with combined LFOs:
Wow: slow (0.5–3 Hz), larger depth
Flutter: fast (5–20 Hz), smaller depth
speed_mod[n] = 1 + wow_depth*sin(2π*wow_rate*n/sr) + flutter_depth*noise_filtered[n]
Variable-rate read from delay line
```
- **Params:** wow_rate, wow_depth ∈ [0.001, 0.01], flutter_rate, flutter_depth ∈ [0.0001, 0.002]
- **Perceptual:** Warped, unstable pitch — analog tape character
- **Numba:** Yes

## N004 — Telephone Effect
```
Bandpass filter: 300 Hz – 3400 Hz (telephone bandwidth)
Add subtle distortion (soft clip) and noise
```
- **Params:** low_cut ∈ [200, 500], high_cut ∈ [2500, 4000], distortion_amount, noise_level
- **Perceptual:** Tiny, lo-fi, telephone speaker
- **Numba:** Yes (biquad chain)

## N005 — Radio Tuning Effect
```
Sweep bandpass filter to "tune in" to signal
Add AM-style modulation and noise bursts
Intermittent signal: multiply by a slowly varying gate
```
- **Params:** sweep_rate, noise_level, modulation_depth, signal_clarity ∈ [0.3, 1.0]
- **Perceptual:** Searching through radio dial, finding the signal
- **Numba:** Yes

## N006 — Underwater Effect
```
Strong lowpass filter (~500 Hz) + subtle chorus + mild pitch wobble
Optional: add bubble sounds (short filtered noise bursts)
```
- **Params:** depth (controls cutoff 200–1000 Hz), bubble_density, chorus_rate
- **Perceptual:** Muffled, submerged, aquatic
- **Numba:** Yes

## N007 — AM Radio Effect
```
AM modulation: y[n] = x[n] * (1 + m * carrier[n])
Bandlimit to 5kHz
Add crackle and hum (50/60 Hz)
```
- **Params:** carrier_freq ∈ [500, 1500] kHz (simulated), modulation_index m, noise_level, hum_level
- **Perceptual:** Vintage AM radio broadcast
- **Numba:** Yes

---

# O. SPATIAL & STEREO

## O001 — Haas Effect (Stereo Widener)
```
L = x[n]
R = x[n - haas_delay_samples] * gain
Delay 1–30ms creates phantom stereo image
```
- **Params:** haas_delay_ms ∈ [1, 30], gain ∈ [0.5, 1.0]
- **Perceptual:** Mono signal becomes wide stereo
- **Numba:** Yes

## O002 — Mid-Side Processing
```
Mid = (L + R) / 2
Side = (L - R) / 2
Process mid and side independently (EQ, compress, etc.)
Recombine: L = Mid + Side, R = Mid - Side
```
- **Params:** mid_gain ∈ [-6, 6] dB, side_gain ∈ [-6, 12] dB
- **Perceptual:** Stereo width control, mono compatibility adjustment
- **Numba:** Yes

## O003 — Binaural Panning (Simplified HRTF)
```
Model interaural time difference (ITD) and level difference (ILD):
ITD = (d/c) * sin(θ) where d = head diameter, c = speed of sound
ILD approximated by frequency-dependent shelving filter
```
- **Params:** azimuth_degrees ∈ [-90, 90], head_size_cm ∈ [15, 20]
- **Perceptual:** 3D spatial positioning, works best on headphones
- **Numba:** Yes

## O004 — Distance Simulation
```
Near → far:
  amplitude = 1 / distance
  lowpass cutoff decreases with distance (air absorption)
  wet reverb amount increases with distance
  direct/reverb ratio shifts
```
- **Params:** distance ∈ [0.5, 100] meters
- **Perceptual:** Sound sources placed at varying distances
- **Numba:** Yes (combine filter + gain + reverb mix)

---

# P. ENVELOPE & DYNAMICS EFFECTS

## P001 — Envelope Reshaping
```
Extract envelope: env = lowpass(|x|, 20Hz)
Normalize signal: carrier = x / (env + eps)
Apply new envelope: y = carrier * new_env(t)
New envelope shapes: ramp up, ramp down, triangle, pulse, ADSR
```
- **Params:** new_shape ∈ {ramp_up, ramp_down, triangle, pulse, adsr}, shape_params
- **Perceptual:** Same spectral content with completely different amplitude contour
- **Numba:** Yes

## P002 — Envelope Inversion
```
Extract envelope, apply inverse:
y = x * (1 / (env + eps)) — loud parts become quiet, quiet parts become loud
With gain control to prevent explosion
```
- **Params:** smoothing_ms ∈ [5, 50], max_gain ∈ [10, 100]
- **Perceptual:** Dynamic inversion — ghostly, sustained, everything at same level
- **Numba:** Yes

## P003 — Noise Gate with Decay
```
When signal drops below threshold, instead of hard mute:
Apply exponential decay to the gated segments
Sounds like notes that ring out naturally then fade
```
- **Params:** threshold_db ∈ [-50, -20], decay_ms ∈ [50, 500], hysteresis_db ∈ [2, 6]
- **Perceptual:** Clean gating with musical decay tails
- **Numba:** Yes

## P004 — Rhythmic Gain Sequencer
```
16 or 32 step gain pattern, each step = gain value 0–1
Step through at tempo-synced rate
y[n] = x[n] * pattern[step]
```
- **Params:** pattern (list of gains), step_ms or bpm, pattern_length ∈ [4, 32]
- **Perceptual:** Rhythmic volume pattern, trance gate
- **Numba:** Yes

## P005 — Live Buffer Freeze / Stutter Hold
```
Capture a clean loop at trigger point:
1. Crossfade from original into loop
2. Hold the loop for hold_ms duration
3. Crossfade back to original
Loop boundaries are crossfaded for seamless repetition.
Distinct from granular freeze (I002): clean loop, no granular texture.
```
- **Params:** freeze_pos ∈ [0, 1], loop_ms ∈ [30, 200], fade_ms ∈ [5, 50], hold_ms ∈ [300, 2000]
- **Perceptual:** Smooth sustained loop of a single moment — DJ-style freeze
- **Numba:** Yes

---

# Q. COMBINATION & META-EFFECTS

## Q001 — Serial Chain (2 random effects)
```
Pick two effects from this catalog with random params
y = effect_B(effect_A(x))
```
- **Params:** effect_A_id, effect_A_params, effect_B_id, effect_B_params

## Q002 — Serial Chain (3 random effects)
```
y = effect_C(effect_B(effect_A(x)))
```
- **Params:** Three effect IDs and param sets

## Q003 — Parallel Mix (2 effects)
```
y = mix_a * effect_A(x) + mix_b * effect_B(x)
```
- **Params:** Two effect IDs and params, mix_a, mix_b

## Q004 — Wet/Dry Crossfade Over Time
```
Start fully dry, end fully wet (or vice versa):
y[n] = (1-alpha(n)) * x[n] + alpha(n) * effect(x)[n]
alpha sweeps 0→1 over duration
```
- **Params:** effect_id, effect_params, direction ∈ {dry_to_wet, wet_to_dry}

## Q005 — Feedback Through Effect
```
y[n] = x[n] + feedback * effect(y[n-block_size])
Process in blocks, feed output back through effect
```
- **Params:** effect_id, effect_params, block_size_ms ∈ [10, 100], feedback ∈ [0.1, 0.8]
- **Perceptual:** Emergent behavior from feedback — can be wild

---

# R. MISCELLANEOUS NOVEL

## R001 — Audio Fractalization
```
Replace each sample with a scaled copy of the entire signal:
For N iterations: at each scale level, the full signal is embedded
Output length grows — truncate or overlap-add
```
- **Params:** num_scales ∈ [2, 5], scale_ratio ∈ [0.3, 0.7], amplitude_per_scale
- **Perceptual:** Self-similar structure at multiple time scales
- **Numba:** Yes

## R002 — Spectral Peak Tracking + Resynthesis
```
Find N strongest spectral peaks per frame (frequency + magnitude)
Resynthesize using only those peaks (additive synthesis)
Optional: modify frequencies, add vibrato to each partial
```
- **Params:** num_peaks ∈ [5, 50], vibrato_depth, vibrato_rate
- **Perceptual:** Reduced, clarified timbre — only dominant partials remain
- **NumPy:** STFT + peak finding + oscillator bank

## R003 — Autoregressive Model Resynthesis
```
Fit LPC model to input: x[n] = Σ a_k * x[n-k] + e[n]
Modify LPC coefficients (scale, rotate, permute)
Drive with original residual (or noise) to resynthesize
```
- **Params:** lpc_order ∈ [10, 50], coefficient_modification ∈ {scale, rotate, jitter}
- **Perceptual:** Modified vocal/timbral character while preserving excitation
- **NumPy:** LPC analysis (Levinson-Durbin)

## R004 — Spectral Painting
```
Create an arbitrary spectral shape (drawn curve, mathematical function)
For each frame, multiply magnitudes by this shape
Shape can evolve over time (different function per frame)
```
- **Params:** shape_type ∈ {sine_wave, gaussian_peaks, sawtooth, random_curve}, shape_params, evolution_rate
- **Perceptual:** Sculptured spectrum, forced into arbitrary shapes
- **NumPy:** STFT

## R005 — Phase Gradient Manipulation
```
For each STFT frame:
  Compute phase gradient (derivative across bins)
  Multiply gradient by factor → affects group delay
  Integrate back to get modified phases
```
- **Params:** gradient_scale ∈ [0.1, 5.0]
- **Perceptual:** Temporal smearing or sharpening without changing magnitudes
- **NumPy:** STFT

## R006 — Spectral Entropy Gate
```
For each frame, compute spectral entropy: H = -Σ p_k * log(p_k)
where p_k = |X[k]|² / Σ|X[k]|²
Gate frames based on entropy: high entropy = noise-like, low = tonal
Keep only tonal (or only noisy) frames
```
- **Params:** entropy_threshold, mode ∈ {keep_tonal, keep_noisy}
- **Perceptual:** Separate tonal and noise-like content by information content
- **NumPy:** STFT

## R007 — Wavelet Decomposition + Manipulation
```
Multi-level wavelet decomposition (Haar or Daubechies)
Modify detail coefficients at each level:
  - Zero out (denoise)
  - Amplify (enhance transients at that scale)
  - Permute (shuffle detail)
  - Threshold (sparse representation)
Reconstruct
```
- **Params:** wavelet_type ∈ {haar, db4, db8}, num_levels ∈ [3, 8], modification per level
- **Perceptual:** Scale-specific manipulation — different time resolutions affected independently
- **NumPy:** Wavelet transform via convolution/decimation

## R008 — Hilbert Envelope + Fine Structure Swap
```
Analytic signal: z = x + j*hilbert(x)
Envelope = |z|, fine structure = cos(angle(z))
Swap: apply envelope of signal A to fine structure of signal B (or synthetic)
```
- **Params:** fine_structure_source ∈ {original, noise, sine, chirp}, envelope_modification
- **Perceptual:** Keeps rhythm/dynamics but changes tonal content (or vice versa)
- **NumPy:** Hilbert transform

## R009 — Spectral Freeze with Drift
```
Freeze spectrum at one point, then slowly drift:
  mag_frozen = mag at freeze point
  For each subsequent frame: mag = mag_frozen + drift_rate * (mag_current - mag_frozen)
Very slow drift = near-frozen. Fast drift = follows input loosely
```
- **Params:** freeze_point, drift_rate ∈ [0.001, 0.1]
- **Perceptual:** Semi-frozen sustain that slowly evolves
- **NumPy:** STFT

## R010 — Sample-Level Markov Chain
```
Quantize input to N levels
Build transition probability matrix from input
Generate new audio by walking the Markov chain
Seed with a chunk of original audio
```
- **Params:** num_levels ∈ [16, 256], order ∈ [1, 3] (higher order = more context)
- **Perceptual:** Statistical reconstruction — preserves local character, loses global structure
- **Numba:** Yes

## R011 — Frequency Domain Convolution (Spectral Multiply)
```
Multiply FFT magnitudes of input with magnitudes of a synthetic signal
Synthetic: sawtooth spectrum, harmonic series, noise spectrum, formant shape
Unlike regular convolution — this is pointwise multiply in frequency domain
```
- **Params:** synthetic_type, synthetic_params
- **Perceptual:** Spectral imprinting — forces harmonic structure of synthetic onto input
- **NumPy:** FFT

## R012 — Audio Quine (Convolve Subsegment with Whole)
```
Take a short chunk of the input (50–200ms)
Convolve the entire input with that chunk as IR
Creates self-referential resonance
```
- **Params:** chunk_start_ms, chunk_length_ms ∈ [50, 200]
- **Perceptual:** Input filtered by its own content — unique self-referential color
- **NumPy:** FFT convolution

## R013 — Spectral Phase Vocoder with Modified Hop
```
Standard phase vocoder but with non-standard analysis/synthesis hop ratios
Analysis hop ≠ synthesis hop creates time stretch
But also modify phase advancement rate for pitch effects
```
- **Params:** analysis_hop, synthesis_hop, phase_advance_rate
- **Perceptual:** Combined time and pitch manipulation with vocoder artifacts
- **NumPy:** STFT

## R014 — Karplus-Strong Cloud
```
Multiple KS resonators (10–30) tuned to random frequencies
Feed input into all simultaneously
Sum outputs
```
- **Params:** num_strings ∈ [10, 30], freq_range ∈ [50, 2000], decay_range
- **Perceptual:** Input excites a cluster of sympathetic strings — harp/piano-like resonance
- **Numba:** Yes

## R015 — Feedback FM Synthesis Effect
```
y[n] = x[n] + fb * sin(2π * carrier_freq * n / sr + mod_index * y[n-1])
Feedback FM creates chaotic spectra for high mod_index
```
- **Params:** carrier_freq ∈ [50, 2000], mod_index ∈ [0.5, 10], feedback ∈ [0.1, 0.9]
- **Perceptual:** Complex, evolving, metallic to chaotic depending on modulation index
- **Numba:** Yes

## R016 — Feedback AM Synthesis Effect
```
y[n] = x[n] * sin(2π * carrier * n/sr + fb * y[n-1])
AM multiplies carrier with input (creates sum/difference sidebands).
Feedback makes the carrier phase evolve chaotically.
Distinct from R015 (FM): AM preserves input envelope, FM adds pitched tones.
```
- **Params:** carrier_freq ∈ [20, 2000], feedback ∈ [0, 0.95], depth ∈ [0.1, 1.0]
- **Perceptual:** Metallic sidebands with chaotic evolution from feedback
- **Numba:** Yes

---

# VARIANT GENERATION STRATEGY

To generate ~1000 perceptually different outputs from 1 input:

1. **Single effects with varied params (500+ variants):**
   - For each of the ~120 effects above, generate 3–8 parameter sets
   - Sample params using latin hypercube sampling for coverage
   - Log-uniform for frequencies, linear for mix amounts

2. **Two-effect chains (300+ variants):**
   - Randomly pair effects from different categories
   - Use Q001 template
   - Moderate params (don't stack extreme settings)

3. **Three-effect chains (100+ variants):**
   - Use Q002 template
   - Ensure category diversity (e.g., distortion → spectral → delay, not three delays)

4. **Parallel combinations (100+ variants):**
   - Use Q003 template
   - Mix contrasting effects (e.g., frozen spectral + granular scatter)

5. **Perceptual deduplication:**
   - Compute MFCC features for each output
   - Compute spectral centroid, spectral spread, zero-crossing rate, RMS envelope
   - If cosine similarity of feature vectors > 0.95, discard the duplicate

6. **Harshness filter (post-chain safety):**
   - Reject if peak frequency energy above 14kHz exceeds -6dB (harsh aliasing)
   - Reject if crest factor > 20dB (extreme transients)
   - Apply soft limiter + lowpass to all others

## Output format:
```
output/
  manifest.csv     (columns: filename, effect_chain, params_json, category, novelty_tier)
  001_simple_delay_350ms_fb06.wav
  002_lorenz_ringmod_r28.wav
  ...
  1000_chain_paulstretch_x8_to_spectral_freeze.wav
```
