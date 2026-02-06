# ML-Exploration-Assisted FDN Reverb: Project Roadmap

## What We're Building

A custom reverb built from scratch, component by component. The architecture is an **8-node Feedback Delay Network (FDN)** with time-varying parameters at three modulation timescales. We use **ML techniques to explore the vast parameter space offline**, discover novel and interesting configurations, then deploy the best ones as a **real-time DSP plugin (VST)**. The parameter space is intentionally wide — including configurations that aren't reverbs at all (resonators, multi-tap delays, self-oscillating drones, metallic textures, silence, dry passthrough) — so that ML exploration can find genuinely novel sounds on the boundary between "interesting" and "garbage."

Everything is written from scratch. No DSP libraries. The programmer has no DSP background and will learn DSP concepts as needed, one primitive at a time, by building and hearing each piece.

---

## Core Design Principles

1. **Build from scratch, learn as we go.** Every DSP primitive is hand-written. Understanding comes from hearing what each piece does, not from reading textbooks.
2. **The GUI and ML use the same interface.** Both are just different ways of calling `render_fdn(params) → audio`. A slider and an optimizer are the same thing — a parameter source.
3. **Offline exploration, real-time deployment.** ML runs overnight in batch. The final plugin is pure DSP with no ML at runtime.
4. **Wide parameter ranges, including "broken."** We deliberately allow configurations that explode, ring, go silent, or sound terrible. The interesting territory is at the boundary.

---

## The Six DSP Primitives (Built in Order)

Everything in the project is assembled from these six building blocks. Each is small, self-contained, and should be understood completely before moving on.

### Primitive 1: Circular Buffer (Delay Line)
- An array with a write index that wraps around
- Read from an offset position to get delayed signal
- ~20 lines of Python
- **Test:** Feed in a clap/impulse, hear a single echo
- **DSP concept learned:** What a delay line is and does

### Primitive 2: Fractional Delay Interpolation
- When the read position falls between samples, interpolate
- Linear interpolation: 3 lines. Cubic (Lagrange3rd): ~10 lines
- Necessary for modulated (time-varying) delay times
- **Test:** Slowly modulate delay time by hand, hear pitch shift
- **DSP concept learned:** Why you need interpolation, what artifacts sound like without it

### Primitive 3: One-Pole Filter
- `y[n] = (1 - a) * x[n] + a * y[n-1]`
- One line of math. Lowpass when `a` is between 0 and 1
- This is the damping filter — makes high frequencies decay faster (simulates air absorption)
- **Test:** Put it in a feedback delay loop, hear repeats get progressively darker
- **DSP concept learned:** What a lowpass filter does, how frequency-dependent decay works

### Primitive 4: Biquad Filter
- 5 coefficients, 2 state variables, 2 lines of state update
- Same structure handles lowpass, highpass, bandpass, shelving — different coefficient formulas
- Used for more precise tonal shaping inside the FDN feedback paths
- **Test:** Sweep cutoff frequency on white noise, hear the filter shape
- **DSP concept learned:** Second-order filters, resonance, Q factor

### Primitive 5: Allpass Filter
- Delay line with a specific feedback/feedforward structure (~5 lines)
- Passes all frequencies at equal amplitude, but smears their phase/timing
- This is what turns discrete echoes into smooth, diffuse wash
- **Test:** Feed impulse through chain of allpasses, hear transient smear into cloud
- **DSP concept learned:** Phase vs. amplitude, diffusion

### Primitive 6: Matrix-Vector Multiply (Feedback Matrix)
- 8x8 matrix times 8-element vector of delay line outputs
- Householder matrix: `A = I - (2/N) * ones * ones^T` — uniform coupling, O(N) computation
- Controls how energy flows between delay lines — this IS the reverb topology
- **Test:** Compare diagonal matrix (isolated comb filters, metallic ringing) vs. Householder (smooth, coupled, lush)
- **DSP concept learned:** How inter-channel mixing creates density, why topology matters

---

## Phase 1: Static FDN in Python (Learn the Fundamentals)

**Goal:** A working 8-node FDN reverb that processes audio offline (read WAV → process → write WAV). No real-time, no GUI, no ML yet.

### Architecture
```
Input → [Pre-delay] → [Input Diffusion (allpass chain)] → FDN Loop → [Wet/Dry Mix] → Output

FDN Loop (per sample):
  1. Read from 8 delay lines
  2. Apply per-node one-pole damping filter to each read value
  3. Multiply 8-vector by 8x8 Householder feedback matrix
  4. Scale by global feedback gain
  5. Add input signal (distributed to nodes via input gains vector)
  6. Write result back to delay lines
  7. Sum weighted outputs from all nodes (output gains vector)
```

### Parameters (Static FDN — ~30 total)
- 8 delay times (0.1ms to 500ms each — spanning resonators through large halls)
- 8 damping filter coefficients (per-node high-frequency decay rate)
- Global feedback gain (0.0 to 1.0+, where >1.0 is unstable/explosive)
- Input gains vector (8 values — how input distributes to nodes)
- Output gains vector (8 values — which nodes contribute to output and how much)
- Pre-delay time (0 to 250ms)
- Input diffusion amount (allpass chain coefficient, 0 to ~0.7)
- Wet/dry mix (0.0 to 1.0)

### Steps
1. Build and test each primitive in isolation (Primitives 1-6 above)
2. Wire them into the FDN loop
3. Process a clap/impulse → examine and listen to the impulse response
4. Process music/speech → listen for quality, ringing, coloration
5. Hand-tune parameters to get a "decent room" sound — this teaches what each parameter does perceptually
6. Verify edge cases: feedback=0 (multi-tap delay), all damping=1 (no filtering), short delays (<10ms, enters resonator territory), feedback >1.0 (controlled explosion with limiting)

### Deliverables
- `delay_line.py` — circular buffer with fractional read
- `filters.py` — one-pole, biquad, allpass
- `fdn.py` — the 8-node FDN engine, function signature: `process_fdn(input_audio, params) → output_audio`
- `render.py` — script that loads WAV, processes through FDN, saves output WAV
- A handful of parameter presets that sound good, found by hand

---

## Phase 2: Real-Time Audio + GUI (Hear It Live)

**Goal:** Hook the FDN to a microphone, hear reverb in real time. Build a GUI with sliders that control every parameter. This GUI calls the exact same `process_fdn(params)` function that ML will later use.

### Real-Time Audio Setup
- Use `sounddevice` (PortAudio wrapper) for mic input → speaker output
- Callback-based: `callback(indata, outdata, frames, time, status)`
- Start with blocksize=1024 (~23ms latency) for pure Python
- Later optimize with Numba `@jit` or a small C inner loop via `ctypes` to get down to blocksize=128 (~3ms)

### Performance Path
```
Step 1: Pure Python, blocksize=1024-2048 (~23-46ms latency)
        → Good enough for exploration and learning
        
Step 2: Numba @jit on the inner FDN loop
        → Approaches C speed, blocksize=256 (~6ms)
        
Step 3: Small C function for the per-sample FDN loop, called via ctypes
        → Native speed, blocksize=128 (~3ms), indistinguishable from zero latency
```

### GUI (Python, tkinter or PyQt)
- Sliders for every parameter (30+ sliders, organized by category)
- XY pad (2D) — maps (x, y) position to a parameter vector via a mapping function
  - Initially: manually assigned axes (e.g., x=size, y=brightness)
  - Later: axes mapped to learned latent space from VAE (Phase 4)
- Real-time waveform/spectrum display (nice to have, not essential)
- Preset save/load (JSON files containing parameter vectors)
- **Critical:** The GUI writes to a shared `params` dict. The audio callback reads from it. Same interface the ML optimizer will use.

### The Shared Interface
```python
# This is the contract. GUI, ML, and manual scripting all use this.
params = {
    "delay_times": [float] * 8,       # in samples
    "damping_coeffs": [float] * 8,    # 0.0 to 1.0
    "feedback_gain": float,            # 0.0 to ~1.05
    "input_gains": [float] * 8,
    "output_gains": [float] * 8,
    "pre_delay": float,                # in samples
    "diffusion": float,                # 0.0 to 0.7
    "wet_dry": float,                  # 0.0 to 1.0
    # ... modulation params added in Phase 3
}

def render_fdn(input_audio: np.ndarray, params: dict) -> np.ndarray:
    """The single entry point. GUI sliders, ML optimizers, and batch
    rendering all call this same function."""
    ...
```

### Deliverables
- `realtime.py` — mic → FDN → speakers via sounddevice callback
- `gui.py` — parameter control GUI with sliders and XY pad
- `presets.py` — save/load parameter dicts as JSON
- Ability to twiddle knobs and hear changes in real time through a mic

---

## Phase 3: Time-Varying Parameters (Dynamic FDN)

**Goal:** Add modulation — parameters that change over time. This transforms the FDN from a static system to a living, breathing one. The parameter space explodes, creating the vast territory for ML to explore.

### Three Timescales of Modulation

**Slow (0.01–0.5 Hz):** The reverb character evolves over seconds. Delay times drift, filter cutoffs shift, the feedback matrix blends between configurations. Perceptually: the "space" breathes, changes shape. This is where preset morphing lives.

**Medium/LFO (0.5–20 Hz):** The standard modulation range. Delay time modulation at ~2 Hz with ±4–7 samples depth eliminates metallic ringing (fills in comb filter notches). Filter cutoff modulation creates a breathing, vocal quality. This is the difference between "cheap reverb" and "expensive reverb."

**Fast/Audio-rate (20 Hz+):** The novel territory. Delay time modulation at audio rates creates FM-like sidebands — the reverb tail acquires inharmonic spectral content. Filter modulation creates ring-modulation effects. Audio-rate feedback matrix modulation (topology itself vibrating) is essentially unexplored and potentially very interesting.

### New Parameters Per Modulated Parameter
Each modulatable parameter gets:
- `mod_depth` — how much it moves (0 = static, effectively disabling modulation)
- `mod_rate` — how fast (0.01 Hz to 1000+ Hz, spanning all three timescales)
- `mod_waveform` — sine, triangle, sample-and-hold (random), envelope-follower
- `mod_phase` — relative to other modulators (for correlated vs. independent movement)

### What Gets Modulated
- **Delay times** (most impactful — even tiny modulation transforms character)
- **Damping filter coefficients** (tonal character of decay evolves over time)
- **Feedback matrix coefficients** (topology breathes — signals rerouted through network)
- **Output tap gains** (spatial image shifts, perceived size pulses)

### Structured Modulation (Reduce Parameter Count)
Instead of fully independent modulation per node (which would add 8×4=32 params per modulated parameter type), use a hierarchical structure:
- Global master rate (one clock)
- Per-node rate multiplier (integer ratios: 1x, 2x, 3x — creates rhythmic relationships)
- Global depth, with per-node depth scaling
- Correlation parameter (all nodes in sync vs. fully independent)

### Implementation
```python
# In the per-sample FDN loop:
for each sample n:
    for each node i:
        phase = (mod_rate[i] * n / sample_rate + mod_phase[i]) % 1.0
        lfo_value = waveform_func(phase, mod_waveform[i])
        current_delay[i] = base_delay[i] + mod_depth_delay[i] * lfo_value
        current_damping[i] = base_damping[i] + mod_depth_damping[i] * lfo_value
    # ... process FDN with current_* values
```

### What the Full Parameter Space Now Contains
With modulation enabled, the explorable space includes:
- Silence (wet=0 or all feedback=0)
- Dry passthrough (wet=0, dry=1)
- Single echo / multi-tap delay (feedback=0)
- Conventional room/hall/plate reverb (moderate static params, subtle LFO modulation)
- Lush, expensive-sounding reverbs (slow, subtle delay modulation)
- Tape-warped, lo-fi reverbs (medium modulation, imperfect character)
- Metallic resonators / pitched tones (very short delays <10ms)
- Self-oscillating drones (feedback >1.0 with limiting)
- FM-like alien textures (audio-rate delay modulation)
- Breathing, evolving soundscapes (slow topology modulation)
- Rhythmic, pulsing effects (synced modulation rates)
- Everything in between

### Deliverables
- Extended `fdn.py` with modulation system
- Extended `params` dict with modulation parameters
- Extended GUI with modulation controls
- Ability to hear all three timescales in real-time through mic
- Several hand-discovered interesting modulation configurations

---

## Phase 4: ML-Assisted Exploration (Find the Gold)

**Goal:** Use ML techniques to systematically explore the now-vast parameter space. Find configurations that sound interesting — especially in the novel territory that's hard to find by hand. Build a navigable map of the "good" region.

### The Exploration Loop
```
1. Choose a parameter vector (via ML strategy)
2. Call render_fdn(test_signal, params) → audio
3. Evaluate quality (perceptual features, human rating, or learned metric)
4. Update the ML model
5. Repeat
```

### Quality Evaluation (The Loss Function)

**Automated metrics (fast, for large-scale search):**
- RT60 (decay time) — in target range?
- Echo density over time — does it build to sufficient density?
- Spectral centroid of tail — does it evolve (darken) over time?
- Stability check — does it explode or DC-offset?
- Energy envelope shape — exponential decay vs. weird pulsing vs. infinite sustain

**Human-in-the-loop (slow, for refinement):**
- Listen to 200-500 rendered samples, rate on a scale of 1-5
- Train a small neural network to predict ratings from parameter vectors
- Use this surrogate model as a fast proxy for subsequent search

**Novelty bonus:**
- Reward configurations that are far (in parameter space) from known-good presets
- This pushes exploration toward the boundary regions where novel sounds live

### ML Search Strategies

**Strategy 1: Bayesian Optimization (efficient local refinement)**
- Gaussian Process models the quality function
- Acquisition function (Expected Improvement) chooses where to sample next
- Efficient: finds good settings in ~100-300 evaluations
- Use `scipy.optimize` or `optuna` (lightweight, pip-installable)
- Best for: refining a region you know is promising

**Strategy 2: CMA-ES (global exploration)**
- Covariance Matrix Adaptation Evolution Strategy
- Maintains a population, adapts search distribution
- Naturally maintains diversity — finds multiple good regions
- Use `pycma` (pip-installable, ~50KB)
- Best for: initial broad search, finding diverse configurations

**Strategy 3: VAE Latent Space Learning (navigation)**
- Collect 500-2000 rated parameter vectors from Strategies 1 & 2
- Train a Variational Autoencoder: encoder maps params→2D, decoder maps 2D→params
- Regularize with perceptual features (configurations that sound similar should be nearby)
- The 2D latent space becomes the XY pad in the GUI
- Write from scratch: encoder and decoder are small MLPs (~100 lines with NumPy, or use PyTorch if preferred for autograd)
- Best for: building the navigable map of reverb-space

### Workflow
1. Run CMA-ES overnight with automated metrics — generates ~5000 configurations, filters to ~500 "interesting" ones
2. Listen to the 500, rate a subset, train surrogate quality model
3. Run Bayesian optimization guided by surrogate model — refines toward your taste
4. Collect all good configurations, train VAE
5. Plug VAE's 2D latent space into GUI's XY pad
6. Navigate the learned space in real-time, discover configurations by wandering

### Deliverables
- `evaluate.py` — automated perceptual metrics for FDN output
- `search_cmaes.py` — CMA-ES exploration script
- `search_bayesian.py` — Bayesian optimization script
- `surrogate.py` — learned quality prediction model (trained on human ratings)
- `vae.py` — VAE for latent space learning
- `explorer_gui.py` — extended GUI with VAE-mapped XY pad
- A curated library of ML-discovered configurations (JSON preset files)
- A trained VAE model (weights file) defining the 2D reverb-space

---

## Phase 5: Real-Time VST Plugin (Ship It)

**Goal:** Port the FDN engine to C++ (JUCE) or Rust (nih-plug), wrap it as a VST3/AU plugin with a GUI that includes the latent-space XY pad. The ML scaffolding stays in Python — only the discovered configurations and the DSP engine ship.

### What Ships
- The FDN algorithm (identical to the Python version, just in C++/Rust)
- A set of curated presets (ML-discovered parameter vectors)
- The XY pad with pre-computed latent space coordinates for presets
- Interpolation between presets based on XY position (barycentric/natural neighbor)
- Standard knobs for direct parameter control

### What Doesn't Ship
- Python, NumPy, any ML framework
- The VAE itself (we export its output: a 2D → params lookup table or interpolation mesh)
- Any training code

### Framework Choice
- **JUCE (C++):** Industry standard, 20-year ecosystem, VST3/AU/AAX. More boilerplate, but maximum compatibility. Already has `dsp::DelayLine` as reference (though we write our own).
- **nih-plug (Rust):** Leaner, modern, ISC license, VST3/CLAP. Less boilerplate, safer memory model. Smaller ecosystem.

### Deliverables
- Working VST3/AU plugin
- GUI with XY pad, preset browser, per-parameter knobs
- Documentation of the signal flow and every parameter

---

## File/Folder Structure

```
reverb-project/
├── ROADMAP.md              ← this file
├── primitives/
│   ├── delay_line.py       ← Phase 1, Primitive 1-2
│   ├── filters.py          ← Phase 1, Primitive 3-5
│   └── matrix.py           ← Phase 1, Primitive 6
├── engine/
│   ├── fdn.py              ← Phase 1, the core FDN
│   ├── fdn_modulated.py    ← Phase 3, time-varying extension
│   └── params.py           ← parameter dict schema + defaults
├── audio/
│   ├── render.py           ← offline WAV processing
│   ├── realtime.py         ← mic → FDN → speakers
│   └── test_signals/       ← impulses, claps, music clips for testing
├── gui/
│   ├── gui.py              ← parameter control GUI
│   ├── xy_pad.py           ← 2D navigation widget
│   └── presets/            ← JSON preset files
├── ml/
│   ├── evaluate.py         ← perceptual quality metrics
│   ├── search_cmaes.py     ← CMA-ES exploration
│   ├── search_bayesian.py  ← Bayesian optimization
│   ├── surrogate.py        ← learned quality model
│   └── vae.py              ← latent space learning
├── optimize/
│   ├── numba_fdn.py        ← Numba-jitted FDN inner loop
│   └── c_fdn/              ← C inner loop + ctypes wrapper
└── plugin/                 ← Phase 5, JUCE or nih-plug project
```

---

## Key Technical References (For Reading, Not Importing)

These papers/resources explain the math behind what we're building. Read as needed, not upfront.

- **Jot & Chaigne (1991)** — Original FDN methodology (lossless prototype → add absorption)
- **Schlecht & Habets (2015)** — Time-varying feedback matrices (the theory behind Phase 3's topology modulation)
- **Tom Erbe, "Building the Erbe-Verb" (ICMC 2015)** — Audio-rate FDN modulation, the most creatively ambitious existing reverb design
- **Dal Santo et al., "Differentiable FDN for Colorless Reverberation" (DAFx 2023)** — How to make FDNs gradient-optimizable (relevant background for Phase 4)
- **Steinmetz et al., "ST-ITO" (ISMIR 2024)** — CMA-ES for audio effect parameter search (direct precedent for Phase 4's CMA-ES approach)
- **Dattorro, "Effect Design Part 1" (JAES 1997)** — The "Rosetta Stone" of reverb design, excellent for building intuition
- **Signalsmith Audio (ADC 2021)** — Practical Householder/Hadamard FDN implementation notes
- **Valhalla DSP blog** — Sean Costello's posts on reverb design philosophy and delay modulation

---

## What's Novel About This Project

Per the SOTA research: each component has academic precedent, but the full combination does not exist anywhere.

**Has been done:**
- 8-node FDN with Householder matrix and per-node damping (standard since Jot 1991)
- LFO-rate delay modulation in reverbs (standard since EMT 250, 1976)
- Bayesian optimization for reverb parameter matching (Bona et al. 2022)
- CMA-ES for audio effect parameter search (ST-ITO 2024)

**Has not been done:**
- ML-discovered multi-timescale modulation patterns (rate, depth, waveform, phase optimized across slow/medium/fast timescales by ML)
- VAE latent space of dynamic FDN configurations (including modulation envelopes, not just static parameters)
- XY-pad navigation of an ML-learned reverb space
- Systematic audio-rate FDN modulation exploration via optimization (Erbe did it manually; nobody has searched this space with ML)

---

## Getting Started: First Session

The very first thing to build is the delay line. Here's the exact sequence:

1. Create `primitives/delay_line.py`
2. Implement a circular buffer: write(sample), read(delay_in_samples) → sample
3. Add fractional delay read with linear interpolation
4. Write a test: feed in an impulse (single sample of 1.0, rest zeros), read back with various delays
5. Save the output as a WAV file and listen — you should hear a single echo
6. Add feedback: `output = read(delay); write(input + feedback * output)` — listen to repeating echoes
7. Experiment: very short delay (<5ms) with high feedback — hear it become a pitched tone (this is a comb filter, the most primitive resonator)

From there, each subsequent session adds one primitive or one architectural layer, always with immediate audio feedback.