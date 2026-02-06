# TODO — ML-Exploration-Assisted FDN Reverb

## Phase 1: Static FDN in Python
- [ ] **DSP Primitives**
  - [x] Primitive 1: Circular Buffer (Delay Line)
    - [x] Implement write(sample) / read(delay) with wrapping index
    - [x] Test with impulse — hear single echo
    - [x] Test with feedback loop — hear repeating echoes
    - [x] Test very short delay (<5ms) + high feedback — hear pitched tone (comb filter)
  - [x] Primitive 2: Fractional Delay Interpolation
    - [x] Linear interpolation for fractional read positions
    - [x] Cubic (Lagrange 3rd order) interpolation
    - [x] Test: slowly modulate delay time, hear pitch shift
  - [x] Primitive 3: One-Pole Filter
    - [x] Implement `y[n] = (1-a)*x[n] + a*y[n-1]`
    - [x] Test in a feedback delay loop — hear repeats darken over time
  - [x] Primitive 4: Biquad Filter
    - [x] Implement 5-coefficient / 2-state-variable structure
    - [x] Coefficient formulas for lowpass, highpass, bandpass, shelving
    - [x] Test: sweep cutoff on white noise
  - [x] Primitive 5: Allpass Filter
    - [x] Implement delay-based allpass (feedback/feedforward structure)
    - [x] Test: impulse through chain of allpasses — hear transient smear to cloud
  - [x] Primitive 6: Feedback Matrix (Matrix-Vector Multiply)
    - [x] Implement Householder matrix: `A = I - (2/N) * ones * ones^T`
    - [x] Test: compare diagonal (isolated comb, metallic) vs Householder (smooth, coupled)
- [x] **Wire FDN Engine**
  - [x] Implement 8-node FDN loop (read delays -> damping -> matrix -> feedback -> write)
  - [x] Input distribution via input gains vector
  - [x] Output summing via output gains vector
  - [x] Pre-delay stage
  - [x] Input diffusion (allpass chain)
  - [x] Wet/dry mix
- [x] **Offline Rendering**
  - [x] Load WAV -> process through FDN -> save output WAV
  - [x] Process impulse — examine and listen to impulse response
  - [x] Process music/speech — listen for quality, ringing, coloration
- [x] **Hand-Tuning & Edge Cases**
  - [x] Hand-tune a "decent room" preset
  - [x] Test feedback=0 (multi-tap delay)
  - [x] Test all damping=1 (no filtering)
  - [x] Test short delays <10ms (resonator territory)
  - [x] Test feedback >1.0 (controlled explosion with limiting)
  - [x] Save several hand-tuned presets as JSON
- [x] **Deliverables check**
  - [x] `primitives/delay_line.py`
  - [x] `primitives/filters.py`
  - [x] `primitives/matrix.py`
  - [x] `engine/fdn.py`
  - [x] `engine/params.py`
  - [x] `audio/render.py`

## Phase 2: Real-Time Audio + GUI
- [x] **Audio Playback**
  - [x] WAV file loading (source selection from test signals or file picker)
  - [x] Render source through FDN with current params
  - [x] Playback via sounddevice (play dry / play wet / stop)
- [x] **GUI**
  - [x] Sliders for all ~30 parameters (organized by tabs: Main, Delays, Gains)
  - [x] XY pad (2D) with assignable axes (e.g., x=feedback, y=damping)
  - [x] Preset save/load (JSON) with preset browser tab
  - [x] Matrix topology selector (dropdown)
- [x] **Performance Optimization**
  - [x] Numba `@jit` on inner FDN loop — target blocksize=256 (~6ms)
  - [ ] (Optional) C inner loop via ctypes — target blocksize=128 (~3ms)
- [ ] **Deliverables check**
  - [ ] `audio/realtime.py`
  - [ ] `gui/gui.py`
  - [ ] `gui/xy_pad.py`
  - [ ] `gui/presets/` directory with JSON presets

## Phase 2b: Audio Quality & Output Improvements
- [x] **DC Blocking Filter**
  - [x] Add one-pole high-pass (~5 Hz) in feedback path per node
  - [x] Prevents DC offset accumulation with saturation/high feedback
- [x] **Stereo Output**
  - [x] Pan 8 output taps across stereo field
  - [x] Return 2-channel array from render_fdn
  - [x] Add stereo width knob to GUI (0=mono, 1=full stereo)
  - [x] Update playback and WAV export for stereo

## Phase 3: Time-Varying Parameters (Dynamic FDN)
- [x] **Modulation System**
  - [x] LFO generator (sine, triangle, sample-and-hold)
  - [x] Per-parameter modulation: depth, rate, waveform, phase
  - [x] Structured modulation (global master rate, per-node multipliers, correlation param)
- [x] **Modulatable Parameters**
  - [x] Delay time modulation (most impactful)
  - [x] Damping coefficient modulation
  - [x] Feedback matrix coefficient modulation (blend between two matrices)
  - [x] Output tap gain modulation
- [x] **Three Timescales**
  - [x] Slow (0.01-0.5 Hz) — character evolves over seconds
  - [x] Medium/LFO (0.5-20 Hz) — eliminates metallic ringing, breathing quality
  - [x] Fast/audio-rate (20 Hz+) — FM-like sidebands, novel territory
- [x] **Integration**
  - [x] `engine/numba_fdn_mod.py` — modulated FDN engine with Numba JIT
  - [x] `engine/fdn.py` auto-routes to modulated path when modulation active
  - [x] Extend params dict with modulation parameters
  - [x] Extend GUI with modulation controls (scrollable params page)
  - [x] 3 modulation presets: lush_chorus_room, breathing_space, fm_alien_texture

## Phase 4: ML-Assisted Exploration
- [ ] **Quality Evaluation**
  - [ ] Automated metrics: RT60, echo density, spectral centroid, stability, energy envelope
  - [ ] Human rating pipeline: render samples, rate 1-5, collect dataset
  - [ ] Surrogate model: small NN trained on human ratings
  - [ ] Novelty bonus (distance from known-good presets)
- [ ] **Search Strategies**
  - [ ] CMA-ES broad exploration (pycma) — generate ~5000 configs, filter to ~500
  - [ ] Bayesian optimization (optuna/scipy) — refine promising regions
  - [ ] Listen to filtered results, rate a subset, train surrogate model
  - [ ] Run Bayesian optimization guided by surrogate
- [ ] **VAE Latent Space**
  - [ ] Collect 500-2000 rated parameter vectors
  - [ ] Train VAE: encoder (params -> 2D), decoder (2D -> params)
  - [ ] Regularize with perceptual features
  - [ ] Plug 2D latent space into GUI XY pad
- [ ] **Deliverables check**
  - [ ] `ml/evaluate.py`
  - [ ] `ml/search_cmaes.py`
  - [ ] `ml/search_bayesian.py`
  - [ ] `ml/surrogate.py`
  - [ ] `ml/vae.py`
  - [ ] Curated preset library (JSON)
  - [ ] Trained VAE weights file

## Phase 5: Real-Time VST Plugin
- [ ] **Choose framework** (JUCE C++ or nih-plug Rust)
- [ ] **Port FDN engine** to C++/Rust (identical algorithm, no Python/ML at runtime)
- [ ] **Export from ML pipeline**
  - [ ] Curated presets as parameter vectors
  - [ ] Pre-computed 2D latent space coordinates (lookup table / interpolation mesh)
- [ ] **Plugin GUI**
  - [ ] XY pad with latent-space navigation
  - [ ] Preset browser
  - [ ] Per-parameter knobs
- [ ] **Build & Package**
  - [ ] VST3 format
  - [ ] AU format (macOS)
  - [ ] Signal flow documentation


# extras
- [x] lock parameters so that randomize does not change them
- The gold-standard export

Format: WAV (PCM, not compressed)

Bit depth: 24-bit

Sample rate: 44.1 kHz or 48 kHz

Dither: Off (unless you’re going from higher bit depth down to 24-bit)

That combo gives you excellent dynamic range, low noise, and full compatibility with studios, plugins, and distributors.