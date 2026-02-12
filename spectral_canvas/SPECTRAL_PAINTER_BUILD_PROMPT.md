# SPECTRAL PAINTER — Claude Code Build Prompt

## What You Are Building

A **spectral audio painting application** in Python + Tkinter where users draw on a 2D canvas (X = time, Y = frequency, brightness = amplitude) and hear the result as music. The same drawing is interpretable through **multiple synthesis engines**, each producing radically different sound from identical visual input. Users can also **import WAV files** as editable spectrograms and apply drawn spectral effects.

Target aesthetic: experimental electronic music (Four Tet, Aphex Twin, Autechre). Target user: musician/creative, not an engineer.

---

## Technology Stack (Mandatory)

| Component | Technology | Why |
|-----------|-----------|-----|
| GUI | **Tkinter** (stdlib) | No pip install for GUI, cross-platform, Canvas widget is perfect |
| Canvas rendering | **Pillow (PIL)** | Fast image manipulation backing the Tkinter canvas |
| DSP / math | **NumPy + SciPy** | FFT, STFT/ISTFT, windowing, signal generation |
| DSP acceleration | **Numba** (`@njit`) | JIT-compile hot loops (Karplus-Strong, grain scheduling, oscillator banks) |
| Audio I/O | **sounddevice** | PortAudio wrapper, NumPy-native, callback API for streaming |
| WAV files | **soundfile** | Read/write WAV, FLAC, OGG |
| Pitch utilities | **librosa** | `griffinlim`, `stft/istft`, `midi_to_hz`, `hz_to_midi`, `note_to_hz` |
| Phase reconstruction | **librosa.griffinlim** | Magnitude-only → audio when no phase exists |

```bash
pip install numpy scipy numba sounddevice soundfile librosa Pillow
```

**Do NOT use**: PyQt, Pygame, matplotlib for the GUI. Tkinter only.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        SpectralPainter                          │
│                     (Tkinter main window)                       │
├─────────────────┬───────────────────┬───────────────────────────┤
│   CanvasPanel   │  ControlPanel     │   TransportPanel          │
│   (tk.Canvas +  │  (engine select,  │   (play, stop, export,    │
│    PIL backing)  │   brush controls, │    duration, timeline)    │
│                 │   scale/tuning)   │                           │
└────────┬────────┴────────┬──────────┴──────────┬────────────────┘
         │                 │                      │
         ▼                 ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                     SpectrogramModel                            │
│  - self.magnitude: np.ndarray (H, W) float32, values 0.0–1.0   │
│  - self.phase: np.ndarray (H, W) float32 or None               │
│  - self.sr: int = 44100                                         │
│  - self.n_fft: int = 2048                                       │
│  - self.hop_length: int = 512                                   │
│  - self.duration: float (seconds, derived from W)               │
│  - H = n_fft // 2 + 1 = 1025 frequency bins                    │
│  - W = number of time frames                                    │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Engine (ABC)                                 │
│  def render(magnitude, phase, sr, n_fft, hop) -> np.ndarray     │
│                                                                  │
│  Subclasses:                                                     │
│    RandomPhaseEngine                                             │
│    AdditiveEngine                                                │
│    SubtractiveEngine                                             │
│    KarplusStrongEngine                                           │
│    GriffinLimEngine                                              │
│    GranularEngine                                                │
│    FMEngine                                                      │
│    WavetableEngine                                               │
│    FormantEngine                                                 │
│    PhaseVocoderEngine                                            │
│    SpectralFilterEngine (for imported audio effects mode)        │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     AudioPlayer                                  │
│  - Pre-renders audio via selected engine                         │
│  - Plays via sounddevice.play(audio, sr)                         │
│  - Exports via soundfile.write()                                 │
│  - Shows playback cursor position on canvas                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Canvas Specification

### Dimensions and Mapping

- **Canvas pixel size**: 1200 × 600 pixels (user-resizable)
- **Internal spectrogram**: shape `(1025, N_frames)` where 1025 = n_fft/2 + 1 frequency bins
- Canvas Y-axis maps to frequency via **logarithmic scale**:
  - Top of canvas = `max_freq` (default 20,000 Hz)
  - Bottom of canvas = `min_freq` (default 20 Hz)
  - Conversion: `freq = 2^(log2(max_freq) - (y_pixel / height) * (log2(max_freq) - log2(min_freq)))`
  - Inverse: `y_pixel = height * (log2(max_freq) - log2(freq)) / (log2(max_freq) - log2(min_freq))`
- Canvas X-axis maps linearly to time frames
- **Brightness** (0.0 = black/silent, 1.0 = white/full amplitude) maps to spectrogram magnitude
- Display the canvas with a **color map** — use a perceptually uniform colormap (viridis, magma, or inferno palette) rather than plain grayscale. Map 0.0 → dark/black, 1.0 → bright/hot. Store the colormap as a 256-entry RGB lookup table and apply it when converting the float32 backing array to the PIL display image.

### Drawing Tools

Implement these brush modes:

1. **Draw** (default): Gaussian soft brush adds brightness
   - Kernel: `K(x,y) = exp(-(x² + y²) / (2 * (radius * softness)²))`
   - Apply as `magnitude[region] = np.clip(magnitude[region] + K * brush_intensity, 0, 1)`
2. **Erase**: Same kernel, subtracts brightness
   - `magnitude[region] = np.clip(magnitude[region] - K * brush_intensity, 0, 1)`
3. **Line**: Click start + end, draw a Gaussian-brushed straight line between them (Bresenham with brush stamp at each point)
4. **Harmonic brush**: Drawing at frequency f₀ automatically fills in harmonics at 2f₀, 3f₀, 4f₀... with amplitudes A_n = intensity / n^α (α adjustable, default 1.0)
5. **Rectangle select + fill**: Drag to select region, fill with constant or gradient brightness

### Brush Controls (in ControlPanel)

- **Radius**: 1–50 pixels (slider)
- **Softness**: 0.1–2.0 (slider, controls Gaussian sigma relative to radius)
- **Intensity**: 0.0–1.0 (slider)
- **Harmonic rolloff** (α): 0.5–3.0 (slider, only visible in harmonic brush mode)
- **Number of harmonics**: 1–16 (slider, only visible in harmonic brush mode)

### Mouse Interaction

- **Left-click drag**: Paint with current brush
- **Right-click drag**: Erase (always, regardless of brush mode)
- **Mouse wheel**: Zoom brush radius
- **Shift+click**: Straight line from last click point
- **Ctrl+Z**: Undo (keep a stack of magnitude array snapshots, max 20)
- **Ctrl+Shift+Z**: Redo

### Frequency/Pitch Display

- Show a **frequency ruler** on the left edge with musical note labels (C2, C3, C4, etc.) at their logarithmic positions
- Show a **time ruler** on the top edge with seconds/beats
- On mouse hover, display current `(frequency Hz, note name, time)` in a status bar at bottom
- Use `librosa.hz_to_note()` for note name display

### Scale Snapping (Toggle On/Off)

When enabled, quantize the Y-coordinate of brush strokes to the nearest note in the selected scale:
- Scales: Chromatic (no snapping), Major, Minor, Pentatonic, Blues, Dorian, Mixolydian, Whole Tone, Harmonic Minor
- Root note selector (C through B)
- Snap function: convert pixel Y → freq → MIDI → nearest scale degree → freq → pixel Y
- Visual: draw faint horizontal guide lines on canvas at all valid scale pitches

---

## Synthesis Engines (Implement All)

Each engine is a class with a `render(magnitude, phase, sr, n_fft, hop_length) -> np.ndarray` method. The `phase` parameter is `None` for blank canvas, or a float32 array when imported audio provides original phase.

### Engine 1: Random Phase ISTFT ⭐ (implement first)

The simplest and most important engine. Instant ambient textures.

```
For each time frame column of magnitude:
  Assign random phase: φ ~ Uniform(0, 2π) for each bin
  Construct complex spectrum: S = magnitude * exp(j * φ)
  ISTFT with overlap-add using Hann window
```

Use `scipy.signal.istft` directly:
- Construct complex STFT: `Zxx = magnitude * np.exp(1j * np.random.uniform(0, 2*np.pi, magnitude.shape))`
- Invert: `_, audio = scipy.signal.istft(Zxx, fs=sr, window='hann', nperseg=n_fft, noverlap=n_fft - hop_length)`
- Normalize: `audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.9`

**Sound character**: Dreamy, smeared, shimmering pad. All transients dissolve.

### Engine 2: Additive Synthesis via IFFT

Each column is treated as an FFT magnitude frame. Assign random phase per frame, IFFT with overlap-add:

```
output = np.zeros(total_samples)
window = scipy.signal.windows.hann(n_fft, sym=False)
for i, col in enumerate(magnitude.T):
    spectrum = col * np.exp(1j * np.random.uniform(0, 2*np.pi, len(col)))
    frame = np.fft.irfft(spectrum, n=n_fft) * window
    start = i * hop_length
    output[start:start + n_fft] += frame
```

**Sound character**: Clean, organ-like, precise. Faithful to drawing but clinical.

### Engine 3: Subtractive Synthesis (Spectral Filtering)

Generate a harmonically rich source signal (sawtooth wave at a configurable base frequency, or white noise), compute its STFT, multiply by the drawn magnitude as a filter mask, ISTFT back:

```
source = generate_sawtooth(duration, base_freq=55.0, sr=sr)  # or white noise
_, _, Zxx_source = scipy.signal.stft(source, fs=sr, window='hann', nperseg=n_fft, noverlap=n_fft-hop_length)
# Resize magnitude to match Zxx_source shape if needed
Zxx_filtered = Zxx_source * magnitude_resized
_, audio = scipy.signal.istft(Zxx_filtered, fs=sr, window='hann', nperseg=n_fft, noverlap=n_fft-hop_length)
```

Provide a source selector dropdown: Sawtooth (55 Hz), Sawtooth (110 Hz), White Noise, Pink Noise, Imported Audio.

**Sound character**: Warm, thick, analog. The source waveform's DNA bleeds through.

### Engine 4: Karplus-Strong Physical Modeling

Interpret the spectrogram as a **note event map**: find connected bright regions, extract their center frequency (from Y position) and amplitude envelope (from brightness over time). For each detected note event, run KS synthesis:

**Use Numba for the inner loop:**

```python
@numba.njit
def karplus_strong(frequency, duration_samples, amplitude, sr, damping=0.996):
    period = int(sr / frequency)
    if period < 2:
        return np.zeros(duration_samples, dtype=np.float32)
    # Initialize delay line with noise burst
    delay_line = np.random.randn(period).astype(np.float32) * amplitude
    output = np.zeros(duration_samples, dtype=np.float32)
    for i in range(duration_samples):
        output[i] = delay_line[i % period]
        if i >= period:
            output[i] = damping * 0.5 * (output[i - period] + output[i - period + 1])
            delay_line[i % period] = output[i]
    return output
```

Note detection: threshold the magnitude at 0.1, find connected components via `scipy.ndimage.label`, extract bounding box, center frequency, and mean amplitude for each.

**Sound character**: Plucked strings, harps, marimbas. Strikingly organic.

### Engine 5: Griffin-Lim Reconstruction

Use librosa's optimized implementation:

```python
audio = librosa.griffinlim(
    magnitude * 80,  # scale up from 0-1 to reasonable dB range
    n_iter=60,
    hop_length=hop_length,
    win_length=n_fft,
    window='hann',
    momentum=0.99
)
```

**Sound character**: Metallic, robotic, ghostly. Artifacts are the aesthetic.

### Engine 6: Granular Synthesis

Interpret each bright pixel as a potential grain source. For each time frame:
- Scan frequency bins; where magnitude > threshold, spawn grains
- Each grain: frequency from bin center, amplitude from magnitude, duration 5–50 ms (density slider), Hann windowed
- Overlap grains with random jitter in time (±2 ms) for natural texture

**Use Numba for grain rendering.** Pre-allocate output buffer, iterate grains, use `np.sin` with phase accumulator.

**Sound character**: Pointillist, cloud-like, shimmering texture.

### Engine 7: FM Synthesis

Map spectral peaks to FM parameters:
- Detect peaks in each frame (local maxima in magnitude)
- Peak frequency → carrier frequency
- Peak magnitude → modulation index β (map 0–1 to 0–8)
- Fixed carrier:modulator ratio (default 1:1, configurable 1:2, 1:3, 2:3)
- `y(t) = A * sin(2π * f_c * t + β * sin(2π * f_m * t))`

Sum all active FM voices per frame with crossfading between frames.

**Sound character**: Metallic bells, glassy tones, DX7 electric piano.

### Engine 8: Wavetable Synthesis

Treat each time-frame column as a spectral snapshot:
- IFFT each column into a single-cycle waveform (stored in a wavetable)
- During playback, scan through the wavetable over time
- Interpolate between adjacent waveforms (linear or cubic)
- Play at a configurable base pitch (default: 220 Hz)

**Sound character**: Morphing, evolving, modern synth. Like Serum.

### Engine 9: Formant Synthesis

Detect 2–5 strongest peaks per frame as formant candidates:
- Smooth the magnitude spectrum (moving average, width ~300 Hz) to find the spectral envelope
- Find peaks of the smoothed envelope → formant frequencies
- Drive a bank of 2nd-order resonant bandpass filters with a pulse train or noise excitation
- Formant center frequencies from Y position of peaks, bandwidth ~100–200 Hz

**Sound character**: Vocal, choral, eerie singing. Drawing vowel shapes = hearing vowels.

### Engine 10: Phase Vocoder (Time-Stretch Mode)

This engine lets the user **stretch or compress time** while preserving pitch:
- Use the drawn spectrogram as target magnitudes
- Propagate phase using instantaneous frequency estimation:
  `φ[m+1,k] = φ[m,k] + 2π * k * hop_length / n_fft + Δφ_unwrapped`
- Configurable time stretch factor (0.25x to 4x)

Use `librosa.phase_vocoder` for the core algorithm.

**Sound character**: Crystalline for moderate stretch, watery/phasey at extremes.

### Engine 11: Spectral Filter (Effects Mode) — For Imported Audio

When audio is imported, this engine applies the **drawn canvas as a spectral effect**:
- The canvas painting acts as a time-frequency gain mask
- `Y[m,k] = X_imported[m,k] * canvas_magnitude[m,k]`
- Bright areas pass audio through; dark areas silence it
- Original phase is preserved

Additional effect sub-modes (selectable via dropdown):
- **Multiply** (filter): Y = X * G (default)
- **Add** (layer): Y = X + new_content (drawn content added as new frequencies with synthetic phase)
- **Spectral Freeze**: Hold one frame's magnitude with randomized phase → infinite sustain of a moment

**Sound character**: Depends on source audio. Drawing shapes the timbre in real time.

---

## Audio Import Pipeline

### Import WAV → Canvas

```python
def import_audio(filepath):
    y, sr = librosa.load(filepath, sr=44100, mono=True)
    # Compute STFT
    D = librosa.stft(y, n_fft=2048, hop_length=512, window='hann')
    magnitude = np.abs(D)         # shape: (1025, N_frames)
    phase = np.angle(D)           # preserve for re-synthesis
    # Convert to dB, normalize to 0-1 for display
    mag_db = librosa.amplitude_to_db(magnitude, ref=np.max)
    mag_normalized = (mag_db - mag_db.min()) / (mag_db.max() - mag_db.min() + 1e-8)
    return mag_normalized, phase, sr
```

Display the normalized magnitude on canvas. Store original phase separately. When re-rendering:
- For bins the user **hasn't modified**: use original phase (sounds best)
- For bins the user **has painted/erased**: use Griffin-Lim or random phase for those regions
- Track modifications with a boolean mask: `modified_mask[y, x] = True` wherever the user has drawn

### Export WAV

```python
def export_audio(audio, sr, filepath):
    # Normalize to prevent clipping
    audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.9
    soundfile.write(filepath, audio, sr, subtype='PCM_16')
```

---

## GUI Layout (Tkinter)

```
┌──────────────────────────────────────────────────────────────────────┐
│ File  Edit  View  Engine  Help                          [Menu Bar]   │
├──────────────────────────────────────────────────────────────────────┤
│ [Draw][Erase][Line][Harmonic][Select]  Radius:[===] Soft:[===]      │
│ Intensity:[===]  Scale:[Chromatic ▼] Root:[C ▼]  [Snap: ☐]         │
├────────┬─────────────────────────────────────────────────────────────┤
│  C7 ── │                                                            │
│  C6 ── │           SPECTROGRAM CANVAS                               │
│  C5 ── │              (1200 x 600)                                  │
│  C4 ── │                                                            │
│  C3 ── │          [colored spectrogram display]                     │
│  C2 ── │                                                            │
│  C1 ── │                                                            │
│ freq   │  0.0s          1.0s          2.0s          3.0s            │
│ ruler  │  ▲ playback cursor (vertical line)                         │
├────────┴─────────────────────────────────────────────────────────────┤
│ [▶ Play] [⏹ Stop] [💾 Export WAV] [📂 Import WAV]                   │
│ Engine: [Random Phase ▼]  Duration: [5.0]s  Source: [Sawtooth ▼]    │
├──────────────────────────────────────────────────────────────────────┤
│ Status: freq=440.0 Hz (A4) | time=1.23s | engine=Random Phase       │
└──────────────────────────────────────────────────────────────────────┘
```

### Key UI Details

- **Engine dropdown**: Lists all 11 engines by name. Changing engine does NOT re-render automatically — user clicks Play to hear the new engine's interpretation.
- **Duration spinner**: Sets the canvas duration in seconds (1–30). Changing this resizes the internal magnitude array (resamples existing content via `scipy.ndimage.zoom`).
- **Playback cursor**: A vertical line that sweeps across the canvas during playback, synchronized to `sounddevice` playback position.
- **Dark theme**: Use `bg='#1a1a2e'`, `fg='#e0e0e0'`, canvas background `'#0a0a1a'`. Accent color `'#e94560'` for active controls.
- **Keyboard shortcuts**:
  - Space = Play/Stop toggle
  - 1–9 = Select engine by number
  - B/E/L/H = Select brush mode (draw/erase/line/harmonic)
  - `[` / `]` = Decrease/increase brush radius
  - Ctrl+S = Export WAV
  - Ctrl+O = Import WAV
  - Ctrl+Z / Ctrl+Shift+Z = Undo/Redo
  - Ctrl+A = Select all
  - Delete = Clear selection (or entire canvas if no selection)

---

## Rendering Pipeline

When the user clicks **Play**:

1. Get current `magnitude` array from `SpectrogramModel`
2. Get current engine from dropdown
3. **In a background thread** (to keep GUI responsive):
   a. Call `engine.render(magnitude, phase, sr, n_fft, hop_length)` → `audio: np.ndarray`
   b. Normalize audio to [-0.9, 0.9]
   c. Signal main thread that audio is ready
4. Main thread: call `sounddevice.play(audio, sr)`
5. Start a Tkinter `after()` loop that queries `sounddevice.get_stream().time` and moves the playback cursor

**Threading**: Use `threading.Thread(target=render_fn, daemon=True)` + `queue.Queue` to communicate results back to the main thread. Check the queue in a Tkinter `after(50, check_queue)` poll loop.

**Do NOT use multiprocessing** — shared numpy arrays are simpler with threads + Numba releasing the GIL.

---

## Numba Acceleration Guide

Use `@numba.njit` (nopython mode) for:
- Karplus-Strong inner loop
- Grain rendering (overlap-add of thousands of tiny sine bursts)
- Gaussian brush kernel application (per-pixel loop on large canvases)
- FM synthesis voice rendering
- Additive synthesis oscillator bank (if not using IFFT approach)

**Pattern for Numba-accelerated brush:**

```python
@numba.njit
def apply_brush(magnitude, cx, cy, radius, softness, intensity, is_erase):
    h, w = magnitude.shape
    r = int(radius * 3)  # 3-sigma extent
    sigma = radius * softness
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            y = cy + dy
            x = cx + dx
            if 0 <= y < h and 0 <= x < w:
                k = math.exp(-(dx*dx + dy*dy) / (2 * sigma * sigma))
                if is_erase:
                    magnitude[y, x] = max(0.0, magnitude[y, x] - k * intensity)
                else:
                    magnitude[y, x] = min(1.0, magnitude[y, x] + k * intensity)
```

---

## File Structure

```
spectral_painter/
├── main.py                    # Entry point, creates SpectralPainterApp
├── model.py                   # SpectrogramModel (data layer)
├── canvas.py                  # CanvasPanel (Tkinter Canvas + PIL backing)
├── controls.py                # ControlPanel (brushes, sliders, dropdowns)
├── transport.py               # TransportPanel (play, stop, export, import)
├── player.py                  # AudioPlayer (sounddevice integration)
├── engines/
│   ├── __init__.py            # Engine ABC + registry
│   ├── random_phase.py
│   ├── additive.py
│   ├── subtractive.py
│   ├── karplus_strong.py
│   ├── griffin_lim.py
│   ├── granular.py
│   ├── fm.py
│   ├── wavetable.py
│   ├── formant.py
│   ├── phase_vocoder.py
│   └── spectral_filter.py
├── dsp/
│   ├── __init__.py
│   ├── brush.py               # Numba-accelerated brush kernels
│   ├── pitch.py               # Scale snapping, freq↔pixel conversion
│   ├── generators.py          # Sawtooth, noise, pulse train generators
│   └── utils.py               # Normalization, crossfade, window functions
└── requirements.txt
```

---

## Build Order (Phases)

### Phase 1: Core Canvas + Random Phase Engine (GET THIS WORKING FIRST)

Build and verify these in order:
1. `model.py` — SpectrogramModel with numpy magnitude array, resize, clear
2. `dsp/pitch.py` — Frequency ↔ pixel conversion (logarithmic)
3. `dsp/brush.py` — Gaussian brush kernel (Numba-accelerated)
4. `canvas.py` — Tkinter Canvas with PIL backing image, mouse draw, colormap display, frequency ruler, time ruler, hover coordinates
5. `engines/random_phase.py` — Random phase ISTFT render
6. `player.py` — sounddevice.play() wrapper with playback cursor callback
7. `transport.py` — Play, Stop, Export buttons
8. `main.py` — Wire everything together, dark theme

**Milestone**: User can draw on canvas and hear ambient textures. Export to WAV works.

### Phase 2: More Engines + Import

9. `engines/additive.py`
10. `engines/subtractive.py` + `dsp/generators.py` (sawtooth, noise)
11. `engines/karplus_strong.py` (Numba inner loop)
12. `engines/griffin_lim.py`
13. Audio import pipeline (WAV → STFT → canvas display with phase preservation)
14. `engines/spectral_filter.py` (multiply mode for imported audio)
15. Engine selector dropdown wired up

**Milestone**: 6 engines working, audio import/export, spectral filtering of imported audio.

### Phase 3: Musical Features

16. Scale snapping with visual guide lines
17. Harmonic brush tool
18. Undo/redo stack (snapshot-based)
19. Line drawing tool
20. Rectangle select + fill tool
21. `engines/granular.py`
22. `engines/fm.py`
23. `engines/wavetable.py`

**Milestone**: Full drawing toolkit, 9 engines, musical constraints.

### Phase 4: Polish

24. `engines/formant.py`
25. `engines/phase_vocoder.py`
26. Spectral freeze sub-mode
27. Keyboard shortcuts
28. Playback cursor synchronized to canvas
29. Background thread rendering with progress indication
30. Window resizing / canvas zoom

---

## Critical Implementation Notes

### Canvas ↔ Spectrogram Coordinate Mapping

The canvas displays a **resampled view** of the spectrogram. The internal spectrogram has shape `(1025, N_frames)` but the canvas might be `(600, 1200)` pixels. Drawing operations must:

1. Convert canvas (x_pixel, y_pixel) → spectrogram (time_frame, freq_bin)
2. Apply brush to the spectrogram array (not the display image)
3. Re-render the affected region of the display image from the spectrogram

**Y-axis is logarithmic**: pixel 0 (top) = 20,000 Hz, pixel 599 (bottom) = 20 Hz. The spectrogram's frequency bins are linearly spaced (bin k = k * sr / n_fft). So mapping requires:

```python
def pixel_y_to_freq(y_pixel, canvas_height, min_freq=20, max_freq=20000):
    ratio = y_pixel / canvas_height
    return 2 ** (np.log2(max_freq) - ratio * (np.log2(max_freq) - np.log2(min_freq)))

def freq_to_bin(freq, sr=44100, n_fft=2048):
    return int(round(freq * n_fft / sr))

def pixel_y_to_bin(y_pixel, canvas_height, sr=44100, n_fft=2048):
    freq = pixel_y_to_freq(y_pixel, canvas_height)
    return freq_to_bin(freq, sr, n_fft)
```

The brush must be applied in **spectrogram space** (linear frequency bins), not pixel space, to avoid distortion from the log mapping. When the user draws at pixel (x, y), compute the frequency, find the corresponding bin range (accounting for brush radius in Hz, not pixels), and apply the Gaussian kernel in bin space.

### Display Rendering

Convert float32 magnitude array → colormap → PIL Image → Tkinter PhotoImage:

```python
def magnitude_to_image(magnitude, colormap_lut):
    # magnitude: (H, W) float32 0-1, where H=1025, W=n_frames
    # Flip vertically (low freq at bottom)
    mag_flipped = magnitude[::-1, :]
    # Quantize to 0-255
    indices = (np.clip(mag_flipped, 0, 1) * 255).astype(np.uint8)
    # Apply colormap LUT (256, 3) uint8
    rgb = colormap_lut[indices]  # (H, W, 3) uint8
    # Resize to canvas dimensions
    img = Image.fromarray(rgb, 'RGB')
    img = img.resize((canvas_width, canvas_height), Image.NEAREST)
    return img
```

Use `ImageTk.PhotoImage(img)` to display. **Keep a reference** to the PhotoImage to prevent garbage collection.

Update only the dirty region after brush strokes for performance. Full redraw on resize/import.

### Audio Normalization

**Always normalize before playback and export**:
```python
def normalize(audio, target_peak=0.9):
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio * (target_peak / peak)
    return audio.astype(np.float32)
```

### STFT Parameters (Constants)

```python
SR = 44100
N_FFT = 2048
HOP_LENGTH = 512
N_FREQ_BINS = N_FFT // 2 + 1  # = 1025
WINDOW = 'hann'
```

These must be consistent across ALL engines and the import pipeline.

---

## What "Done" Looks Like

The application launches with a single `python main.py`. The user sees a dark-themed window with a black spectrogram canvas. They can:

1. **Draw** with a soft Gaussian brush — color appears using a viridis/magma colormap
2. **Select an engine** from a dropdown (all 11 available)
3. **Click Play** — hear their drawing as sound within 1–2 seconds
4. **Switch engines** — click Play again, same drawing sounds completely different
5. **Import a WAV** — see its spectrogram appear on canvas, edit it, re-render through any engine
6. **Export** — save the rendered audio as a WAV file
7. **Use musical helpers** — snap to scale, draw with harmonic brush
8. **See frequency/note labels** on the Y-axis, time on X-axis
9. **Watch a playback cursor** sweep across the canvas during playback
10. **Undo/redo** drawing actions with Ctrl+Z

The whole thing should feel like a **musical instrument** — immediate, responsive, visual, expressive. Not like an engineering tool.

---

## Common Pitfalls to Avoid

- **DO NOT** compute anything expensive on the main thread — rendering goes in a background thread
- **DO NOT** allocate arrays inside Numba `@njit` functions — pre-allocate and pass as parameters
- **DO NOT** forget to flip the Y-axis — frequency 0 (DC) is at the bottom of the spectrogram but low pixel indices are at the top of the canvas
- **DO NOT** use `plt.show()` or matplotlib anywhere — this is a Tkinter app
- **DO NOT** hardcode sample rate or FFT size — use the constants from the model
- **DO NOT** try real-time streaming synthesis — pre-render the full audio, then play it. Much simpler.
- **DO NOT** forget `root.mainloop()` or the window won't stay open
- **DO NOT** forget to keep references to PhotoImage objects — Tkinter garbage collects them otherwise
- **DO NOT** use `time.sleep()` in the main thread — use `root.after()` for timed callbacks
- **DO NOT** share numpy arrays across threads without care — the render thread should work on a **copy** of the magnitude array

---

## Final Notes

- Start with Phase 1. Get drawing + random phase + playback working end-to-end before adding more engines.
- Test each engine individually before integrating it into the UI.
- The Karplus-Strong engine needs Numba — test that `numba.njit` compiles successfully before building the full engine.
- The subtractive engine is trivially simple once random phase works (it's the same ISTFT but multiplied against a source STFT).
- For imported audio, the spectral filter engine is the most useful "effect" — prioritize it.
- Make the app look good. Dark theme, colormap display, clean layout. This is a creative tool.
