# CLAUDE.md — Project Reference

## What This Is

An 8-node Feedback Delay Network (FDN) reverb built entirely from scratch (no DSP libraries). ML techniques explore the parameter space offline to discover novel sound configurations, which are then deployed as a real-time VST plugin. See `PLAN.md` for the full roadmap and `TODO.md` for current progress.

## Key Architecture

- **Single entry point:** `render_fdn(input_audio, params) -> output_audio` — GUI sliders, ML optimizers, and batch rendering all call this same function.
- **Params dict** is the shared contract between GUI, ML, and manual scripting. Defined in `engine/params.py`.
- **ML is offline only.** The final plugin ships pure DSP with no ML at runtime.

## Project Structure

```
primitives/          DSP building blocks (delay_line, filters, matrix)
engine/              FDN core (fdn.py, fdn_modulated.py, params.py)
audio/               Offline render + real-time audio I/O
gui/                 Parameter control GUI + XY pad + presets
ml/                  Evaluation metrics, search strategies, VAE
optimize/            Performance (Numba, C inner loop)
plugin/              VST plugin (JUCE or nih-plug)
```

## Phases (Summary)

1. **Static FDN in Python** — Build 6 DSP primitives, wire into 8-node FDN, offline WAV processing
2. **Real-Time Audio + GUI** — sounddevice callback, slider GUI, XY pad, preset save/load
3. **Time-Varying FDN** — Modulation at 3 timescales (slow/LFO/audio-rate), parameter space explodes
4. **ML Exploration** — CMA-ES broad search, Bayesian refinement, human-in-the-loop surrogate, VAE latent space
5. **VST Plugin** — Port DSP to C++/Rust, ship presets + latent space lookup, no Python/ML at runtime

## Workflow

- Track progress in `TODO.md` — check off items (`- [x]`) as they are completed.
- Consult `PLAN.md` for detailed context on any phase or primitive.

## Development Approach

- Everything from scratch. No DSP libraries.
- Build each primitive in isolation, test by listening.
- Pure Python first, optimize later (Numba -> C via ctypes).
- Wide parameter ranges including intentionally "broken" configs — the interesting territory is at the boundary.

## The 6 DSP Primitives (build order)

1. Circular buffer (delay line)
2. Fractional delay interpolation (linear + cubic)
3. One-pole filter (damping)
4. Biquad filter (precise tonal shaping)
5. Allpass filter (diffusion)
6. Matrix-vector multiply (Householder feedback matrix)

## Quick Reference

- **FDN node count:** 8
- **Feedback matrix:** Householder (`A = I - (2/N) * ones * ones^T`), O(N) computation
- **Modulation timescales:** Slow (0.01-0.5 Hz), Medium/LFO (0.5-20 Hz), Fast/audio-rate (20+ Hz)
- **ML search:** CMA-ES for broad exploration, Bayesian optimization for refinement, VAE for navigation
- **Target sample rate:** Not specified yet — use 44100 Hz as default
- **Key papers:** Jot & Chaigne 1991, Erbe-Verb (ICMC 2015), Dattorro 1997, ST-ITO (ISMIR 2024)
