Research the optimal and most musically interesting parameter configurations for an audio fractalization DSP effect. The algorithm works as follows:

## Algorithm

The core effect takes a mono audio signal and creates self-similar fractal-like texture by layering compressed-and-tiled copies of the signal at multiple timescales:

```
for each scale s from 1 to num_scales:
    compressed_length = signal_length * (scale_ratio ^ s)
    downsample the signal to compressed_length (nearest-neighbor or linear interpolation)
    optionally reverse the compressed chunk
    tile (repeat) the compressed chunk to fill the original signal length
    add to output with gain = amplitude_decay ^ s
normalize output to input peak level
```

This can also be applied in the spectral domain (to STFT magnitude frames instead of raw samples), and the two modes can be blended. The output can be fed back through the fractalizer multiple times with tanh saturation between passes.

## Full Signal Chain

```
Input -> Pre-Filter (LP/HP/BP) -> Fractalize (x iterations, with tanh saturation + decay between passes) -> Output Gain -> Bitcrusher + Sample Rate Reducer -> Post-Filter (LP/HP) -> Noise Gate -> Limiter -> Wet/Dry Mix -> Output
```

## Parameter Space

| Parameter | Range | Description |
|-----------|-------|-------------|
| num_scales | 2-8 | Number of fractal scale layers |
| scale_ratio | 0.1-0.9 | Compression ratio per scale (0.5 = halve length each level) |
| amplitude_decay | 0.1-1.0 | Gain decay per scale layer |
| interp | nearest / linear | Resampling mode. Nearest = aliased/gritty, linear = smooth |
| reverse_scales | on/off | Reverse the compressed chunks before tiling |
| scale_offset | 0.0-1.0 | Phase offset for tile start position |
| iterations | 1-4 | Number of feedback passes through the fractalizer |
| iter_decay | 0.3-1.0 | Gain applied between iterations |
| saturation | 0.0-1.0 | tanh soft-clipping between iterations |
| spectral | 0.0-1.0 | Blend between time-domain (0) and STFT spectral-domain (1) fractalization |
| window_size | 256-8192 | STFT window size for spectral mode |
| filter_type | bypass/LP/HP/BP | Pre-fractal filter |
| filter_freq | 20-20000 Hz | Pre-filter cutoff |
| filter_q | 0.1-10.0 | Pre-filter resonance |
| post_filter_type | bypass/LP/HP | Post-fractal filter |
| post_filter_freq | 20-20000 Hz | Post-filter cutoff |
| crush | 0.0-1.0 | Bitcrusher (16-bit to 4-bit) |
| decimate | 0.0-1.0 | Sample rate reduction (zero-order hold) |
| gate | 0.0-1.0 | Noise gate threshold |
| wet_dry | 0.0-1.0 | Dry/wet mix |
| output_gain | 0.0-1.0 | Output level (0=-36dB, 0.5=unity, 1=+36dB) |

## What I Need

Research and reason deeply about:

1. **DSP theory of the fractal tiling operation** — What happens mathematically when you tile compressed copies of a signal? What are the spectral consequences? How does scale_ratio relate to the resulting harmonic/inharmonic content? What aliasing patterns does nearest-neighbor create vs linear interpolation?

2. **Interaction effects between parameters** — How do num_scales and scale_ratio interact (e.g. 8 scales at 0.5 ratio means the last layer is 1/256th the original length — what does that sound like)? How does iteration with saturation create emergent behavior? What happens when spectral and time-domain modes are blended?

3. **Musically useful configurations** — Based on your analysis, propose 20-30 specific parameter sets organized by musical category. For each, explain the DSP reasoning for why it should sound interesting. Categories should include:
   - Subtle enhancement (guitar, vocals, synths)
   - Rhythmic/percussive textures
   - Ambient/pad textures
   - Aggressive/industrial
   - Glitch/experimental
   - Sound design (foley, transitions, risers)
   - Combinations with the filter/crush/decimate chain that create novel timbres

4. **Scale ratio sweet spots** — Are there mathematically interesting ratios (golden ratio ~0.618, powers of 2, etc.) that create more musical results? What about integer relationships between scale layers?

5. **The spectral mode** — When you apply the compress-and-tile operation to STFT magnitude frames instead of raw samples, what's the perceptual difference? When would you use spectral vs time-domain vs blended?

6. **Iteration dynamics** — With saturation in the feedback loop, the system has nonlinear dynamics. At what parameter ranges does it produce warm saturation vs chaotic behavior? What are the stable regimes?

7. **Pre/post filter strategies** — When does it help to highpass before fractalization (removing bass that would create rumble when tiled)? When does post-LP help (taming aliasing from nearest-neighbor)?

**For each proposed parameter set, output it as a JSON object with all parameter values specified.** Name each one and explain what it should sound like and why the parameter choices create that result.
