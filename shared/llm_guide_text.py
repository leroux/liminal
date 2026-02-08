"""Condensed parameter guides for the LLM tuner system prompt.

Both pedals import from here — single source of truth.
"""

REVERB_GUIDE = """You are an expert audio DSP engineer tuning an 8-node Feedback Delay Network reverb.

SIGNAL CHAIN: Input -> Pre-delay -> Input Diffusion (allpass chain) -> FDN Loop -> Wet/Dry Mix -> Output
FDN Loop: Read 8 delay lines -> one-pole damping -> feedback matrix multiply -> scale by feedback gain + saturate (tanh) -> write back to delay lines -> sum weighted outputs

PARAMETERS AND RANGES:

Global:
- feedback_gain (0.0-2.0): Energy recirculation. 0=no reverb, 0.85=medium room, 0.95+=long tail. >1.0 WILL explode unless saturation is turned up.
- wet_dry (0.0-1.0): 0=dry only, 0.5=equal blend, 1.0=100% reverb.
- diffusion (0.0-0.7): Allpass chain smearing. 0=sharp attacks, 0.5+=heavy smearing. 4 stages internally.
- saturation (0.0-1.0): Tanh soft-clipping in feedback loop. 0=clean/linear, 0.3=warm, 0.7+=aggressive distortion. Prevents explosion when feedback>1.0.
- stereo_width (0.0-1.0): 0=mono, 1=full stereo spread.
- pre_delay (0-11025 samples): Silence before reverb. 0-441=intimate, 882-2205=medium room, 3528+=large hall. (44100 samples = 1 second)

Matrix:
- matrix_type: "householder" (smooth, default), "hadamard", "diagonal" (metallic/comb), "random_orthogonal", "circulant" (ring), "stautner_puckette" (classic paired)

Per-node (8 values each):
- delay_times (1-13230 samples): Delay lengths. Short <660=small resonant space, 2205-3528=medium room, long=hall. Use prime-ish ratios for density.
- damping_coeffs (0.0-0.99): One-pole lowpass per node. 0=bright, 0.3=warm, 0.7+=dark/muffled.
- input_gains (0.0-0.5): How much input feeds each node. Default 0.125 (equal).
- output_gains (0.0-2.0): Each node's contribution to output. 0=silent, 1=normal, >1=amplified.
- node_pans (-1.0 to 1.0): Stereo position per node. -1=left, 0=center, 1=right.

Modulation:
- mod_master_rate (0.0-1000.0 Hz): LFO speed. 0=off, 0.1=slow evolve, 2=chorus, 80+=FM territory.
- mod_depth_delay (0.0-100.0 samples per node): Delay time swing. 3-5=subtle chorus, 20+=pitch wobble.
- mod_depth_damping (0.0-0.5 per node): Brightness modulation. Creates breathing bright/dark.
- mod_depth_output (0.0-1.0 per node): Output gain modulation. Tremolo-like amplitude variation.
- mod_depth_matrix (0.0-1.0): Blend toward second matrix over time.
- mod_correlation (0.0-1.0): Phase spread. 1=all in sync, 0=maximum decorrelation (wider stereo).
- mod_waveform: 0=sine, 1=triangle, 2=sample_and_hold (stepped random).
- mod_node_rate_mult (0.25-4.0 per node): Per-node LFO rate = master * multiplier. Use integer ratios for rhythmic relationships.

RECIPES:
- Natural room: feedback 0.7-0.9, damping 0.2-0.4, diffusion 0.4-0.5, householder, delays 20-80ms
- Infinite drone: feedback 1.0-1.5, saturation 0.3-0.6, low damping
- Metallic/comb: diagonal matrix, short delays 1-5ms, high feedback
- Dark ambient wash: damping 0.7+, long delays, feedback 0.9+, high diffusion
- Chorus reverb: mod_master_rate 1-3 Hz, mod_depth_delay 3-8 samples, mod_correlation 0.3-0.6
"""

LOSSY_GUIDE = """You are an expert audio engineer tuning a codec artifact emulator (lossy audio effect).

SIGNAL CHAIN: Input -> Spectral Loss (STFT) -> Crush/Decimate -> Packets -> Filter -> Verb -> Gate -> Limiter -> Wet/Dry Mix -> Output

PARAMETERS AND RANGES:

Spectral Loss:
- inverse: 0=Standard (hear processed signal), 1=Inverse (hear the residual — everything Standard discards)
- jitter (0.0-1.0): Random phase perturbation amount. 0=off, 1=max. Independent of magnitude processing.
- loss (0.0-1.0): Destruction amount. 0=clean, 0.5=noticeable degradation, 1.0=heavily destroyed.
- window_size (64-16384): FFT window size in samples. Large=smooth/dark, small=glitchy/garbled. Common: 4096/2048/1024/512/256.
- hop_divisor (1-8): Overlap ratio = 1/hop_divisor. 4=75% overlap (default), 2=50%, 8=87.5%.
- n_bands (2-64): Number of Bark-like bands for psychoacoustic gating. Fewer=coarser, more dramatic.
- global_amount (0.0-1.0): Master intensity multiplier for all spectral processing.
- phase_loss (0.0-1.0): Phase quantization. 0=off, higher=more smeared/phasey.
- quantizer: 0=uniform (classic), 1=compand (MP3-style power-law codec).
- pre_echo (0.0-1.0): Boosts loss before transients, mimicking MP3 pre-echo artifacts.
- transient_ratio (1.5-20.0): Energy ratio threshold for pre-echo detection. Lower=more sensitive.
- noise_shape (0.0-1.0): Envelope-following quantization — coarser in quiet bands.
- weighting (0.0-1.0): 0=equal frequency weighting, 1=psychoacoustic ATH model (like real codecs).
- hf_threshold (0.0-1.0): Loss level where HF rolloff begins. Default 0.3.
- slushy_rate (0.001-0.5): Freeze slushy drift speed. Low=slow morphing, high=tracks input.

Crush (time-domain):
- crush (0.0-1.0): Bitcrusher. 0=16-bit (clean), 1=4-bit (destroyed).
- decimate (0.0-1.0): Sample rate reduction. 0=full rate, 1=extreme aliasing.

Packets:
- packets: 0=Clean, 1=Packet Loss (dropouts), 2=Packet Repeat (stutters).
- packet_rate (0.0-1.0): Probability of entering bad state (dropout/repeat).
- packet_size (5.0-200.0 ms): Chunk length for packet processing.

Filter:
- filter_type: 0=Bypass, 1=Bandpass, 2=Notch.
- filter_freq (20.0-20000.0 Hz): Center frequency.
- filter_width (0.0-1.0): 0=narrow/high-Q (resonant), 1=wide/low-Q (gentle).
- filter_slope: 0=6dB/oct (gentle), 1=24dB/oct (steep), 2=96dB/oct (brick wall).

Effects:
- verb (0.0-1.0): Lo-fi Schroeder reverb mix. Intentionally cheap and metallic.
- decay (0.0-1.0): Reverb decay length.
- freeze: 0=off, 1=on. Captures and holds spectral snapshot.
- freeze_mode: 0=Slushy (slowly evolving freeze), 1=Solid (static freeze).
- freezer (0.0-1.0): Frozen/live blend. 0=fully live, 1=fully frozen.
- gate (0.0-1.0): Noise gate threshold. 0=off.

Output:
- wet_dry (0.0-1.0): 0=dry, 1=wet.

RECIPES:
- Underwater/streaming: loss 0.7-0.9, window_size 4096, inverse 0, no crush
- Glitchy digital: loss 0.5, window_size 256-512, packet loss, crush 0.3
- Lo-fi radio: loss 0.4, bandpass filter at 800-2000Hz, verb 0.2, decimate 0.3
- Frozen texture: freeze on, slushy mode, loss 0.5, verb 0.3
- Extreme destruction: loss 1.0, crush 0.6, decimate 0.5, packet repeat
"""


# Per-parameter descriptions for JSON schema "description" fields.
# Keys match default_params() keys. Used by LLMTuner to auto-generate the schema.

REVERB_PARAM_DESCRIPTIONS = {
    "delay_times": "8 values, range 1-13230 samples. Delay lengths per node. Prime-ish ratios for density.",
    "damping_coeffs": "8 values, range 0.0-0.99. One-pole lowpass per node. 0=bright, 0.3=warm, 0.7+=dark.",
    "feedback_gain": "Range 0.0-2.0. Energy recirculation. 0=no reverb, 0.85=medium, >1.0 explodes without saturation.",
    "input_gains": "8 values, range 0.0-0.5. How much input feeds each node. Default 0.125.",
    "output_gains": "8 values, range 0.0-2.0. Each node's output contribution. 0=silent, 1=normal.",
    "pre_delay": "Range 0-11025 samples. Silence before reverb. 441=10ms, 2205=50ms.",
    "diffusion": "Range 0.0-0.7. Allpass smearing. 0=sharp, 0.5+=heavy smearing.",
    "diffusion_stages": "Number of allpass stages (usually 4).",
    "diffusion_delays": "4 allpass delay times in samples.",
    "wet_dry": "Range 0.0-1.0. 0=dry, 0.5=equal blend, 1.0=fully wet.",
    "saturation": "Range 0.0-1.0. Tanh soft-clip in feedback. Prevents explosion when feedback>1.0.",
    "matrix_type": "Feedback matrix topology. Options: householder, hadamard, diagonal, random_orthogonal, circulant, stautner_puckette.",
    "matrix_seed": "Random seed for random_orthogonal matrix.",
    "node_pans": "8 values, range -1.0 to 1.0. Stereo position per node. -1=left, 0=center, 1=right.",
    "stereo_width": "Range 0.0-1.0. 0=mono, 1=full stereo.",
    "mod_master_rate": "Range 0.0-1000.0 Hz. LFO speed. 0=off, 0.1=slow, 2=chorus, 80+=FM.",
    "mod_node_rate_mult": "8 values, range 0.25-4.0. Per-node LFO rate multiplier.",
    "mod_correlation": "Range 0.0-1.0. Phase spread. 1=sync, 0=max decorrelation.",
    "mod_waveform": "0=sine, 1=triangle, 2=sample_and_hold.",
    "mod_depth_delay": "8 values, range 0.0-100.0 samples. Delay time modulation depth.",
    "mod_depth_damping": "8 values, range 0.0-0.5. Damping modulation depth.",
    "mod_depth_output": "8 values, range 0.0-1.0. Output gain modulation depth.",
    "mod_depth_matrix": "Range 0.0-1.0. Matrix blend modulation depth.",
    "mod_rate_scale_delay": "Range 0.01-10.0. Delay modulation rate multiplier.",
    "mod_rate_scale_damping": "Range 0.01-10.0. Damping modulation rate multiplier.",
    "mod_rate_scale_output": "Range 0.01-10.0. Output modulation rate multiplier.",
    "mod_rate_matrix": "Range 0.0-1000.0 Hz. Matrix modulation rate.",
    "mod_matrix2_type": "Second matrix type for modulation blending.",
    "mod_matrix2_seed": "Random seed for second matrix.",
}

LOSSY_PARAM_DESCRIPTIONS = {
    "inverse": "0=Standard (hear processed), 1=Inverse (hear residual).",
    "jitter": "Range 0.0-1.0. Random phase perturbation. 0=off, 1=max. Independent of magnitude processing.",
    "loss": "Range 0.0-1.0. Destruction amount. 0=clean, 1.0=destroyed.",
    "window_size": "Range 64-16384. FFT window size in samples. Large=smooth, small=glitchy.",
    "hop_divisor": "Range 1-8. Overlap = 1/hop_divisor. 4=75% overlap (default).",
    "n_bands": "Range 2-64. Bark-like bands for gating. Fewer=coarser/dramatic.",
    "global_amount": "Range 0.0-1.0. Master intensity multiplier.",
    "phase_loss": "Range 0.0-1.0. Phase quantization. 0=off, 1=extreme.",
    "quantizer": "0=uniform (classic), 1=compand (MP3-style codec).",
    "pre_echo": "Range 0.0-1.0. Pre-echo artifact enhancement.",
    "transient_ratio": "Range 1.5-20.0. Energy ratio threshold for pre-echo detection.",
    "noise_shape": "Range 0.0-1.0. Envelope-following quantization noise shaping.",
    "weighting": "Range 0.0-1.0. 0=equal freq weighting, 1=psychoacoustic ATH model.",
    "hf_threshold": "Range 0.0-1.0. Loss level where HF rolloff begins. Default 0.3.",
    "slushy_rate": "Range 0.001-0.5. Freeze slushy drift speed.",
    "crush": "Range 0.0-1.0. Bitcrusher. 0=16-bit clean, 1=4-bit destroyed.",
    "decimate": "Range 0.0-1.0. Sample rate reduction. 0=full rate, 1=extreme aliasing.",
    "packets": "0=Clean, 1=Packet Loss (dropouts), 2=Packet Repeat (stutters).",
    "packet_rate": "Range 0.0-1.0. Probability of dropout/repeat.",
    "packet_size": "Range 5.0-200.0 ms. Packet chunk length.",
    "filter_type": "0=Bypass, 1=Bandpass, 2=Notch.",
    "filter_freq": "Range 20.0-20000.0 Hz. Filter center frequency.",
    "filter_width": "Range 0.0-1.0. 0=narrow/resonant, 1=wide/gentle.",
    "filter_slope": "0=6dB/oct (gentle), 1=24dB/oct (steep), 2=96dB/oct (brick wall).",
    "verb": "Range 0.0-1.0. Lo-fi Schroeder reverb mix.",
    "decay": "Range 0.0-1.0. Reverb decay length.",
    "verb_position": "0=Pre (before loss), 1=Post (after filter).",
    "freeze": "0=off, 1=on. Captures spectral snapshot.",
    "freeze_mode": "0=Slushy (slowly evolving), 1=Solid (static freeze).",
    "freezer": "Range 0.0-1.0. Frozen/live blend. 0=live, 1=frozen.",
    "gate": "Range 0.0-1.0. Noise gate threshold. 0=off.",
    "threshold": "Range 0.0-1.0. Limiter threshold. 0=heavy, 1=light.",
    "auto_gain": "Range 0.0-1.0. Automatic gain compensation.",
    "loss_gain": "Range 0.0-1.0. Wet signal gain. 0.5=unity.",
    "bounce": "0=off, 1=on. Parameter modulation.",
    "bounce_target": "Index of parameter to modulate.",
    "bounce_rate": "Range 0.0-1.0. Modulation LFO rate.",
    "bounce_lfo_min": "Range 0.01-50.0 Hz. LFO minimum frequency.",
    "bounce_lfo_max": "Range 0.01-50.0 Hz. LFO maximum frequency.",
    "wet_dry": "Range 0.0-1.0. 0=dry, 1=wet.",
    "seed": "Random seed for reproducibility.",
}
