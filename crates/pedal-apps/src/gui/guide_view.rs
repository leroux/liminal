/// Guide text views — static reference text for each pedal.
use vizia::prelude::*;

pub const REVERB_GUIDE: &str = r#"# Reverb -- FDN Algorithmic Reverb

## Signal Chain
Input -> Pre-delay -> Diffusion allpass cascade -> 8-node FDN -> Stereo panning -> Wet/Dry mix -> Output

## Parameters

### Global
- feedback_gain (0.0-0.999): Overall decay. Higher = longer tail.
- wet_dry (0.0-1.0): Dry/wet mix. 0=dry, 1=fully wet.
- diffusion (0.0-1.0): Allpass diffusion amount.
- saturation (0.0-1.0): Soft clipping in feedback loop.
- pre_delay (0-4410 samples): Gap before reverb onset.
- stereo_width (0.0-1.0): Stereo spread.

### Per-Node (8 nodes)
- delay_times: Delay line lengths in samples.
- damping_coeffs (0.0-1.0): Per-node lowpass damping.
- input_gains: How much input feeds each node.
- output_gains: Tap levels from each node.
- node_pans (-1 to 1): Stereo position of each node.

### Matrix
- matrix_type: Mixing matrix type.
- matrix_seed: RNG seed for random_orthogonal.

### Modulation
- mod_master_rate (0-10 Hz): LFO rate. 0=no modulation.
- mod_depth_delay/damping/output: Per-node modulation depths.
- mod_correlation (0-1): Phase spread across nodes.
"#;

pub const LOSSY_GUIDE: &str = r#"# Lossy -- Codec Emulation Effect

## Signal Chain
Input -> Spectral Loss -> Bitcrush -> Packet Loss -> Filter -> Effects -> Bounce -> Mix -> Output

## Parameters

### Spectral Loss
- loss (0.0-1.0): How many FFT bins to zero.
- window_size (64-16384): STFT window size.
- phase_loss (0.0-1.0): Phase randomization.

### Crush
- crush (0.0-1.0): Bit depth reduction.
- decimate (0.0-1.0): Sample rate reduction.

### Packets
- packets (0/1): Enable packet loss.
- packet_rate (0.0-1.0): Dropout frequency.

### Filter
- filter_type (0-3): off/lowpass/highpass/bandpass.
- filter_freq (20-20000 Hz): Cutoff.

### Effects
- verb (0.0-1.0): Reverb amount.
- gate (0.0-1.0): Noise gate threshold.
"#;

pub const FRACTAL_GUIDE: &str = r#"# Fractal -- Audio Fractalization Effect

## Signal Chain
Input -> Pre-filter -> Fractal core -> Iteration feedback -> Spectral fractal -> Post-filter -> Effects -> Mix -> Output

## Parameters

### Core
- num_scales (2-8): Number of time-scale layers.
- scale_ratio (0.1-0.9): Ratio between successive scales.
- amplitude_decay (0.1-1.0): Volume per scale.

### Layers
- layer_gain_1..7 (0.0-2.0): Per-scale level.
- layer_spread (0.0-1.0): Stereo spread.

### Iteration / Feedback
- iterations (1-4): Fractal passes.
- saturation (0.0-1.0): Soft clipping.
- feedback (0.0-0.95): Output feedback.

### Spectral
- spectral (0.0-1.0): Time vs spectral blend.
- window_size (256-8192): STFT window.
"#;

/// Build a guide view for the given pedal text.
pub fn guide_view(cx: &mut Context, text: &'static str) {
    ScrollView::new(cx, |cx| {
        VStack::new(cx, |cx| {
            for line in text.lines() {
                if let Some(heading) = line.strip_prefix("# ") {
                    Label::new(cx, heading)
                        .class("bright")
                        .font_size(18.0);
                } else if let Some(heading) = line.strip_prefix("## ") {
                    Label::new(cx, heading)
                        .class("bright")
                        .font_size(15.0);
                } else if let Some(heading) = line.strip_prefix("### ") {
                    Label::new(cx, heading)
                        .class("bright")
                        .font_size(13.0);
                } else if line.starts_with("- ") {
                    Label::new(cx, line).class("dim");
                } else if line.is_empty() {
                    Element::new(cx).height(Pixels(8.0));
                } else {
                    Label::new(cx, line);
                }
            }
        })
        .vertical_gap(Pixels(2.0))
        .padding(Pixels(8.0));
    });
}
