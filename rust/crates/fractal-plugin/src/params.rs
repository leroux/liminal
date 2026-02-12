//! nih-plug parameter declarations for the Fractal plugin.
//!
//! Maps all fractal DSP parameters to nih-plug FloatParam/IntParam/EnumParam
//! with appropriate ranges, defaults, and display formatting.

use nih_plug::prelude::*;
use nih_plug_egui::EguiState;
use std::sync::Arc;

/// Interpolation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Enum)]
pub enum InterpMode {
    #[name = "Nearest"]
    Nearest,
    #[name = "Linear"]
    Linear,
}

/// Pre-filter type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Enum)]
pub enum PreFilterType {
    #[name = "Bypass"]
    Bypass,
    #[name = "Lowpass"]
    Lowpass,
    #[name = "Highpass"]
    Highpass,
    #[name = "Bandpass"]
    Bandpass,
}

/// Post-filter type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Enum)]
pub enum PostFilterType {
    #[name = "Bypass"]
    Bypass,
    #[name = "Lowpass"]
    Lowpass,
    #[name = "Highpass"]
    Highpass,
}

#[derive(Params)]
pub struct FractalPluginParams {
    #[persist = "editor-state"]
    pub editor_state: Arc<EguiState>,

    // --- Core Fractal ---
    #[id = "num_scales"]
    pub num_scales: IntParam,
    #[id = "scale_ratio"]
    pub scale_ratio: FloatParam,
    #[id = "amplitude_decay"]
    pub amplitude_decay: FloatParam,
    #[id = "interp"]
    pub interp: EnumParam<InterpMode>,
    #[id = "reverse_scales"]
    pub reverse_scales: BoolParam,
    #[id = "scale_offset"]
    pub scale_offset: FloatParam,

    // --- Iteration / Feedback ---
    #[id = "iterations"]
    pub iterations: IntParam,
    #[id = "iter_decay"]
    pub iter_decay: FloatParam,
    #[id = "saturation"]
    pub saturation: FloatParam,

    // --- Spectral ---
    #[id = "spectral"]
    pub spectral: FloatParam,
    #[id = "window_size"]
    pub window_size: IntParam,

    // --- Pre-filter ---
    #[id = "filter_type"]
    pub filter_type: EnumParam<PreFilterType>,
    #[id = "filter_freq"]
    pub filter_freq: FloatParam,
    #[id = "filter_q"]
    pub filter_q: FloatParam,

    // --- Post-filter ---
    #[id = "post_filter_type"]
    pub post_filter_type: EnumParam<PostFilterType>,
    #[id = "post_filter_freq"]
    pub post_filter_freq: FloatParam,

    // --- Effects ---
    #[id = "gate"]
    pub gate: FloatParam,
    #[id = "crush"]
    pub crush: FloatParam,
    #[id = "decimate"]
    pub decimate: FloatParam,

    // --- Layers ---
    #[id = "layer_gain_1"]
    pub layer_gain_1: FloatParam,
    #[id = "layer_gain_2"]
    pub layer_gain_2: FloatParam,
    #[id = "layer_gain_3"]
    pub layer_gain_3: FloatParam,
    #[id = "layer_gain_4"]
    pub layer_gain_4: FloatParam,
    #[id = "layer_gain_5"]
    pub layer_gain_5: FloatParam,
    #[id = "layer_gain_6"]
    pub layer_gain_6: FloatParam,
    #[id = "layer_gain_7"]
    pub layer_gain_7: FloatParam,
    #[id = "fractal_only_wet"]
    pub fractal_only_wet: BoolParam,
    #[id = "layer_spread"]
    pub layer_spread: FloatParam,
    #[id = "layer_detune"]
    pub layer_detune: FloatParam,
    #[id = "layer_delay"]
    pub layer_delay: FloatParam,
    #[id = "layer_tilt"]
    pub layer_tilt: FloatParam,

    // --- Feedback ---
    #[id = "feedback"]
    pub feedback: FloatParam,

    // --- Output ---
    #[id = "wet_dry"]
    pub wet_dry: FloatParam,
    #[id = "output_gain"]
    pub output_gain: FloatParam,
    #[id = "threshold"]
    pub threshold: FloatParam,
}

impl Default for FractalPluginParams {
    fn default() -> Self {
        Self {
            editor_state: EguiState::from_size(600, 820),

            // --- Core Fractal ---
            num_scales: IntParam::new("Scales", 3, IntRange::Linear { min: 2, max: 8 }),
            scale_ratio: FloatParam::new(
                "Ratio",
                0.5,
                FloatRange::Linear { min: 0.1, max: 0.9 },
            ),
            amplitude_decay: FloatParam::new(
                "Decay",
                0.707,
                FloatRange::Linear { min: 0.1, max: 1.0 },
            ),
            interp: EnumParam::new("Interp", InterpMode::Nearest),
            reverse_scales: BoolParam::new("Reverse", false),
            scale_offset: FloatParam::new(
                "Offset",
                0.0,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            ),

            // --- Iteration / Feedback ---
            iterations: IntParam::new("Iterations", 1, IntRange::Linear { min: 1, max: 4 }),
            iter_decay: FloatParam::new(
                "Iter Decay",
                0.8,
                FloatRange::Linear { min: 0.3, max: 1.0 },
            ),
            saturation: FloatParam::new(
                "Saturation",
                0.0,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            ),

            // --- Spectral ---
            spectral: FloatParam::new(
                "Spectral",
                0.0,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            ),
            window_size: IntParam::new(
                "Window Size",
                2048,
                IntRange::Linear { min: 256, max: 8192 },
            ),

            // --- Pre-filter ---
            filter_type: EnumParam::new("Pre Filter", PreFilterType::Bypass),
            filter_freq: FloatParam::new(
                "Filter Freq",
                2000.0,
                FloatRange::Skewed {
                    min: 20.0,
                    max: 20000.0,
                    factor: FloatRange::skew_factor(-2.0),
                },
            )
            .with_unit(" Hz"),
            filter_q: FloatParam::new(
                "Filter Q",
                0.707,
                FloatRange::Skewed {
                    min: 0.1,
                    max: 10.0,
                    factor: FloatRange::skew_factor(-1.0),
                },
            ),

            // --- Post-filter ---
            post_filter_type: EnumParam::new("Post Filter", PostFilterType::Bypass),
            post_filter_freq: FloatParam::new(
                "Post Freq",
                8000.0,
                FloatRange::Skewed {
                    min: 20.0,
                    max: 20000.0,
                    factor: FloatRange::skew_factor(-2.0),
                },
            )
            .with_unit(" Hz"),

            // --- Effects ---
            gate: FloatParam::new("Gate", 0.0, FloatRange::Linear { min: 0.0, max: 1.0 }),
            crush: FloatParam::new("Crush", 0.0, FloatRange::Linear { min: 0.0, max: 1.0 }),
            decimate: FloatParam::new(
                "Decimate",
                0.0,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            ),

            // --- Layers ---
            layer_gain_1: FloatParam::new(
                "Layer 1 Gain",
                1.0,
                FloatRange::Linear { min: 0.0, max: 2.0 },
            ),
            layer_gain_2: FloatParam::new(
                "Layer 2 Gain",
                1.0,
                FloatRange::Linear { min: 0.0, max: 2.0 },
            ),
            layer_gain_3: FloatParam::new(
                "Layer 3 Gain",
                1.0,
                FloatRange::Linear { min: 0.0, max: 2.0 },
            ),
            layer_gain_4: FloatParam::new(
                "Layer 4 Gain",
                1.0,
                FloatRange::Linear { min: 0.0, max: 2.0 },
            ),
            layer_gain_5: FloatParam::new(
                "Layer 5 Gain",
                1.0,
                FloatRange::Linear { min: 0.0, max: 2.0 },
            ),
            layer_gain_6: FloatParam::new(
                "Layer 6 Gain",
                1.0,
                FloatRange::Linear { min: 0.0, max: 2.0 },
            ),
            layer_gain_7: FloatParam::new(
                "Layer 7 Gain",
                1.0,
                FloatRange::Linear { min: 0.0, max: 2.0 },
            ),
            fractal_only_wet: BoolParam::new("Only Wet", false),
            layer_spread: FloatParam::new(
                "Spread",
                0.0,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            ),
            layer_detune: FloatParam::new(
                "Detune",
                0.0,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            ),
            layer_delay: FloatParam::new(
                "Layer Delay",
                0.0,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            ),
            layer_tilt: FloatParam::new(
                "Tilt",
                0.0,
                FloatRange::Linear { min: -1.0, max: 1.0 },
            ),

            // --- Feedback ---
            feedback: FloatParam::new(
                "Feedback",
                0.0,
                FloatRange::Linear { min: 0.0, max: 0.95 },
            ),

            // --- Output ---
            wet_dry: FloatParam::new("Wet/Dry", 1.0, FloatRange::Linear { min: 0.0, max: 1.0 }),
            output_gain: FloatParam::new(
                "Output Gain",
                0.5,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            ),
            threshold: FloatParam::new(
                "Threshold",
                0.5,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            ),
        }
    }
}

impl FractalPluginParams {
    /// Convert current nih-plug param values to a fractal-dsp FractalParams struct.
    pub fn to_dsp_params(&self) -> fractal_dsp::FractalParams {
        fractal_dsp::FractalParams {
            num_scales: self.num_scales.value(),
            scale_ratio: self.scale_ratio.value() as f64,
            amplitude_decay: self.amplitude_decay.value() as f64,
            interp: self.interp.value() as i32,
            reverse_scales: if self.reverse_scales.value() { 1 } else { 0 },
            scale_offset: self.scale_offset.value() as f64,
            iterations: self.iterations.value(),
            iter_decay: self.iter_decay.value() as f64,
            saturation: self.saturation.value() as f64,
            spectral: self.spectral.value() as f64,
            window_size: self.window_size.value(),
            filter_type: self.filter_type.value() as i32,
            filter_freq: self.filter_freq.value() as f64,
            filter_q: self.filter_q.value() as f64,
            post_filter_type: self.post_filter_type.value() as i32,
            post_filter_freq: self.post_filter_freq.value() as f64,
            gate: self.gate.value() as f64,
            crush: self.crush.value() as f64,
            decimate: self.decimate.value() as f64,
            layer_gain_1: self.layer_gain_1.value() as f64,
            layer_gain_2: self.layer_gain_2.value() as f64,
            layer_gain_3: self.layer_gain_3.value() as f64,
            layer_gain_4: self.layer_gain_4.value() as f64,
            layer_gain_5: self.layer_gain_5.value() as f64,
            layer_gain_6: self.layer_gain_6.value() as f64,
            layer_gain_7: self.layer_gain_7.value() as f64,
            fractal_only_wet: if self.fractal_only_wet.value() { 1 } else { 0 },
            layer_spread: self.layer_spread.value() as f64,
            layer_detune: self.layer_detune.value() as f64,
            layer_delay: self.layer_delay.value() as f64,
            layer_tilt: self.layer_tilt.value() as f64,
            feedback: self.feedback.value() as f64,
            bounce: 0,
            bounce_target: 0,
            bounce_rate: 0.3,
            bounce_lfo_min: 0.1,
            bounce_lfo_max: 5.0,
            wet_dry: self.wet_dry.value() as f64,
            output_gain: self.output_gain.value() as f64,
            threshold: self.threshold.value() as f64,
            seed: 42,
            meta: None,
        }
    }
}
