//! nih-plug parameter declarations for the Reverb plugin.
//!
//! Maps reverb DSP parameters to nih-plug FloatParam/IntParam/EnumParam
//! with appropriate ranges, defaults, and display formatting.

use nih_plug::prelude::*;
use nih_plug_egui::EguiState;
use std::sync::Arc;

const SR: f64 = 44100.0;

/// Feedback matrix type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Enum)]
pub enum MatrixType {
    #[name = "Householder"]
    Householder,
    #[name = "Hadamard"]
    Hadamard,
    #[name = "Diagonal"]
    Diagonal,
    #[name = "Random Orthogonal"]
    RandomOrthogonal,
    #[name = "Circulant"]
    Circulant,
    #[name = "Stautner-Puckette"]
    StautnerPuckette,
}

impl MatrixType {
    pub fn to_string_id(self) -> String {
        match self {
            MatrixType::Householder => "householder",
            MatrixType::Hadamard => "hadamard",
            MatrixType::Diagonal => "diagonal",
            MatrixType::RandomOrthogonal => "random_orthogonal",
            MatrixType::Circulant => "circulant",
            MatrixType::StautnerPuckette => "stautner_puckette",
        }
        .to_string()
    }
}

/// Modulation LFO waveform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Enum)]
pub enum ModWaveform {
    #[name = "Sine"]
    Sine,
    #[name = "Triangle"]
    Triangle,
    #[name = "Sample & Hold"]
    SampleAndHold,
}

#[derive(Params)]
pub struct ReverbPluginParams {
    #[persist = "editor-state"]
    pub editor_state: Arc<EguiState>,

    // --- Global ---
    #[id = "feedback_gain"]
    pub feedback_gain: FloatParam,
    #[id = "wet_dry"]
    pub wet_dry: FloatParam,
    #[id = "diffusion"]
    pub diffusion: FloatParam,
    #[id = "diffusion_stages"]
    pub diffusion_stages: IntParam,
    #[id = "saturation"]
    pub saturation: FloatParam,
    #[id = "pre_delay"]
    pub pre_delay: IntParam,
    #[id = "stereo_width"]
    pub stereo_width: FloatParam,

    // --- Matrix ---
    #[id = "matrix_type"]
    pub matrix_type: EnumParam<MatrixType>,
    #[id = "matrix_seed"]
    pub matrix_seed: IntParam,

    // --- Delay Times (8 nodes) ---
    #[id = "delay_time_1"]
    pub delay_time_1: IntParam,
    #[id = "delay_time_2"]
    pub delay_time_2: IntParam,
    #[id = "delay_time_3"]
    pub delay_time_3: IntParam,
    #[id = "delay_time_4"]
    pub delay_time_4: IntParam,
    #[id = "delay_time_5"]
    pub delay_time_5: IntParam,
    #[id = "delay_time_6"]
    pub delay_time_6: IntParam,
    #[id = "delay_time_7"]
    pub delay_time_7: IntParam,
    #[id = "delay_time_8"]
    pub delay_time_8: IntParam,

    // --- Damping (8 nodes) ---
    #[id = "damping_1"]
    pub damping_1: FloatParam,
    #[id = "damping_2"]
    pub damping_2: FloatParam,
    #[id = "damping_3"]
    pub damping_3: FloatParam,
    #[id = "damping_4"]
    pub damping_4: FloatParam,
    #[id = "damping_5"]
    pub damping_5: FloatParam,
    #[id = "damping_6"]
    pub damping_6: FloatParam,
    #[id = "damping_7"]
    pub damping_7: FloatParam,
    #[id = "damping_8"]
    pub damping_8: FloatParam,

    // --- Input Gains (8 nodes) ---
    #[id = "input_gain_1"]
    pub input_gain_1: FloatParam,
    #[id = "input_gain_2"]
    pub input_gain_2: FloatParam,
    #[id = "input_gain_3"]
    pub input_gain_3: FloatParam,
    #[id = "input_gain_4"]
    pub input_gain_4: FloatParam,
    #[id = "input_gain_5"]
    pub input_gain_5: FloatParam,
    #[id = "input_gain_6"]
    pub input_gain_6: FloatParam,
    #[id = "input_gain_7"]
    pub input_gain_7: FloatParam,
    #[id = "input_gain_8"]
    pub input_gain_8: FloatParam,

    // --- Output Gains (8 nodes) ---
    #[id = "output_gain_1"]
    pub output_gain_1: FloatParam,
    #[id = "output_gain_2"]
    pub output_gain_2: FloatParam,
    #[id = "output_gain_3"]
    pub output_gain_3: FloatParam,
    #[id = "output_gain_4"]
    pub output_gain_4: FloatParam,
    #[id = "output_gain_5"]
    pub output_gain_5: FloatParam,
    #[id = "output_gain_6"]
    pub output_gain_6: FloatParam,
    #[id = "output_gain_7"]
    pub output_gain_7: FloatParam,
    #[id = "output_gain_8"]
    pub output_gain_8: FloatParam,

    // --- Node Pans (8 nodes) ---
    #[id = "node_pan_1"]
    pub node_pan_1: FloatParam,
    #[id = "node_pan_2"]
    pub node_pan_2: FloatParam,
    #[id = "node_pan_3"]
    pub node_pan_3: FloatParam,
    #[id = "node_pan_4"]
    pub node_pan_4: FloatParam,
    #[id = "node_pan_5"]
    pub node_pan_5: FloatParam,
    #[id = "node_pan_6"]
    pub node_pan_6: FloatParam,
    #[id = "node_pan_7"]
    pub node_pan_7: FloatParam,
    #[id = "node_pan_8"]
    pub node_pan_8: FloatParam,

    // --- Modulation ---
    #[id = "mod_master_rate"]
    pub mod_master_rate: FloatParam,
    #[id = "mod_correlation"]
    pub mod_correlation: FloatParam,
    #[id = "mod_waveform"]
    pub mod_waveform: EnumParam<ModWaveform>,
    #[id = "mod_depth_delay"]
    pub mod_depth_delay: FloatParam,
    #[id = "mod_depth_damping"]
    pub mod_depth_damping: FloatParam,
    #[id = "mod_depth_output"]
    pub mod_depth_output: FloatParam,
    #[id = "mod_depth_matrix"]
    pub mod_depth_matrix: FloatParam,
    #[id = "mod_rate_scale_delay"]
    pub mod_rate_scale_delay: FloatParam,
    #[id = "mod_rate_scale_damping"]
    pub mod_rate_scale_damping: FloatParam,
    #[id = "mod_rate_scale_output"]
    pub mod_rate_scale_output: FloatParam,
    #[id = "mod_rate_matrix"]
    pub mod_rate_matrix: FloatParam,
    #[id = "mod_matrix2_type"]
    pub mod_matrix2_type: EnumParam<MatrixType>,
    #[id = "mod_matrix2_seed"]
    pub mod_matrix2_seed: IntParam,
}

fn ms_to_samples(ms: f64) -> i32 {
    (ms / 1000.0 * SR) as i32
}

fn delay_param(name: &str, default_ms: f64) -> IntParam {
    IntParam::new(
        name,
        ms_to_samples(default_ms),
        IntRange::Linear {
            min: 1,
            max: ms_to_samples(300.0),
        },
    )
}

fn damping_param(name: &str) -> FloatParam {
    FloatParam::new(name, 0.3, FloatRange::Linear { min: 0.0, max: 0.99 })
}

fn input_gain_param(name: &str) -> FloatParam {
    FloatParam::new(name, 0.125, FloatRange::Linear { min: 0.0, max: 0.5 })
}

fn output_gain_param(name: &str) -> FloatParam {
    FloatParam::new(name, 1.0, FloatRange::Linear { min: 0.0, max: 2.0 })
}

fn pan_param(name: &str, default: f32) -> FloatParam {
    FloatParam::new(name, default, FloatRange::Linear { min: -1.0, max: 1.0 })
}

impl Default for ReverbPluginParams {
    fn default() -> Self {
        let default_pans: [f32; 8] = [
            -1.0, -0.714, -0.429, -0.143, 0.143, 0.429, 0.714, 1.0,
        ];
        let default_delays_ms: [f64; 8] = [29.7, 37.1, 41.3, 47.9, 53.1, 59.3, 67.7, 73.1];

        Self {
            editor_state: EguiState::from_size(620, 780),

            // --- Global ---
            feedback_gain: FloatParam::new(
                "Feedback",
                0.85,
                FloatRange::Linear { min: 0.0, max: 2.0 },
            ),
            wet_dry: FloatParam::new("Wet/Dry", 0.5, FloatRange::Linear { min: 0.0, max: 1.0 }),
            diffusion: FloatParam::new(
                "Diffusion",
                0.5,
                FloatRange::Linear { min: 0.0, max: 0.7 },
            ),
            diffusion_stages: IntParam::new(
                "Diff Stages",
                4,
                IntRange::Linear { min: 1, max: 4 },
            ),
            saturation: FloatParam::new(
                "Saturation",
                0.0,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            ),
            pre_delay: IntParam::new(
                "Pre-Delay",
                ms_to_samples(10.0),
                IntRange::Linear {
                    min: 0,
                    max: ms_to_samples(250.0),
                },
            ),
            stereo_width: FloatParam::new(
                "Stereo Width",
                1.0,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            ),

            // --- Matrix ---
            matrix_type: EnumParam::new("Matrix", MatrixType::Householder),
            matrix_seed: IntParam::new("Seed", 42, IntRange::Linear { min: 1, max: 9999 }),

            // --- Delay Times ---
            delay_time_1: delay_param("Delay 1", default_delays_ms[0]),
            delay_time_2: delay_param("Delay 2", default_delays_ms[1]),
            delay_time_3: delay_param("Delay 3", default_delays_ms[2]),
            delay_time_4: delay_param("Delay 4", default_delays_ms[3]),
            delay_time_5: delay_param("Delay 5", default_delays_ms[4]),
            delay_time_6: delay_param("Delay 6", default_delays_ms[5]),
            delay_time_7: delay_param("Delay 7", default_delays_ms[6]),
            delay_time_8: delay_param("Delay 8", default_delays_ms[7]),

            // --- Damping ---
            damping_1: damping_param("Damp 1"),
            damping_2: damping_param("Damp 2"),
            damping_3: damping_param("Damp 3"),
            damping_4: damping_param("Damp 4"),
            damping_5: damping_param("Damp 5"),
            damping_6: damping_param("Damp 6"),
            damping_7: damping_param("Damp 7"),
            damping_8: damping_param("Damp 8"),

            // --- Input Gains ---
            input_gain_1: input_gain_param("In 1"),
            input_gain_2: input_gain_param("In 2"),
            input_gain_3: input_gain_param("In 3"),
            input_gain_4: input_gain_param("In 4"),
            input_gain_5: input_gain_param("In 5"),
            input_gain_6: input_gain_param("In 6"),
            input_gain_7: input_gain_param("In 7"),
            input_gain_8: input_gain_param("In 8"),

            // --- Output Gains ---
            output_gain_1: output_gain_param("Out 1"),
            output_gain_2: output_gain_param("Out 2"),
            output_gain_3: output_gain_param("Out 3"),
            output_gain_4: output_gain_param("Out 4"),
            output_gain_5: output_gain_param("Out 5"),
            output_gain_6: output_gain_param("Out 6"),
            output_gain_7: output_gain_param("Out 7"),
            output_gain_8: output_gain_param("Out 8"),

            // --- Node Pans ---
            node_pan_1: pan_param("Pan 1", default_pans[0]),
            node_pan_2: pan_param("Pan 2", default_pans[1]),
            node_pan_3: pan_param("Pan 3", default_pans[2]),
            node_pan_4: pan_param("Pan 4", default_pans[3]),
            node_pan_5: pan_param("Pan 5", default_pans[4]),
            node_pan_6: pan_param("Pan 6", default_pans[5]),
            node_pan_7: pan_param("Pan 7", default_pans[6]),
            node_pan_8: pan_param("Pan 8", default_pans[7]),

            // --- Modulation ---
            mod_master_rate: FloatParam::new(
                "Mod Rate",
                0.0,
                FloatRange::Skewed {
                    min: 0.0,
                    max: 1000.0,
                    factor: FloatRange::skew_factor(-2.0),
                },
            )
            .with_unit(" Hz"),
            mod_correlation: FloatParam::new(
                "Mod Corr",
                1.0,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            ),
            mod_waveform: EnumParam::new("Mod Wave", ModWaveform::Sine),
            mod_depth_delay: FloatParam::new(
                "Depth Delay",
                0.0,
                FloatRange::Linear {
                    min: 0.0,
                    max: 100.0,
                },
            ),
            mod_depth_damping: FloatParam::new(
                "Depth Damp",
                0.0,
                FloatRange::Linear { min: 0.0, max: 0.5 },
            ),
            mod_depth_output: FloatParam::new(
                "Depth Out",
                0.0,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            ),
            mod_depth_matrix: FloatParam::new(
                "Depth Matrix",
                0.0,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            ),
            mod_rate_scale_delay: FloatParam::new(
                "Rate Delay",
                1.0,
                FloatRange::Linear {
                    min: 0.01,
                    max: 10.0,
                },
            ),
            mod_rate_scale_damping: FloatParam::new(
                "Rate Damp",
                1.0,
                FloatRange::Linear {
                    min: 0.01,
                    max: 10.0,
                },
            ),
            mod_rate_scale_output: FloatParam::new(
                "Rate Out",
                1.0,
                FloatRange::Linear {
                    min: 0.01,
                    max: 10.0,
                },
            ),
            mod_rate_matrix: FloatParam::new(
                "Rate Matrix",
                0.0,
                FloatRange::Skewed {
                    min: 0.0,
                    max: 1000.0,
                    factor: FloatRange::skew_factor(-2.0),
                },
            )
            .with_unit(" Hz"),
            mod_matrix2_type: EnumParam::new("Matrix 2", MatrixType::RandomOrthogonal),
            mod_matrix2_seed: IntParam::new("Seed 2", 137, IntRange::Linear { min: 1, max: 9999 }),
        }
    }
}

impl ReverbPluginParams {
    /// Convert current nih-plug param values to a reverb-dsp ReverbParams struct.
    pub fn to_dsp_params(&self) -> reverb_dsp::ReverbParams {
        let mod_dd = self.mod_depth_delay.value() as f64;
        let mod_da = self.mod_depth_damping.value() as f64;
        let mod_do = self.mod_depth_output.value() as f64;

        reverb_dsp::ReverbParams {
            delay_times: vec![
                self.delay_time_1.value(),
                self.delay_time_2.value(),
                self.delay_time_3.value(),
                self.delay_time_4.value(),
                self.delay_time_5.value(),
                self.delay_time_6.value(),
                self.delay_time_7.value(),
                self.delay_time_8.value(),
            ],
            damping_coeffs: vec![
                self.damping_1.value() as f64,
                self.damping_2.value() as f64,
                self.damping_3.value() as f64,
                self.damping_4.value() as f64,
                self.damping_5.value() as f64,
                self.damping_6.value() as f64,
                self.damping_7.value() as f64,
                self.damping_8.value() as f64,
            ],
            input_gains: vec![
                self.input_gain_1.value() as f64,
                self.input_gain_2.value() as f64,
                self.input_gain_3.value() as f64,
                self.input_gain_4.value() as f64,
                self.input_gain_5.value() as f64,
                self.input_gain_6.value() as f64,
                self.input_gain_7.value() as f64,
                self.input_gain_8.value() as f64,
            ],
            output_gains: vec![
                self.output_gain_1.value() as f64,
                self.output_gain_2.value() as f64,
                self.output_gain_3.value() as f64,
                self.output_gain_4.value() as f64,
                self.output_gain_5.value() as f64,
                self.output_gain_6.value() as f64,
                self.output_gain_7.value() as f64,
                self.output_gain_8.value() as f64,
            ],
            node_pans: vec![
                self.node_pan_1.value() as f64,
                self.node_pan_2.value() as f64,
                self.node_pan_3.value() as f64,
                self.node_pan_4.value() as f64,
                self.node_pan_5.value() as f64,
                self.node_pan_6.value() as f64,
                self.node_pan_7.value() as f64,
                self.node_pan_8.value() as f64,
            ],
            feedback_gain: self.feedback_gain.value() as f64,
            wet_dry: self.wet_dry.value() as f64,
            diffusion: self.diffusion.value() as f64,
            diffusion_stages: self.diffusion_stages.value(),
            diffusion_delays: [5.3, 7.9, 11.7, 16.1]
                .iter()
                .map(|ms| (ms / 1000.0 * SR) as i32)
                .collect(),
            saturation: self.saturation.value() as f64,
            pre_delay: self.pre_delay.value(),
            stereo_width: self.stereo_width.value() as f64,
            matrix_type: self.matrix_type.value().to_string_id(),
            matrix_seed: self.matrix_seed.value(),
            matrix_custom: None,
            mod_master_rate: self.mod_master_rate.value() as f64,
            mod_node_rate_mult: vec![1.0; 8],
            mod_correlation: self.mod_correlation.value() as f64,
            mod_waveform: self.mod_waveform.value() as i32,
            mod_depth_delay: vec![mod_dd; 8],
            mod_depth_damping: vec![mod_da; 8],
            mod_depth_output: vec![mod_do; 8],
            mod_depth_matrix: self.mod_depth_matrix.value() as f64,
            mod_rate_scale_delay: self.mod_rate_scale_delay.value() as f64,
            mod_rate_scale_damping: self.mod_rate_scale_damping.value() as f64,
            mod_rate_scale_output: self.mod_rate_scale_output.value() as f64,
            mod_rate_matrix: self.mod_rate_matrix.value() as f64,
            mod_matrix2_type: self.mod_matrix2_type.value().to_string_id(),
            mod_matrix2_seed: self.mod_matrix2_seed.value(),
            meta: None,
        }
    }
}
