//! nih-plug parameter declarations for the Lossy plugin.
//!
//! Maps all 42 lossy DSP parameters to nih-plug FloatParam/IntParam/EnumParam
//! with appropriate ranges, defaults, and display formatting.

use nih_plug::prelude::*;
use nih_plug_egui::EguiState;
use std::sync::Arc;

/// Lossy processing modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Enum)]
pub enum SpectralMode {
    #[name = "Standard"]
    Standard,
    #[name = "Inverse (Residual)"]
    Inverse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Enum)]
pub enum QuantizerType {
    #[name = "Uniform"]
    Uniform,
    #[name = "Companding"]
    Companding,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Enum)]
pub enum PacketMode {
    #[name = "Off"]
    Off,
    #[name = "Loss"]
    Loss,
    #[name = "Repeat"]
    Repeat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Enum)]
pub enum FilterType {
    #[name = "Bypass"]
    Bypass,
    #[name = "Bandpass"]
    Bandpass,
    #[name = "Notch"]
    Notch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Enum)]
pub enum FilterSlope {
    #[name = "6 dB/oct"]
    Slope6,
    #[name = "24 dB/oct"]
    Slope24,
    #[name = "96 dB/oct"]
    Slope96,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Enum)]
pub enum VerbPosition {
    #[name = "Pre"]
    Pre,
    #[name = "Post"]
    Post,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Enum)]
pub enum FreezeMode {
    #[name = "Slushy"]
    Slushy,
    #[name = "Solid"]
    Solid,
}

#[derive(Params)]
pub struct LossyPluginParams {
    #[persist = "editor-state"]
    pub editor_state: Arc<EguiState>,

    // --- Spectral Loss ---
    #[id = "mode"]
    pub mode: EnumParam<SpectralMode>,
    #[id = "jitter"]
    pub jitter: FloatParam,
    #[id = "loss"]
    pub loss: FloatParam,
    #[id = "window_size"]
    pub window_size: IntParam,
    #[id = "hop_divisor"]
    pub hop_divisor: IntParam,
    #[id = "n_bands"]
    pub n_bands: IntParam,
    #[id = "global_amount"]
    pub global_amount: FloatParam,
    #[id = "phase_loss"]
    pub phase_loss: FloatParam,
    #[id = "quantizer"]
    pub quantizer: EnumParam<QuantizerType>,
    #[id = "pre_echo"]
    pub pre_echo: FloatParam,
    #[id = "noise_shape"]
    pub noise_shape: FloatParam,
    #[id = "weighting"]
    pub weighting: FloatParam,
    #[id = "hf_threshold"]
    pub hf_threshold: FloatParam,
    #[id = "transient_ratio"]
    pub transient_ratio: FloatParam,
    #[id = "slushy_rate"]
    pub slushy_rate: FloatParam,

    // --- Crush ---
    #[id = "crush"]
    pub crush: FloatParam,
    #[id = "decimate"]
    pub decimate: FloatParam,

    // --- Packets ---
    #[id = "packets"]
    pub packets: EnumParam<PacketMode>,
    #[id = "packet_rate"]
    pub packet_rate: FloatParam,
    #[id = "packet_size"]
    pub packet_size: FloatParam,

    // --- Filter ---
    #[id = "filter_type"]
    pub filter_type: EnumParam<FilterType>,
    #[id = "filter_freq"]
    pub filter_freq: FloatParam,
    #[id = "filter_width"]
    pub filter_width: FloatParam,
    #[id = "filter_slope"]
    pub filter_slope: EnumParam<FilterSlope>,

    // --- Reverb ---
    #[id = "verb"]
    pub verb: FloatParam,
    #[id = "decay"]
    pub decay: FloatParam,
    #[id = "verb_position"]
    pub verb_position: EnumParam<VerbPosition>,

    // --- Freeze ---
    #[id = "freeze"]
    pub freeze: BoolParam,
    #[id = "freeze_mode"]
    pub freeze_mode: EnumParam<FreezeMode>,
    #[id = "freezer"]
    pub freezer: FloatParam,

    // --- Gate ---
    #[id = "gate"]
    pub gate: FloatParam,

    // --- Hidden ---
    #[id = "threshold"]
    pub threshold: FloatParam,
    #[id = "auto_gain"]
    pub auto_gain: FloatParam,
    #[id = "loss_gain"]
    pub loss_gain: FloatParam,

    // --- Output ---
    #[id = "wet_dry"]
    pub wet_dry: FloatParam,
}

impl Default for LossyPluginParams {
    fn default() -> Self {
        Self {
            editor_state: EguiState::from_size(620, 780),

            // --- Spectral Loss ---
            mode: EnumParam::new("Mode", SpectralMode::Standard),
            jitter: FloatParam::new("Jitter", 0.0, FloatRange::Linear { min: 0.0, max: 1.0 }),
            loss: FloatParam::new("Loss", 0.5, FloatRange::Linear { min: 0.0, max: 1.0 }),
            window_size: IntParam::new("Window Size", 2048, IntRange::Linear { min: 64, max: 16384 }),
            hop_divisor: IntParam::new("Hop Divisor", 4, IntRange::Linear { min: 1, max: 8 }),
            n_bands: IntParam::new("Bands", 21, IntRange::Linear { min: 2, max: 64 }),
            global_amount: FloatParam::new("Global", 1.0, FloatRange::Linear { min: 0.0, max: 1.0 }),
            phase_loss: FloatParam::new("Phase Loss", 0.0, FloatRange::Linear { min: 0.0, max: 1.0 }),
            quantizer: EnumParam::new("Quantizer", QuantizerType::Uniform),
            pre_echo: FloatParam::new("Pre-Echo", 0.0, FloatRange::Linear { min: 0.0, max: 1.0 }),
            noise_shape: FloatParam::new("Noise Shape", 0.0, FloatRange::Linear { min: 0.0, max: 1.0 }),
            weighting: FloatParam::new("Weighting", 1.0, FloatRange::Linear { min: 0.0, max: 1.0 }),
            hf_threshold: FloatParam::new("HF Threshold", 0.3, FloatRange::Linear { min: 0.0, max: 1.0 }),
            transient_ratio: FloatParam::new("Transient Thr", 4.0, FloatRange::Linear { min: 1.5, max: 20.0 }),
            slushy_rate: FloatParam::new("Slushy Rate", 0.03, FloatRange::Linear { min: 0.001, max: 0.5 }),

            // --- Crush ---
            crush: FloatParam::new("Crush", 0.0, FloatRange::Linear { min: 0.0, max: 1.0 }),
            decimate: FloatParam::new("Decimate", 0.0, FloatRange::Linear { min: 0.0, max: 1.0 }),

            // --- Packets ---
            packets: EnumParam::new("Packets", PacketMode::Off),
            packet_rate: FloatParam::new("Pkt Rate", 0.3, FloatRange::Linear { min: 0.0, max: 1.0 }),
            packet_size: FloatParam::new("Pkt Size", 30.0, FloatRange::Linear { min: 5.0, max: 200.0 })
                .with_unit(" ms"),

            // --- Filter ---
            filter_type: EnumParam::new("Filter", FilterType::Bypass),
            filter_freq: FloatParam::new(
                "Freq",
                1000.0,
                FloatRange::Skewed {
                    min: 20.0,
                    max: 20000.0,
                    factor: FloatRange::skew_factor(-2.0),
                },
            )
            .with_unit(" Hz"),
            filter_width: FloatParam::new("Width", 0.5, FloatRange::Linear { min: 0.0, max: 1.0 }),
            filter_slope: EnumParam::new("Slope", FilterSlope::Slope24),

            // --- Reverb ---
            verb: FloatParam::new("Verb", 0.0, FloatRange::Linear { min: 0.0, max: 1.0 }),
            decay: FloatParam::new("Decay", 0.5, FloatRange::Linear { min: 0.0, max: 1.0 }),
            verb_position: EnumParam::new("Verb Pos", VerbPosition::Pre),

            // --- Freeze ---
            freeze: BoolParam::new("Freeze", false),
            freeze_mode: EnumParam::new("Freeze Mode", FreezeMode::Slushy),
            freezer: FloatParam::new("Freezer", 1.0, FloatRange::Linear { min: 0.0, max: 1.0 }),

            // --- Gate ---
            gate: FloatParam::new("Gate", 0.0, FloatRange::Linear { min: 0.0, max: 1.0 }),

            // --- Hidden ---
            threshold: FloatParam::new("Threshold", 0.5, FloatRange::Linear { min: 0.0, max: 1.0 }),
            auto_gain: FloatParam::new("Auto Gain", 0.0, FloatRange::Linear { min: 0.0, max: 1.0 }),
            loss_gain: FloatParam::new("Loss Gain", 0.5, FloatRange::Linear { min: 0.0, max: 1.0 }),

            // --- Output ---
            wet_dry: FloatParam::new("Wet/Dry", 1.0, FloatRange::Linear { min: 0.0, max: 1.0 }),
        }
    }
}

impl LossyPluginParams {
    /// Convert current nih-plug param values to a lossy-dsp LossyParams struct.
    pub fn to_dsp_params(&self) -> lossy_dsp::LossyParams {
        lossy_dsp::LossyParams {
            inverse: self.mode.value() as i32,
            jitter: self.jitter.value() as f64,
            loss: self.loss.value() as f64,
            window_size: self.window_size.value(),
            hop_divisor: self.hop_divisor.value(),
            n_bands: self.n_bands.value(),
            global_amount: self.global_amount.value() as f64,
            phase_loss: self.phase_loss.value() as f64,
            quantizer: self.quantizer.value() as i32,
            pre_echo: self.pre_echo.value() as f64,
            noise_shape: self.noise_shape.value() as f64,
            weighting: self.weighting.value() as f64,
            hf_threshold: self.hf_threshold.value() as f64,
            transient_ratio: self.transient_ratio.value() as f64,
            slushy_rate: self.slushy_rate.value() as f64,
            crush: self.crush.value() as f64,
            decimate: self.decimate.value() as f64,
            packets: self.packets.value() as i32,
            packet_rate: self.packet_rate.value() as f64,
            packet_size: self.packet_size.value() as f64,
            filter_type: self.filter_type.value() as i32,
            filter_freq: self.filter_freq.value() as f64,
            filter_width: self.filter_width.value() as f64,
            filter_slope: match self.filter_slope.value() {
                FilterSlope::Slope6 => 6,
                FilterSlope::Slope24 => 24,
                FilterSlope::Slope96 => 96,
            },
            verb: self.verb.value() as f64,
            decay: self.decay.value() as f64,
            verb_position: self.verb_position.value() as i32,
            freeze: if self.freeze.value() { 1 } else { 0 },
            freeze_mode: self.freeze_mode.value() as i32,
            freezer: self.freezer.value() as f64,
            gate: self.gate.value() as f64,
            threshold: self.threshold.value() as f64,
            auto_gain: self.auto_gain.value() as f64,
            loss_gain: self.loss_gain.value() as f64,
            bounce: 0,
            bounce_target: 0,
            bounce_rate: 0.3,
            bounce_lfo_min: 0.1,
            bounce_lfo_max: 5.0,
            wet_dry: self.wet_dry.value() as f64,
            seed: 42,
            meta: None,
        }
    }
}
