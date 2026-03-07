//! Lossy codec emulator — nih-plug VST3/CLAP/standalone plugin.
//!
//! Wraps the lossy-dsp crate in a real-time plugin with egui GUI.
//! Uses pre-allocated `StereoLossyProcessor` for zero audio-thread allocations.
//! DSP state (reverb comb filters, biquad state) persists across process calls.

mod gui;
mod params;
pub mod presets;

use nih_plug::prelude::*;
use lossy_dsp::StereoLossyProcessor;
use std::sync::Arc;

use params::LossyPluginParams;

pub struct LossyPlugin {
    params: Arc<LossyPluginParams>,
    sample_rate: f32,
    /// Pre-allocated stereo lossy processor — zero allocations after init.
    processor: StereoLossyProcessor,
    /// Pre-allocated f64 input buffers for f32→f64 conversion.
    input_l: Vec<f64>,
    input_r: Vec<f64>,
    /// Pre-allocated f64 output buffers.
    output_l: Vec<f64>,
    output_r: Vec<f64>,
}

/// Maximum buffer size we'll see from a host (pre-allocate for this).
const MAX_BUFFER_SIZE: usize = 8192;

impl Default for LossyPlugin {
    fn default() -> Self {
        Self {
            params: Arc::new(LossyPluginParams::default()),
            sample_rate: 44100.0,
            processor: StereoLossyProcessor::new(),
            input_l: vec![0.0; MAX_BUFFER_SIZE],
            input_r: vec![0.0; MAX_BUFFER_SIZE],
            output_l: vec![0.0; MAX_BUFFER_SIZE],
            output_r: vec![0.0; MAX_BUFFER_SIZE],
        }
    }
}

impl Plugin for LossyPlugin {
    const NAME: &'static str = "Lossy";
    const VENDOR: &'static str = "reverb-project";
    const URL: &'static str = "";
    const EMAIL: &'static str = "";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[
        // Stereo
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(2),
            main_output_channels: NonZeroU32::new(2),
            ..AudioIOLayout::const_default()
        },
        // Mono input -> stereo output
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(1),
            main_output_channels: NonZeroU32::new(2),
            ..AudioIOLayout::const_default()
        },
    ];

    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn initialize(
        &mut self,
        _layout: &AudioIOLayout,
        config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        self.sample_rate = config.sample_rate;

        // Pre-allocate for maximum buffer size the host will use
        let max_buf = config.max_buffer_size as usize;
        if self.input_l.len() < max_buf {
            self.input_l.resize(max_buf, 0.0);
            self.input_r.resize(max_buf, 0.0);
            self.output_l.resize(max_buf, 0.0);
            self.output_r.resize(max_buf, 0.0);
        }

        // No added latency — state persists across process calls
        true
    }

    fn reset(&mut self) {
        self.processor.reset();
    }

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        gui::create(self.params.clone())
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let num_samples = buffer.samples();

        // Build DSP params from GUI (no Vecs, allocation-free)
        let dsp_params = self.params.to_dsp_params();

        // Convert f32 input to f64
        let channel_slices = buffer.as_slice();
        let n_channels = channel_slices.len();
        for i in 0..num_samples {
            self.input_l[i] = channel_slices[0][i] as f64;
            self.input_r[i] = if n_channels > 1 {
                channel_slices[1][i] as f64
            } else {
                channel_slices[0][i] as f64
            };
        }

        // Process through pre-allocated stereo processor
        self.processor.process_stereo(
            &self.input_l[..num_samples],
            &self.input_r[..num_samples],
            &dsp_params,
            &mut self.output_l[..num_samples],
            &mut self.output_r[..num_samples],
        );

        // Convert f64 output back to f32
        let channel_slices = buffer.as_slice();
        for i in 0..num_samples {
            channel_slices[0][i] = self.output_l[i] as f32;
            if n_channels > 1 {
                channel_slices[1][i] = self.output_r[i] as f32;
            }
        }

        ProcessStatus::Normal
    }
}

impl ClapPlugin for LossyPlugin {
    const CLAP_ID: &'static str = "com.reverb-project.lossy";
    const CLAP_DESCRIPTION: Option<&'static str> = Some("Spectral codec emulator");
    const CLAP_MANUAL_URL: Option<&'static str> = None;
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Stereo,
        ClapFeature::Distortion,
    ];
}

impl Vst3Plugin for LossyPlugin {
    const VST3_CLASS_ID: [u8; 16] = *b"LossyCodecEmul!_";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Fx, Vst3SubCategory::Distortion];
}

nih_export_clap!(LossyPlugin);
nih_export_vst3!(LossyPlugin);
