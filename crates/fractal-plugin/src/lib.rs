//! Fractal audio fractalization — nih-plug VST3/CLAP/standalone plugin.
//!
//! Wraps the fractal-dsp crate in a real-time plugin with egui GUI.
//! Uses a collect-and-process strategy: input samples accumulate in a ring buffer,
//! and when enough arrive, the entire chain processes a block. The plugin reports
//! latency equal to the processing block size.

mod gui;
mod params;
pub mod presets;

use nih_plug::prelude::*;
use std::sync::Arc;

use fractal_dsp::processor::StereoFractalProcessor;
use params::FractalPluginParams;

pub struct FractalPlugin {
    params: Arc<FractalPluginParams>,
    sample_rate: f32,
    processor: StereoFractalProcessor,
    /// Per-sample conversion buffers (f32 ↔ f64).
    in_l: Vec<f64>,
    in_r: Vec<f64>,
    out_l: Vec<f64>,
    out_r: Vec<f64>,
}

const MAX_BUFFER_SIZE: usize = 8192;

impl Default for FractalPlugin {
    fn default() -> Self {
        Self {
            params: Arc::new(FractalPluginParams::default()),
            sample_rate: 44100.0,
            processor: StereoFractalProcessor::new(),
            in_l: vec![0.0; MAX_BUFFER_SIZE],
            in_r: vec![0.0; MAX_BUFFER_SIZE],
            out_l: vec![0.0; MAX_BUFFER_SIZE],
            out_r: vec![0.0; MAX_BUFFER_SIZE],
        }
    }
}

impl Plugin for FractalPlugin {
    const NAME: &'static str = "Fractal";
    const VENDOR: &'static str = "reverb-project";
    const URL: &'static str = "";
    const EMAIL: &'static str = "";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(2),
            main_output_channels: NonZeroU32::new(2),
            ..AudioIOLayout::const_default()
        },
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
        context: &mut impl InitContext<Self>,
    ) -> bool {
        self.sample_rate = config.sample_rate;
        context.set_latency_samples(self.processor.latency() as u32);
        let max_buf = config.max_buffer_size as usize;
        if self.in_l.len() < max_buf {
            self.in_l.resize(max_buf, 0.0);
            self.in_r.resize(max_buf, 0.0);
            self.out_l.resize(max_buf, 0.0);
            self.out_r.resize(max_buf, 0.0);
        }
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
        let dsp_params = self.params.to_dsp_params();

        // Convert f32 input to f64
        let channel_slices = buffer.as_slice();
        let num_channels = channel_slices.len();
        for i in 0..num_samples {
            self.in_l[i] = channel_slices[0][i] as f64;
            self.in_r[i] = if num_channels > 1 {
                channel_slices[1][i] as f64
            } else {
                channel_slices[0][i] as f64
            };
        }

        // Process through the shared processor
        self.processor.process(
            &self.in_l[..num_samples],
            &self.in_r[..num_samples],
            &dsp_params,
            &mut self.out_l[..num_samples],
            &mut self.out_r[..num_samples],
        );

        // Write f64 output back to f32 buffer
        let channel_slices = buffer.as_slice();
        for i in 0..num_samples {
            channel_slices[0][i] = self.out_l[i] as f32;
            if num_channels > 1 {
                channel_slices[1][i] = self.out_r[i] as f32;
            }
        }

        ProcessStatus::Normal
    }
}

impl ClapPlugin for FractalPlugin {
    const CLAP_ID: &'static str = "com.reverb-project.fractal";
    const CLAP_DESCRIPTION: Option<&'static str> = Some("Audio fractalization effect");
    const CLAP_MANUAL_URL: Option<&'static str> = None;
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Stereo,
        ClapFeature::Distortion,
    ];
}

impl Vst3Plugin for FractalPlugin {
    const VST3_CLASS_ID: [u8; 16] = *b"FractalAudioFx!_";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Fx, Vst3SubCategory::Distortion];
}

nih_export_clap!(FractalPlugin);
nih_export_vst3!(FractalPlugin);
