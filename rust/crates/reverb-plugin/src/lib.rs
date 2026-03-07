//! FDN reverb — nih-plug VST3/CLAP/standalone plugin.
//!
//! Wraps the reverb-dsp crate in a real-time plugin with egui GUI.
//! Uses pre-allocated `StereoFdnProcessor` for zero audio-thread allocations.
//! DSP state (delay lines, filters, LFOs) persists across process calls.

mod gui;
mod params;
pub mod presets;

use nih_plug::prelude::*;
use reverb_dsp::StereoFdnProcessor;
use std::sync::Arc;

use params::ReverbPluginParams;

pub struct ReverbPlugin {
    params: Arc<ReverbPluginParams>,
    sample_rate: f32,
    /// Pre-allocated stereo FDN processor — zero allocations after init.
    processor: StereoFdnProcessor,
    /// Pre-allocated f64 input buffers for f32→f64 conversion.
    input_l: Vec<f64>,
    input_r: Vec<f64>,
    /// Pre-allocated f64 output buffers.
    output_l: Vec<f64>,
    output_r: Vec<f64>,
    /// Cached DSP params — avoids re-allocating Vecs every process call.
    cached_dsp_params: reverb_dsp::ReverbParams,
}

/// Maximum buffer size we'll see from a host (pre-allocate for this).
const MAX_BUFFER_SIZE: usize = 8192;

impl Default for ReverbPlugin {
    fn default() -> Self {
        Self {
            params: Arc::new(ReverbPluginParams::default()),
            sample_rate: 44100.0,
            processor: StereoFdnProcessor::new(),
            input_l: vec![0.0; MAX_BUFFER_SIZE],
            input_r: vec![0.0; MAX_BUFFER_SIZE],
            output_l: vec![0.0; MAX_BUFFER_SIZE],
            output_r: vec![0.0; MAX_BUFFER_SIZE],
            cached_dsp_params: reverb_dsp::ReverbParams::default(),
        }
    }
}

impl ReverbPlugin {
    /// Update cached DSP params from nih-plug params.
    /// Reuses the existing Vecs when possible to avoid allocation.
    fn update_dsp_params(&mut self) {
        let p = &self.params;
        let d = &mut self.cached_dsp_params;

        // Update Vec contents in-place (no reallocation since lengths don't change)
        d.delay_times[0] = p.delay_time_1.value();
        d.delay_times[1] = p.delay_time_2.value();
        d.delay_times[2] = p.delay_time_3.value();
        d.delay_times[3] = p.delay_time_4.value();
        d.delay_times[4] = p.delay_time_5.value();
        d.delay_times[5] = p.delay_time_6.value();
        d.delay_times[6] = p.delay_time_7.value();
        d.delay_times[7] = p.delay_time_8.value();

        d.damping_coeffs[0] = p.damping_1.value() as f64;
        d.damping_coeffs[1] = p.damping_2.value() as f64;
        d.damping_coeffs[2] = p.damping_3.value() as f64;
        d.damping_coeffs[3] = p.damping_4.value() as f64;
        d.damping_coeffs[4] = p.damping_5.value() as f64;
        d.damping_coeffs[5] = p.damping_6.value() as f64;
        d.damping_coeffs[6] = p.damping_7.value() as f64;
        d.damping_coeffs[7] = p.damping_8.value() as f64;

        d.input_gains[0] = p.input_gain_1.value() as f64;
        d.input_gains[1] = p.input_gain_2.value() as f64;
        d.input_gains[2] = p.input_gain_3.value() as f64;
        d.input_gains[3] = p.input_gain_4.value() as f64;
        d.input_gains[4] = p.input_gain_5.value() as f64;
        d.input_gains[5] = p.input_gain_6.value() as f64;
        d.input_gains[6] = p.input_gain_7.value() as f64;
        d.input_gains[7] = p.input_gain_8.value() as f64;

        d.output_gains[0] = p.output_gain_1.value() as f64;
        d.output_gains[1] = p.output_gain_2.value() as f64;
        d.output_gains[2] = p.output_gain_3.value() as f64;
        d.output_gains[3] = p.output_gain_4.value() as f64;
        d.output_gains[4] = p.output_gain_5.value() as f64;
        d.output_gains[5] = p.output_gain_6.value() as f64;
        d.output_gains[6] = p.output_gain_7.value() as f64;
        d.output_gains[7] = p.output_gain_8.value() as f64;

        d.node_pans[0] = p.node_pan_1.value() as f64;
        d.node_pans[1] = p.node_pan_2.value() as f64;
        d.node_pans[2] = p.node_pan_3.value() as f64;
        d.node_pans[3] = p.node_pan_4.value() as f64;
        d.node_pans[4] = p.node_pan_5.value() as f64;
        d.node_pans[5] = p.node_pan_6.value() as f64;
        d.node_pans[6] = p.node_pan_7.value() as f64;
        d.node_pans[7] = p.node_pan_8.value() as f64;

        d.feedback_gain = p.feedback_gain.value() as f64;
        d.wet_dry = p.wet_dry.value() as f64;
        d.diffusion = p.diffusion.value() as f64;
        d.diffusion_stages = p.diffusion_stages.value();
        d.saturation = p.saturation.value() as f64;
        d.pre_delay = p.pre_delay.value();
        d.stereo_width = p.stereo_width.value() as f64;

        let new_type = p.matrix_type.value().to_string_id();
        if d.matrix_type != new_type {
            d.matrix_type = new_type;
        }
        d.matrix_seed = p.matrix_seed.value();

        d.mod_master_rate = p.mod_master_rate.value() as f64;
        d.mod_correlation = p.mod_correlation.value() as f64;
        d.mod_waveform = p.mod_waveform.value() as i32;

        let mod_dd = p.mod_depth_delay.value() as f64;
        let mod_da = p.mod_depth_damping.value() as f64;
        let mod_do = p.mod_depth_output.value() as f64;
        d.mod_depth_delay.fill(mod_dd);
        d.mod_depth_damping.fill(mod_da);
        d.mod_depth_output.fill(mod_do);

        d.mod_depth_matrix = p.mod_depth_matrix.value() as f64;
        d.mod_rate_scale_delay = p.mod_rate_scale_delay.value() as f64;
        d.mod_rate_scale_damping = p.mod_rate_scale_damping.value() as f64;
        d.mod_rate_scale_output = p.mod_rate_scale_output.value() as f64;
        d.mod_rate_matrix = p.mod_rate_matrix.value() as f64;

        let new_mat2_type = p.mod_matrix2_type.value().to_string_id();
        if d.mod_matrix2_type != new_mat2_type {
            d.mod_matrix2_type = new_mat2_type;
        }
        d.mod_matrix2_seed = p.mod_matrix2_seed.value();
    }
}

impl Plugin for ReverbPlugin {
    const NAME: &'static str = "Reverb";
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

        // Update DSP params from GUI (in-place, no allocation)
        self.update_dsp_params();

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

        // Process through pre-allocated stereo FDN
        self.processor.process_stereo(
            &self.input_l[..num_samples],
            &self.input_r[..num_samples],
            &self.cached_dsp_params,
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

impl ClapPlugin for ReverbPlugin {
    const CLAP_ID: &'static str = "com.reverb-project.reverb";
    const CLAP_DESCRIPTION: Option<&'static str> = Some("FDN algorithmic reverb");
    const CLAP_MANUAL_URL: Option<&'static str> = None;
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Stereo,
        ClapFeature::Reverb,
    ];
}

impl Vst3Plugin for ReverbPlugin {
    const VST3_CLASS_ID: [u8; 16] = *b"ReverbFDNPlugin_";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Fx, Vst3SubCategory::Reverb];
}

nih_export_clap!(ReverbPlugin);
nih_export_vst3!(ReverbPlugin);
