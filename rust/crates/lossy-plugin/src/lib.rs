//! Lossy codec emulator — nih-plug VST3/CLAP/standalone plugin.
//!
//! Wraps the lossy-dsp crate in a real-time plugin with egui GUI.
//! Uses a collect-and-process strategy: input samples accumulate in a ring buffer,
//! and when enough arrive, the entire chain processes a block. The plugin reports
//! latency equal to the processing block size.

mod gui;
mod params;
pub mod presets;

use nih_plug::prelude::*;
use std::sync::Arc;

use params::LossyPluginParams;

/// Processing block size. We collect this many samples before running the chain.
/// Larger = more latency but allows the STFT engine to work efficiently.
/// At 44100 Hz, 8192 samples = ~186ms latency.
const PROCESS_BLOCK: usize = 8192;

pub struct LossyPlugin {
    params: Arc<LossyPluginParams>,
    sample_rate: f32,
    /// Per-channel input accumulation buffer.
    input_buf: [Vec<f64>; 2],
    /// Per-channel output drain buffer (processed audio waiting to be sent).
    output_buf: [Vec<f64>; 2],
    /// Write position into input_buf.
    in_pos: usize,
    /// Read position from output_buf.
    out_pos: usize,
    /// How many valid samples in output_buf from out_pos onward.
    out_avail: usize,
    /// True once we've filled the first block (latency compensation).
    primed: bool,
}

impl Default for LossyPlugin {
    fn default() -> Self {
        Self {
            params: Arc::new(LossyPluginParams::default()),
            sample_rate: 44100.0,
            input_buf: [vec![0.0; PROCESS_BLOCK], vec![0.0; PROCESS_BLOCK]],
            output_buf: [vec![0.0; PROCESS_BLOCK], vec![0.0; PROCESS_BLOCK]],
            in_pos: 0,
            out_pos: 0,
            out_avail: 0,
            primed: false,
        }
    }
}

impl LossyPlugin {
    /// Build a LossyParams from current nih-plug parameter values.
    fn build_dsp_params(&self) -> lossy_dsp::LossyParams {
        self.params.to_dsp_params()
    }

    /// Process one block through the lossy chain.
    fn process_block(&mut self) {
        let dsp_params = self.build_dsp_params();

        let (out_l, out_r) = lossy_dsp::render_lossy_stereo(
            &self.input_buf[0],
            &self.input_buf[1],
            &dsp_params,
        );

        self.output_buf[0] = out_l;
        self.output_buf[1] = out_r;
        self.out_pos = 0;
        self.out_avail = PROCESS_BLOCK;
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
        context.set_latency_samples(PROCESS_BLOCK as u32);
        true
    }

    fn reset(&mut self) {
        for buf in &mut self.input_buf {
            buf.fill(0.0);
        }
        for buf in &mut self.output_buf {
            buf.fill(0.0);
        }
        self.in_pos = 0;
        self.out_pos = 0;
        self.out_avail = 0;
        self.primed = false;
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

        for i in 0..num_samples {
            // Read input sample into accumulation buffer
            let channel_slices = buffer.as_slice();
            for ch in 0..2.min(channel_slices.len()) {
                self.input_buf[ch][self.in_pos] = channel_slices[ch][i] as f64;
            }
            // If mono input, duplicate to right channel
            if channel_slices.len() == 1 {
                self.input_buf[1][self.in_pos] = channel_slices[0][i] as f64;
            }
            self.in_pos += 1;

            // When input buffer is full, process it
            if self.in_pos >= PROCESS_BLOCK {
                self.process_block();
                self.in_pos = 0;
                self.primed = true;
            }

            // Write output (or silence if not yet primed)
            if self.primed && self.out_avail > 0 {
                let channel_slices = buffer.as_slice();
                for ch in 0..2.min(channel_slices.len()) {
                    channel_slices[ch][i] = self.output_buf[ch][self.out_pos] as f32;
                }
                self.out_pos += 1;
                self.out_avail -= 1;
            } else {
                let channel_slices = buffer.as_slice();
                for ch in 0..channel_slices.len() {
                    channel_slices[ch][i] = 0.0;
                }
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
