/// Reverb app data model — params, render, file I/O, playback.
use crate::audio::analysis;
use crate::audio::buffer::AudioBuffer;
use crate::audio::safety;
use crate::common;
use crate::gui::app_shell::AppEvent;
use reverb_dsp::{FdnParams, ReverbParams};
use vizia::prelude::*;

/// Reverb-specific app events.
#[derive(Debug, Clone)]
pub enum ReverbAppEvent {
    SetParam(String, serde_json::Value),
    LoadParams(String),
    Randomize,
    Reset,
}

/// Reverb-specific app data.
#[derive(Clone, Default, Lens)]
pub struct ReverbAppData {
    pub params: FdnParams,
    pub source_audio: Option<AudioBuffer>,
}

impl ReverbAppData {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Model for ReverbAppData {
    fn event(&mut self, cx: &mut EventContext, event: &mut Event) {
        event.map(|e, _| match e {
            ReverbAppEvent::LoadParams(json) => {
                // Try simplified format first (presets use this),
                // fall back to full FdnParams for backward compat.
                if json.contains("\"size\"") || json.contains("\"brightness\"") {
                    if let Ok(s) = ReverbParams::from_json(json) {
                        self.params = s.to_fdn_params();
                        self.params.normalize();
                        return;
                    }
                }
                if let Ok(p) = FdnParams::from_json(json) {
                    self.params = p;
                    self.params.normalize();
                }
            }
            ReverbAppEvent::Reset => {
                self.params = FdnParams::default();
            }
            ReverbAppEvent::Randomize => {
                randomize_params(&mut self.params);
            }
            ReverbAppEvent::SetParam(key, value) => {
                set_param(&mut self.params, key, value);
            }
        });

        event.map(|e, _| {
            if common::handle_common_app_event(cx, e, &mut self.source_audio) {
                return;
            }
            match e {
                AppEvent::Render => {
                    let source = self
                        .source_audio
                        .clone()
                        .unwrap_or_else(|| crate::audio::wav::make_impulse(44100, 2.0));
                    let params = self.params.clone();
                    match render_sync(&params, &source) {
                        Ok((audio, metrics, warning)) => {
                            cx.emit(AppEvent::RenderComplete {
                                audio,
                                metrics: Box::new(metrics),
                                warning,
                            });
                        }
                        Err(msg) => {
                            cx.emit(AppEvent::RenderError(msg));
                        }
                    }
                }
                AppEvent::Randomize => {
                    cx.emit(ReverbAppEvent::Randomize);
                    cx.emit(AppEvent::Render);
                }
                AppEvent::Reset => {
                    cx.emit(ReverbAppEvent::Reset);
                    cx.emit(AppEvent::Render);
                }
                AppEvent::PresetLoaded(json) => {
                    cx.emit(ReverbAppEvent::LoadParams(json.clone()));
                    cx.emit(AppEvent::Render);
                }
                _ => {}
            }
        });
    }
}

/// Synchronous render.
fn render_sync(
    params: &FdnParams,
    source: &AudioBuffer,
) -> Result<(AudioBuffer, crate::audio::analysis::AudioMetrics, String), String> {
    let mono = source.to_mono();

    let output = reverb_dsp::render_fdn(&mono, params);

    // Output is interleaved stereo [L0, R0, L1, R1, ...]
    let n_frames = output.len() / 2;
    let mut left = Vec::with_capacity(n_frames);
    let mut right = Vec::with_capacity(n_frames);
    for i in 0..n_frames {
        left.push(output[i * 2]);
        right.push(output[i * 2 + 1]);
    }

    safety::safety_check(&output)?;

    let (norm_l, warning) = safety::normalize_output(&left, 0.2, 0.9);
    let (norm_r, _) = safety::normalize_output(&right, 0.2, 0.9);

    let audio = AudioBuffer::from_stereo(&norm_l, &norm_r, source.sample_rate);
    let mono_out = audio.to_mono();
    let source_mono = source.to_mono();
    let metrics = analysis::analyze(&mono_out, source.sample_rate, Some(&source_mono));

    Ok((audio, metrics, warning))
}

fn randomize_params(params: &mut FdnParams) {
    use rand::Rng;
    let mut rng = rand::rng();

    let simplified = ReverbParams {
        size: rng.random_range(0.1..0.9),
        decay: rng.random_range(0.3..0.98),
        brightness: rng.random_range(0.2..1.0),
        diffusion: rng.random_range(0.0..0.7),
        mix: rng.random_range(0.2..1.0),
        saturation: rng.random_range(0.0..0.5),
        pre_delay_ms: rng.random_range(0.0..50.0),
        stereo_width: rng.random_range(0.3..1.0),
        ..ReverbParams::default()
    };
    *params = simplified.to_fdn_params();
}

fn set_param(params: &mut FdnParams, key: &str, value: &serde_json::Value) {
    // Convert current full params to simplified, update the field, convert back.
    // This ensures all derived params (delay_times, damping, node_pans, etc.) stay consistent.
    let mut s = ReverbParams::from_fdn_params(params);
    let applied = match key {
        "size" => value.as_f64().map(|v| s.size = v).is_some(),
        "decay" => value.as_f64().map(|v| s.decay = v).is_some(),
        "brightness" => value.as_f64().map(|v| s.brightness = v).is_some(),
        "diffusion" => value.as_f64().map(|v| s.diffusion = v).is_some(),
        "mix" => value.as_f64().map(|v| s.mix = v).is_some(),
        "saturation" => value.as_f64().map(|v| s.saturation = v).is_some(),
        "pre_delay_ms" => value.as_f64().map(|v| s.pre_delay_ms = v).is_some(),
        "stereo_width" => value.as_f64().map(|v| s.stereo_width = v).is_some(),
        "matrix_type" => value.as_str().map(|v| s.matrix_type = v.to_string()).is_some(),
        "mod_rate" => value.as_f64().map(|v| s.mod_rate = v).is_some(),
        "mod_depth" => value.as_f64().map(|v| s.mod_depth = v).is_some(),
        "mod_character" => value.as_f64().map(|v| s.mod_character = v).is_some(),
        "mod_spread" => value.as_f64().map(|v| s.mod_spread = v).is_some(),
        "mod_waveform" => value.as_i64().map(|v| s.mod_waveform = v as i32).is_some(),
        _ => false,
    };
    if applied {
        *params = s.to_fdn_params();
    }
}
