/// Lossy app data model — params, render, file I/O, playback.
use crate::audio::analysis;
use crate::audio::buffer::AudioBuffer;
use crate::audio::safety;
use crate::common;
use crate::gui::app_shell::AppEvent;
use lossy_dsp::LossyParams;
use vizia::prelude::*;

#[derive(Debug, Clone)]
pub enum LossyAppEvent {
    SetParam(String, serde_json::Value),
    LoadParams(String),
    Reset,
}

#[derive(Clone, Default, Lens)]
pub struct LossyAppData {
    pub params: LossyParams,
    pub source_audio: Option<AudioBuffer>,
}

impl LossyAppData {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Model for LossyAppData {
    fn event(&mut self, cx: &mut EventContext, event: &mut Event) {
        event.map(|e, _| match e {
            LossyAppEvent::LoadParams(json) => {
                if let Ok(p) = LossyParams::from_json(json) {
                    self.params = p;
                }
            }
            LossyAppEvent::Reset => {
                self.params = LossyParams::default();
            }
            LossyAppEvent::SetParam(_key, _value) => {
                // TODO: individual param setting
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
                        Err(msg) => cx.emit(AppEvent::RenderError(msg)),
                    }
                }
                AppEvent::Reset => {
                    cx.emit(LossyAppEvent::Reset);
                }
                AppEvent::PresetLoaded(json) => {
                    cx.emit(LossyAppEvent::LoadParams(json.clone()));
                    cx.emit(AppEvent::Render);
                }
                _ => {}
            }
        });
    }
}

fn render_sync(
    params: &LossyParams,
    source: &AudioBuffer,
) -> Result<(AudioBuffer, crate::audio::analysis::AudioMetrics, String), String> {
    let mono = source.to_mono();

    let output = lossy_dsp::render_lossy(&mono, params);

    safety::safety_check(&output)?;

    let (normalized, warning) = safety::normalize_output(&output, 0.2, 0.9);

    let audio = AudioBuffer::from_mono(normalized, source.sample_rate);
    let mono_out = audio.to_mono();
    let source_mono = source.to_mono();
    let metrics = analysis::analyze(&mono_out, source.sample_rate, Some(&source_mono));

    Ok((audio, metrics, warning))
}
