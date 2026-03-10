/// Reverb app data model — params, render, file I/O, playback.
use audio_engine::analysis;
use audio_engine::buffer::AudioBuffer;
use audio_engine::playback::PlayHandle;
use audio_engine::safety;
use reverb_dsp::ReverbParams;
use shared_gui::app_shell::AppEvent;
use std::cell::RefCell;
use vizia::prelude::*;

thread_local! {
    static PLAY_HANDLE: RefCell<Option<PlayHandle>> = const { RefCell::new(None) };
}

/// Reverb-specific app events.
#[derive(Debug, Clone)]
pub enum ReverbAppEvent {
    SetParam(String, serde_json::Value),
    LoadParams(String),
    Randomize,
    Reset,
}

/// Reverb-specific app data.
#[derive(Clone, Lens)]
pub struct ReverbAppData {
    pub params: ReverbParams,
    pub source_audio: Option<AudioBuffer>,
}

impl ReverbAppData {
    pub fn new() -> Self {
        Self {
            params: ReverbParams::default(),
            source_audio: None,
        }
    }
}

impl Model for ReverbAppData {
    fn event(&mut self, cx: &mut EventContext, event: &mut Event) {
        event.map(|e, _| match e {
            ReverbAppEvent::LoadParams(json) => {
                if let Ok(p) = ReverbParams::from_json(json) {
                    self.params = p;
                    self.params.normalize();
                }
            }
            ReverbAppEvent::Reset => {
                self.params = ReverbParams::default();
            }
            ReverbAppEvent::Randomize => {
                randomize_params(&mut self.params);
            }
            ReverbAppEvent::SetParam(key, value) => {
                set_param(&mut self.params, key, value);
            }
        });

        event.map(|e, _| match e {
            AppEvent::WavLoaded(audio) => {
                self.source_audio = Some(audio.clone());
            }

            AppEvent::LoadWav => {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("WAV files", &["wav", "WAV"])
                    .pick_file()
                {
                    match audio_engine::wav::load_wav(&path, 44100) {
                        Ok(audio) => {
                            cx.emit(AppEvent::WavLoaded(audio));
                        }
                        Err(e) => {
                            cx.emit(AppEvent::RenderError(e.to_string()));
                        }
                    }
                }
            }

            AppEvent::SaveWav => {
                if let Some(state) = cx.data::<shared_gui::app_shell::AppState>() {
                    if let Some(audio) = &state.rendered_audio {
                        let audio = audio.clone();
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("WAV files", &["wav"])
                            .set_file_name("output.wav")
                            .save_file()
                        {
                            match audio_engine::wav::save_wav(&path, &audio) {
                                Ok(()) => {
                                    cx.emit(AppEvent::SetStatus(format!(
                                        "Saved: {}",
                                        path.display()
                                    )));
                                }
                                Err(e) => {
                                    cx.emit(AppEvent::RenderError(e.to_string()));
                                }
                            }
                        }
                    }
                }
            }

            AppEvent::Render => {
                let source = self
                    .source_audio
                    .clone()
                    .unwrap_or_else(|| audio_engine::wav::make_impulse(44100, 2.0));
                let params = self.params.clone();
                match render_sync(&params, &source) {
                    Ok((audio, metrics, warning)) => {
                        cx.emit(AppEvent::RenderComplete {
                            audio,
                            metrics,
                            warning,
                        });
                    }
                    Err(msg) => {
                        cx.emit(AppEvent::RenderError(msg));
                    }
                }
            }

            AppEvent::Play => {
                if let Some(state) = cx.data::<shared_gui::app_shell::AppState>() {
                    if let Some(audio) = &state.rendered_audio {
                        let left = audio.left();
                        let right = audio.right();
                        if let Err(e) = start_playback(&left, &right, audio.sample_rate) {
                            cx.emit(AppEvent::SetStatus(format!("Playback error: {e}")));
                            cx.emit(AppEvent::Stop);
                        }
                    }
                }
            }

            AppEvent::PlayDry => {
                if let Some(source) = &self.source_audio {
                    let left = source.left();
                    let right = source.right();
                    if let Err(e) = start_playback(&left, &right, source.sample_rate) {
                        cx.emit(AppEvent::SetStatus(format!("Playback error: {e}")));
                        cx.emit(AppEvent::Stop);
                    }
                }
            }

            AppEvent::Stop => {
                stop_playback();
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
        });
    }
}

fn stop_playback() {
    PLAY_HANDLE.with(|h| {
        if let Some(handle) = h.borrow_mut().take() {
            handle.stop();
        }
    });
}

fn start_playback(left: &[f64], right: &[f64], sr: u32) -> Result<(), String> {
    stop_playback();
    let handle =
        audio_engine::playback::play(left, right, sr, None).map_err(|e| e.to_string())?;
    PLAY_HANDLE.with(|h| {
        *h.borrow_mut() = Some(handle);
    });
    Ok(())
}

/// Synchronous render.
fn render_sync(
    params: &ReverbParams,
    source: &AudioBuffer,
) -> Result<(AudioBuffer, audio_engine::analysis::AudioMetrics, String), String> {
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

fn randomize_params(params: &mut ReverbParams) {
    use rand::Rng;
    let mut rng = rand::rng();

    params.feedback_gain = rng.random_range(0.3..0.98);
    params.wet_dry = rng.random_range(0.2..1.0);
    params.diffusion = rng.random_range(0.0..1.0);
    params.saturation = rng.random_range(0.0..0.5);
    params.stereo_width = rng.random_range(0.3..1.0);
    params.pre_delay = rng.random_range(0..2000);

    for i in 0..8 {
        params.delay_times[i] = rng.random_range(500..5000);
        params.damping_coeffs[i] = rng.random_range(0.0..0.8);
        params.output_gains[i] = rng.random_range(0.3..1.5);
    }
}

fn set_param(params: &mut ReverbParams, key: &str, value: &serde_json::Value) {
    match key {
        "feedback_gain" => {
            if let Some(v) = value.as_f64() {
                params.feedback_gain = v;
            }
        }
        "wet_dry" => {
            if let Some(v) = value.as_f64() {
                params.wet_dry = v;
            }
        }
        "diffusion" => {
            if let Some(v) = value.as_f64() {
                params.diffusion = v;
            }
        }
        "saturation" => {
            if let Some(v) = value.as_f64() {
                params.saturation = v;
            }
        }
        "stereo_width" => {
            if let Some(v) = value.as_f64() {
                params.stereo_width = v;
            }
        }
        "pre_delay" => {
            if let Some(v) = value.as_i64() {
                params.pre_delay = v as i32;
            }
        }
        "mod_master_rate" => {
            if let Some(v) = value.as_f64() {
                params.mod_master_rate = v;
            }
        }
        "mod_correlation" => {
            if let Some(v) = value.as_f64() {
                params.mod_correlation = v;
            }
        }
        _ => {}
    }
}
