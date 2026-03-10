/// Fractal app data model — params, render, file I/O, playback.
use audio_engine::analysis;
use audio_engine::buffer::AudioBuffer;
use audio_engine::playback::PlayHandle;
use audio_engine::safety;
use fractal_dsp::FractalParams;
use shared_gui::app_shell::AppEvent;
use std::cell::RefCell;
use vizia::prelude::*;

thread_local! {
    static PLAY_HANDLE: RefCell<Option<PlayHandle>> = const { RefCell::new(None) };
}

#[derive(Debug, Clone)]
pub enum FractalAppEvent {
    SetParam(String, serde_json::Value),
    LoadParams(String),
    Reset,
}

#[derive(Clone, Lens)]
pub struct FractalAppData {
    pub params: FractalParams,
    pub source_audio: Option<AudioBuffer>,
}

impl FractalAppData {
    pub fn new() -> Self {
        Self {
            params: FractalParams::default(),
            source_audio: None,
        }
    }
}

impl Model for FractalAppData {
    fn event(&mut self, cx: &mut EventContext, event: &mut Event) {
        event.map(|e, _| match e {
            FractalAppEvent::LoadParams(json) => {
                if let Ok(p) = FractalParams::from_json(json) {
                    self.params = p;
                }
            }
            FractalAppEvent::Reset => {
                self.params = FractalParams::default();
            }
            FractalAppEvent::SetParam(_key, _value) => {
                // TODO: individual param setting
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
                        Ok(audio) => cx.emit(AppEvent::WavLoaded(audio)),
                        Err(e) => cx.emit(AppEvent::RenderError(e.to_string())),
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
                                Ok(()) => cx.emit(AppEvent::SetStatus(format!(
                                    "Saved: {}",
                                    path.display()
                                ))),
                                Err(e) => cx.emit(AppEvent::RenderError(e.to_string())),
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
                    Err(msg) => cx.emit(AppEvent::RenderError(msg)),
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

            AppEvent::Reset => {
                cx.emit(FractalAppEvent::Reset);
            }
            AppEvent::PresetLoaded(json) => {
                cx.emit(FractalAppEvent::LoadParams(json.clone()));
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

fn render_sync(
    params: &FractalParams,
    source: &AudioBuffer,
) -> Result<(AudioBuffer, audio_engine::analysis::AudioMetrics, String), String> {
    let mono = source.to_mono();

    let output = fractal_dsp::render_fractal(&mono, params);

    safety::safety_check(&output)?;

    let (normalized, warning) = safety::normalize_output(&output, 0.2, 0.9);

    let audio = AudioBuffer::from_mono(normalized, source.sample_rate);
    let mono_out = audio.to_mono();
    let source_mono = source.to_mono();
    let metrics = analysis::analyze(&mono_out, source.sample_rate, Some(&source_mono));

    Ok((audio, metrics, warning))
}
