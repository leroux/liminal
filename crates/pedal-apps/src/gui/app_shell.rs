/// Shared app scaffold — toolbar, tabs, render thread, playback, history.
use crate::audio::analysis::AudioMetrics;
use crate::audio::buffer::AudioBuffer;
use vizia::prelude::*;

/// Generation history entry.
#[derive(Debug, Clone)]
pub struct HistoryEntry {
    pub params_json: String,
    pub audio: Option<AudioBuffer>,
    pub metrics: Option<AudioMetrics>,
}

/// App-level events shared across all pedal apps.
#[derive(Debug, Clone)]
pub enum AppEvent {
    LoadWav,
    SaveWav,
    WavLoaded(AudioBuffer),
    Render,
    RenderComplete {
        audio: AudioBuffer,
        metrics: Box<AudioMetrics>,
        warning: String,
    },
    RenderError(String),
    Play,
    PlayDry,
    Stop,
    Randomize,
    Reset,
    ParamChanged,
    PresetLoaded(String),
    HistoryBack,
    HistoryForward,
    SelectTab(usize),
    SetStatus(String),
}

/// Shared app state model.
#[derive(Clone, Lens)]
pub struct AppState {
    pub source_audio: Option<AudioBuffer>,
    pub rendered_audio: Option<AudioBuffer>,
    pub dry_audio: Option<AudioBuffer>,
    pub metrics: Option<AudioMetrics>,
    pub params_json: String,
    pub status: String,
    pub is_rendering: bool,
    pub is_playing: bool,
    pub active_tab: usize,
    pub history: Vec<HistoryEntry>,
    pub history_index: usize,
    pub warning: String,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            source_audio: None,
            rendered_audio: None,
            dry_audio: None,
            metrics: None,
            params_json: String::new(),
            status: "Ready".to_string(),
            is_rendering: false,
            is_playing: false,
            active_tab: 0,
            history: Vec::new(),
            history_index: 0,
            warning: String::new(),
        }
    }
}

impl Model for AppState {
    fn event(&mut self, _cx: &mut EventContext, event: &mut Event) {
        event.map(|e, _| match e {
            AppEvent::LoadWav => {
                self.status = "Loading WAV...".to_string();
            }
            AppEvent::WavLoaded(audio) => {
                self.source_audio = Some(audio.clone());
                self.status = format!(
                    "Loaded: {:.1}s, {}ch, {}Hz",
                    audio.duration_seconds(),
                    audio.channels,
                    audio.sample_rate
                );
            }
            AppEvent::Render => {
                if !self.is_rendering {
                    self.is_rendering = true;
                    self.status = "Rendering...".to_string();
                }
            }
            AppEvent::RenderComplete {
                audio,
                metrics,
                warning,
            } => {
                self.rendered_audio = Some(audio.clone());
                self.metrics = Some(*metrics.clone());
                self.warning = warning.clone();
                self.is_rendering = false;

                self.history.truncate(self.history_index + 1);
                self.history.push(HistoryEntry {
                    params_json: self.params_json.clone(),
                    audio: Some(audio.clone()),
                    metrics: Some(*metrics.clone()),
                });
                if self.history.len() > 50 {
                    self.history.remove(0);
                }
                self.history_index = self.history.len() - 1;

                let dur = audio.duration_seconds();
                self.status = format!("Rendered: {dur:.2}s{warning}");
            }
            AppEvent::RenderError(msg) => {
                self.is_rendering = false;
                self.status = format!("Error: {msg}");
            }
            AppEvent::Play | AppEvent::PlayDry => {
                self.is_playing = true;
                self.status = "Playing...".to_string();
            }
            AppEvent::Stop => {
                self.is_playing = false;
                self.status = "Stopped".to_string();
            }
            AppEvent::SelectTab(idx) => {
                self.active_tab = *idx;
            }
            AppEvent::SetStatus(msg) => {
                self.status = msg.clone();
            }
            AppEvent::HistoryBack => {
                if self.history_index > 0 {
                    self.history_index -= 1;
                    if let Some(entry) = self.history.get(self.history_index) {
                        self.params_json = entry.params_json.clone();
                        self.rendered_audio = entry.audio.clone();
                        self.metrics = entry.metrics.clone();
                        self.status = format!(
                            "History: {}/{}",
                            self.history_index + 1,
                            self.history.len()
                        );
                    }
                }
            }
            AppEvent::HistoryForward => {
                if self.history_index + 1 < self.history.len() {
                    self.history_index += 1;
                    if let Some(entry) = self.history.get(self.history_index) {
                        self.params_json = entry.params_json.clone();
                        self.rendered_audio = entry.audio.clone();
                        self.metrics = entry.metrics.clone();
                        self.status = format!(
                            "History: {}/{}",
                            self.history_index + 1,
                            self.history.len()
                        );
                    }
                }
            }
            _ => {}
        });
    }
}

/// Build the shared toolbar.
pub fn toolbar_view(cx: &mut Context) {
    HStack::new(cx, |cx| {
        Button::new(cx, |cx| Label::new(cx, "Load WAV"))
            .on_press(|cx| cx.emit(AppEvent::LoadWav));
        Button::new(cx, |cx| Label::new(cx, "Save WAV"))
            .on_press(|cx| cx.emit(AppEvent::SaveWav));

        Element::new(cx).width(Pixels(16.0));

        Button::new(cx, |cx| Label::new(cx, "Play"))
            .on_press(|cx| cx.emit(AppEvent::Play));
        Button::new(cx, |cx| Label::new(cx, "Dry"))
            .on_press(|cx| cx.emit(AppEvent::PlayDry));
        Button::new(cx, |cx| Label::new(cx, "Stop"))
            .on_press(|cx| cx.emit(AppEvent::Stop));

        Element::new(cx).width(Pixels(16.0));

        Button::new(cx, |cx| Label::new(cx, "Render"))
            .on_press(|cx| cx.emit(AppEvent::Render));
        Button::new(cx, |cx| Label::new(cx, "Random"))
            .on_press(|cx| cx.emit(AppEvent::Randomize));
        Button::new(cx, |cx| Label::new(cx, "Reset"))
            .on_press(|cx| cx.emit(AppEvent::Reset));

        Element::new(cx).width(Stretch(1.0));

        Button::new(cx, |cx| Label::new(cx, "<"))
            .on_press(|cx| cx.emit(AppEvent::HistoryBack));
        Button::new(cx, |cx| Label::new(cx, ">"))
            .on_press(|cx| cx.emit(AppEvent::HistoryForward));
    })
    .class("toolbar")
    .horizontal_gap(Pixels(4.0))
    .padding(Pixels(4.0));
}

/// Build the tab bar from tab names.
pub fn tab_bar_view(cx: &mut Context, tab_names: &[&'static str]) {
    let names: Vec<&'static str> = tab_names.to_vec();
    HStack::new(cx, move |cx| {
        for (i, name) in names.iter().enumerate() {
            let idx = i;
            let label_text: &'static str = name;
            Button::new(cx, move |cx| Label::new(cx, label_text))
                .class("tab-header")
                .checked(AppState::active_tab.map(move |t| *t == idx))
                .on_press(move |cx| cx.emit(AppEvent::SelectTab(idx)));
        }
    })
    .class("tab-bar")
    .horizontal_gap(Pixels(2.0));
}

/// Build the status bar.
pub fn status_bar_view(cx: &mut Context) {
    HStack::new(cx, |cx| {
        Label::new(cx, AppState::status.map(|s| s.clone())).class("status-text");
    })
    .class("status-bar")
    .padding(Pixels(4.0));
}
