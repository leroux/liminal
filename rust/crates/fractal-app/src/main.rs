mod data;

use data::FractalAppData;
use shared_gui::app_shell::{self, AppEvent, AppState};
use shared_gui::guide_view;
use shared_gui::preset_browser::{self, PresetBrowserData};
use shared_gui::theme;
use shared_gui::waveform_view::WaveformView;
use shared_gui::spectrogram_view::SpectrogramView;
use shared_gui::spectrum_view::SpectrumView;
use vizia::prelude::*;

const TAB_NAMES: &[&str] = &[
    "Parameters",
    "Presets",
    "Waveforms",
    "Spectrograms",
    "Spectrum",
    "Guide",
];

fn main() -> Result<(), ApplicationError> {
    let preset_dir = find_preset_dir();

    Application::new(move |cx| {
        let css: &'static str =
            Box::leak(theme::theme_css(&theme::FRACTAL_THEME).into_boxed_str());
        cx.add_stylesheet(css).expect("Failed to load theme CSS");

        AppState::default().build(cx);
        FractalAppData::new().build(cx);
        PresetBrowserData::new(preset_dir.clone()).build(cx);

        let impulse = audio_engine::wav::make_impulse(44100, 2.0);
        cx.emit(AppEvent::WavLoaded(impulse));
        cx.emit(AppEvent::Render);

        VStack::new(cx, |cx| {
            app_shell::toolbar_view(cx);
            app_shell::tab_bar_view(cx, TAB_NAMES);

            Binding::new(cx, AppState::active_tab, |cx, tab| {
                let tab = tab.get(cx);
                VStack::new(cx, move |cx| {
                    match tab {
                        0 => {
                            Label::new(cx, "Fractal parameters -- coming soon")
                                .class("dim");
                        }
                        1 => preset_browser::preset_browser_view(cx),
                        2 => {
                            WaveformView::new(cx);
                        }
                        3 => {
                            SpectrogramView::new(cx);
                        }
                        4 => {
                            SpectrumView::new(cx);
                        }
                        5 => guide_view::guide_view(cx, guide_view::FRACTAL_GUIDE),
                        _ => {}
                    }
                })
                .height(Stretch(1.0));
            });

            app_shell::status_bar_view(cx);
        })
        .class("app-root");
    })
    .title("Fractal - Audio Fractalization Effect")
    .inner_size((1250, 950))
    .run()
}

fn find_preset_dir() -> std::path::PathBuf {
    for c in &[
        "fractal/gui/presets",
        "../../fractal/gui/presets",
        "../../../fractal/gui/presets",
    ] {
        let p = std::path::PathBuf::from(c);
        if p.exists() {
            return p;
        }
    }
    std::path::PathBuf::from("presets")
}
