use pedal_apps::lossy::data::LossyAppData;
use pedal_apps::gui::app_shell::{self, AppEvent, AppState};
use pedal_apps::gui::guide_view;
use pedal_apps::gui::preset_browser::{self, PresetBrowserData};
use pedal_apps::gui::theme;
use pedal_apps::gui::waveform_view::WaveformView;
use pedal_apps::gui::spectrogram_view::SpectrogramView;
use pedal_apps::gui::spectrum_view::SpectrumView;
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
            Box::leak(theme::theme_css(&theme::LOSSY_THEME).into_boxed_str());
        cx.add_stylesheet(css).expect("Failed to load theme CSS");

        AppState::default().build(cx);
        LossyAppData::new().build(cx);
        PresetBrowserData::new(preset_dir.clone()).build(cx);

        let impulse = pedal_apps::audio::wav::make_impulse(44100, 2.0);
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
                            Label::new(cx, "Lossy parameters -- coming soon")
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
                        5 => guide_view::guide_view(cx, guide_view::LOSSY_GUIDE),
                        _ => {}
                    }
                })
                .height(Stretch(1.0));
            });

            app_shell::status_bar_view(cx);
        })
        .class("app-root");
    })
    .title("Lossy - Codec Emulation Effect")
    .inner_size((1250, 950))
    .run()
}

fn find_preset_dir() -> std::path::PathBuf {
    for c in &[
        "presets/lossy",
        "../../presets/lossy",
        "../../../presets/lossy",
    ] {
        let p = std::path::PathBuf::from(c);
        if p.exists() {
            return p;
        }
    }
    std::path::PathBuf::from("presets")
}
