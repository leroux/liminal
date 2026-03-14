use pedal_apps::reverb::data::ReverbAppData;
use pedal_apps::reverb::params_tab;
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
        // Load theme CSS — leaked to 'static because vizia requires it
        let css: &'static str = Box::leak(theme::theme_css(&theme::REVERB_THEME).into_boxed_str());
        cx.add_stylesheet(css).expect("Failed to load theme CSS");

        // Build models
        AppState::default().build(cx);
        ReverbAppData::new().build(cx);
        PresetBrowserData::new(preset_dir.clone()).build(cx);

        // Load default impulse and auto-render
        let impulse = pedal_apps::audio::wav::make_impulse(44100, 0.5);
        cx.emit(AppEvent::WavLoaded(impulse));
        cx.emit(AppEvent::Render);

        // Main layout
        VStack::new(cx, |cx| {
            // Toolbar
            app_shell::toolbar_view(cx);

            // Tab bar
            app_shell::tab_bar_view(cx, TAB_NAMES);

            // Tab content
            Binding::new(cx, AppState::active_tab, |cx, tab| {
                let tab = tab.get(cx);
                VStack::new(cx, move |cx| {
                    match tab {
                        0 => params_tab::params_tab_view(cx),
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
                        5 => guide_view::guide_view(cx, guide_view::REVERB_GUIDE),
                        _ => {}
                    }
                })
                .height(Stretch(1.0));
            });

            // Status bar
            app_shell::status_bar_view(cx);
        })
        .class("app-root");
    })
    .title("Reverb - FDN Algorithmic Reverb")
    .inner_size((1250, 950))
    .run()
}

fn find_preset_dir() -> std::path::PathBuf {
    let candidates = [
        std::path::PathBuf::from("presets/reverb"),
        std::path::PathBuf::from("../../presets/reverb"),
        std::path::PathBuf::from("../../../presets/reverb"),
    ];
    for c in &candidates {
        if c.exists() {
            return c.clone();
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let p = dir.join("presets");
            if p.exists() {
                return p;
            }
        }
    }
    std::path::PathBuf::from("presets")
}
