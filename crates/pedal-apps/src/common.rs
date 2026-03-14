/// Shared playback and file I/O helpers used by all pedal app data models.
use crate::audio::buffer::AudioBuffer;
use crate::audio::playback::PlayHandle;
use crate::gui::app_shell::AppEvent;
use std::cell::RefCell;
use vizia::prelude::*;

thread_local! {
    static PLAY_HANDLE: RefCell<Option<PlayHandle>> = const { RefCell::new(None) };
}

pub fn stop_playback() {
    PLAY_HANDLE.with(|h| {
        if let Some(handle) = h.borrow_mut().take() {
            handle.stop();
        }
    });
}

pub fn start_playback(left: &[f64], right: &[f64], sr: u32) -> Result<(), String> {
    stop_playback();
    let handle =
        crate::audio::playback::play(left, right, sr, None).map_err(|e| e.to_string())?;
    PLAY_HANDLE.with(|h| {
        *h.borrow_mut() = Some(handle);
    });
    Ok(())
}

/// Handle AppEvents common to all pedal apps (file I/O, playback).
/// Returns true if the event was handled.
pub fn handle_common_app_event(
    cx: &mut EventContext,
    event: &AppEvent,
    source_audio: &mut Option<AudioBuffer>,
) -> bool {
    match event {
        AppEvent::WavLoaded(audio) => {
            *source_audio = Some(audio.clone());
            true
        }

        AppEvent::LoadWav => {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("WAV files", &["wav", "WAV"])
                .pick_file()
            {
                match crate::audio::wav::load_wav(&path, 44100) {
                    Ok(audio) => cx.emit(AppEvent::WavLoaded(audio)),
                    Err(e) => cx.emit(AppEvent::RenderError(e.to_string())),
                }
            }
            true
        }

        AppEvent::SaveWav => {
            if let Some(state) = cx.data::<crate::gui::app_shell::AppState>() {
                if let Some(audio) = &state.rendered_audio {
                    let audio = audio.clone();
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("WAV files", &["wav"])
                        .set_file_name("output.wav")
                        .save_file()
                    {
                        match crate::audio::wav::save_wav(&path, &audio) {
                            Ok(()) => cx.emit(AppEvent::SetStatus(format!(
                                "Saved: {}",
                                path.display()
                            ))),
                            Err(e) => cx.emit(AppEvent::RenderError(e.to_string())),
                        }
                    }
                }
            }
            true
        }

        AppEvent::Play => {
            if let Some(state) = cx.data::<crate::gui::app_shell::AppState>() {
                if let Some(audio) = &state.rendered_audio {
                    let left = audio.left();
                    let right = audio.right();
                    if let Err(e) = start_playback(&left, &right, audio.sample_rate) {
                        cx.emit(AppEvent::SetStatus(format!("Playback error: {e}")));
                        cx.emit(AppEvent::Stop);
                    }
                }
            }
            true
        }

        AppEvent::PlayDry => {
            if let Some(source) = source_audio {
                let left = source.left();
                let right = source.right();
                if let Err(e) = start_playback(&left, &right, source.sample_rate) {
                    cx.emit(AppEvent::SetStatus(format!("Playback error: {e}")));
                    cx.emit(AppEvent::Stop);
                }
            }
            true
        }

        AppEvent::Stop => {
            stop_playback();
            true
        }

        _ => false,
    }
}
