/// Spectrogram display view — STFT magnitude with magma colormap.
use crate::app_shell::AppState;
use audio_engine::spectrogram;
use vizia::prelude::*;
use vizia::vg;

/// Custom spectrogram view.
pub struct SpectrogramView;

impl SpectrogramView {
    pub fn new(cx: &mut Context) -> Handle<'_, Self> {
        Self.build(cx, |_| {})
    }
}

impl View for SpectrogramView {
    fn draw(&self, cx: &mut DrawContext, canvas: &Canvas) {
        let bounds = cx.bounds();
        if bounds.w < 10.0 || bounds.h < 10.0 {
            return;
        }

        // Background
        let bg_rect = vg::Rect::from_xywh(bounds.x, bounds.y, bounds.w, bounds.h);
        let mut bg_paint = vg::Paint::default();
        bg_paint.set_color(vg::Color::from_rgb(8, 10, 20));
        canvas.draw_rect(bg_rect, &bg_paint);

        let pad_left = 50.0;
        let pad_right = 10.0;
        let pad_top = 10.0;
        let pad_bottom = 30.0;
        let plot_w = bounds.w - pad_left - pad_right;
        let plot_h = bounds.h - pad_top - pad_bottom;

        let state = cx.data::<AppState>();
        let audio = state.and_then(|s| s.rendered_audio.as_ref());

        if let Some(audio) = audio {
            let mono = audio.to_mono();
            let nfft = 2048;
            let hop = 512;
            let stft = spectrogram::compute_stft(&mono, audio.sample_rate, nfft, hop);
            let lut = spectrogram::magma_lut();

            let n_time = stft.bins.len();
            let n_freq = if n_time > 0 { stft.bins[0].len() } else { 0 };

            if n_time > 0 && n_freq > 0 {
                // bins are already in dB, range [-80, 0]
                let px_w = (plot_w / n_time as f32).max(1.0);
                let px_h = (plot_h / n_freq as f32).max(1.0);

                let mut paint = vg::Paint::default();
                paint.set_style(vg::paint::Style::Fill);

                for (ti, frame) in stft.bins.iter().enumerate() {
                    let x0 = bounds.x + pad_left + (ti as f32 * plot_w / n_time as f32);
                    for (fi, &db_val) in frame.iter().enumerate() {
                        // Flip frequency axis (low freq at bottom)
                        let y0 = bounds.y
                            + pad_top
                            + ((n_freq - 1 - fi) as f32 * plot_h / n_freq as f32);

                        // Normalize dB to 0..1 (bins already in dB, -80..0)
                        let norm = ((db_val + 80.0) / 80.0).clamp(0.0, 1.0);
                        let idx = (norm * 255.0) as usize;
                        let [r, g, b] = lut[idx.min(255)];

                        paint.set_color(vg::Color::from_rgb(r, g, b));
                        canvas.draw_rect(
                            vg::Rect::from_xywh(x0, y0, px_w.ceil(), px_h.ceil()),
                            &paint,
                        );
                    }
                }
            }

            // Time axis labels
            let mut text_paint = vg::Paint::default();
            text_paint.set_color(vg::Color::from_rgb(80, 100, 140));
            let font = vg::Font::default();
            let dur = audio.duration_seconds();
            for i in 0..=4 {
                let t = dur * i as f64 / 4.0;
                let x = bounds.x + pad_left + plot_w * (i as f32 / 4.0);
                canvas.draw_str(
                    &format!("{t:.1}s"),
                    vg::Point::new(x, bounds.y + bounds.h - 8.0),
                    &font,
                    &text_paint,
                );
            }

            // Frequency axis labels
            for &freq in &[100.0_f32, 1000.0, 5000.0, 10000.0] {
                let max_freq = audio.sample_rate as f32 / 2.0;
                let frac = freq / max_freq;
                if frac <= 1.0 {
                    let y = bounds.y + pad_top + plot_h * (1.0 - frac);
                    let label = if freq >= 1000.0 {
                        format!("{:.0}k", freq / 1000.0)
                    } else {
                        format!("{freq:.0}")
                    };
                    canvas.draw_str(
                        &label,
                        vg::Point::new(bounds.x + 4.0, y + 4.0),
                        &font,
                        &text_paint,
                    );
                }
            }
        } else {
            // No audio — show placeholder
            let mut paint = vg::Paint::default();
            paint.set_color(vg::Color::from_rgb(60, 80, 120));
            let font = vg::Font::default();
            canvas.draw_str(
                "Spectrogram (render audio first)",
                vg::Point::new(bounds.x + bounds.w / 2.0 - 80.0, bounds.y + bounds.h / 2.0),
                &font,
                &paint,
            );
        }
    }
}
