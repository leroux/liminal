/// Waveform display view — dual L/R waveforms with metrics overlay.
use crate::gui::app_shell::AppState;
use vizia::prelude::*;
use vizia::vg;

/// Custom waveform view that draws rendered audio with skia_safe.
pub struct WaveformView;

impl WaveformView {
    pub fn new(cx: &mut Context) -> Handle<'_, Self> {
        Self.build(cx, |_| {})
    }
}

impl View for WaveformView {
    fn draw(&self, cx: &mut DrawContext, canvas: &Canvas) {
        let bounds = cx.bounds();
        if bounds.w < 10.0 || bounds.h < 10.0 {
            return;
        }

        let bg_color = vg::Color::from_rgb(12, 14, 28);
        let grid_color = vg::Color::from_argb(60, 60, 80, 120);
        let text_color = vg::Color::from_rgb(100, 140, 200);
        let wave_l_color = vg::Color::from_argb(200, 70, 150, 255);
        let wave_r_color = vg::Color::from_argb(180, 255, 130, 70);

        let pad_left = 50.0;
        let pad_right = 10.0;
        let pad_top = 10.0;
        let metrics_height = 80.0;
        let plot_w = bounds.w - pad_left - pad_right;
        let plot_h = (bounds.h - pad_top - metrics_height) / 2.0;

        // Background
        let bg_rect = vg::Rect::from_xywh(bounds.x, bounds.y, bounds.w, bounds.h);
        let mut bg_paint = vg::Paint::default();
        bg_paint.set_color(bg_color);
        canvas.draw_rect(bg_rect, &bg_paint);

        // Grid paint
        let mut grid_paint = vg::Paint::default();
        grid_paint.set_color(grid_color);
        grid_paint.set_stroke_width(1.0);
        grid_paint.set_style(vg::paint::Style::Stroke);
        grid_paint.set_anti_alias(true);

        let font = vg::Font::default();

        let draw_channel_grid = |canvas: &Canvas, y_off: f32| {
            let cy = bounds.y + y_off + plot_h / 2.0;
            canvas.draw_line(
                vg::Point::new(bounds.x + pad_left, cy),
                vg::Point::new(bounds.x + pad_left + plot_w, cy),
                &grid_paint,
            );
            for frac in [0.25_f32, 0.75] {
                let gy = bounds.y + y_off + plot_h * frac;
                canvas.draw_line(
                    vg::Point::new(bounds.x + pad_left, gy),
                    vg::Point::new(bounds.x + pad_left + plot_w, gy),
                    &grid_paint,
                );
            }
        };

        draw_channel_grid(canvas, pad_top);
        draw_channel_grid(canvas, pad_top + plot_h);

        // Amplitude labels
        let mut text_paint = vg::Paint::default();
        text_paint.set_color(text_color);
        for (label, frac) in [("1.0", 0.0_f32), ("0", 0.5), ("-1", 1.0)] {
            let y = bounds.y + pad_top + plot_h * frac + 12.0;
            canvas.draw_str(label, vg::Point::new(bounds.x + 4.0, y), &font, &text_paint);
        }

        // Channel labels
        let mut l_paint = vg::Paint::default();
        l_paint.set_color(wave_l_color);
        canvas.draw_str(
            "L",
            vg::Point::new(bounds.x + pad_left + 4.0, bounds.y + pad_top + 14.0),
            &font,
            &l_paint,
        );
        let mut r_paint = vg::Paint::default();
        r_paint.set_color(wave_r_color);
        canvas.draw_str(
            "R",
            vg::Point::new(
                bounds.x + pad_left + 4.0,
                bounds.y + pad_top + plot_h + 14.0,
            ),
            &font,
            &r_paint,
        );

        // Get audio data from AppState
        let state = cx.data::<AppState>();
        let audio = state.and_then(|s| s.rendered_audio.as_ref());

        if let Some(audio) = audio {
            let left = audio.left();
            let right = audio.right();

            // Draw waveforms
            draw_waveform(
                canvas,
                &left,
                wave_l_color,
                bounds.x + pad_left,
                bounds.y + pad_top,
                plot_w,
                plot_h,
            );
            draw_waveform(
                canvas,
                &right,
                wave_r_color,
                bounds.x + pad_left,
                bounds.y + pad_top + plot_h,
                plot_w,
                plot_h,
            );

            // Time axis labels
            let dur = audio.duration_seconds();
            let mut time_paint = vg::Paint::default();
            time_paint.set_color(text_color);
            for i in 0..=4 {
                let t = dur * i as f64 / 4.0;
                let x = bounds.x + pad_left + plot_w * (i as f32 / 4.0);
                canvas.draw_str(
                    format!("{t:.1}s"),
                    vg::Point::new(x, bounds.y + pad_top + plot_h * 2.0 + 14.0),
                    &font,
                    &time_paint,
                );
            }
        }

        // Draw metrics
        if let Some(state) = state {
            if let Some(metrics) = &state.metrics {
                let my = bounds.y + bounds.h - metrics_height + 8.0;
                let mut mp = vg::Paint::default();
                mp.set_color(vg::Color::from_rgb(80, 120, 180));

                let lines = [
                    format!(
                        "RT60: {:.2}s  EDT: {:.2}s  C50: {:.1}dB  C80: {:.1}dB",
                        metrics.rt60.unwrap_or(0.0),
                        metrics.edt.unwrap_or(0.0),
                        metrics.c50.unwrap_or(0.0),
                        metrics.c80.unwrap_or(0.0)
                    ),
                    format!(
                        "Centroid: {:.0}Hz  Density: {:.3}  Crest: {:.1}dB  Flatness: {:.3}",
                        metrics.spectral_centroid.unwrap_or(0.0),
                        metrics.echo_density.unwrap_or(0.0),
                        metrics.crest_factor.unwrap_or(0.0),
                        metrics.spectral_flatness.unwrap_or(0.0)
                    ),
                    format!(
                        "RMS: {:.1}dB  Peak: {:.1}dB  BW: {:.0}Hz",
                        metrics.rms_db.unwrap_or(0.0),
                        metrics.peak_db.unwrap_or(0.0),
                        metrics.bandwidth.unwrap_or(0.0)
                    ),
                ];

                for (i, line) in lines.iter().enumerate() {
                    canvas.draw_str(
                        line,
                        vg::Point::new(bounds.x + pad_left, my + i as f32 * 16.0),
                        &font,
                        &mp,
                    );
                }
            }
        }
    }
}

/// Draw a single channel waveform as a filled path.
fn draw_waveform(
    canvas: &Canvas,
    samples: &[f64],
    color: vg::Color,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
) {
    if samples.is_empty() || w < 2.0 {
        return;
    }

    let n = samples.len();
    let pixels = w as usize;

    // Downsample: for each pixel column, find min/max
    let mut paint = vg::Paint::default();
    paint.set_color(color);
    paint.set_stroke_width(1.0);
    paint.set_style(vg::paint::Style::Stroke);
    paint.set_anti_alias(true);

    let cy = y + h / 2.0;

    // Build a path through the waveform center line
    let mut path = vg::Path::new();
    let mut started = false;

    let samples_per_pixel = n as f64 / pixels as f64;

    for px in 0..pixels {
        let start_idx = (px as f64 * samples_per_pixel) as usize;
        let end_idx = (((px + 1) as f64 * samples_per_pixel) as usize).min(n);

        if start_idx >= end_idx {
            continue;
        }

        // Find the sample closest to the center of this pixel range
        let mid_idx = (start_idx + end_idx) / 2;
        let val = samples[mid_idx].clamp(-1.0, 1.0) as f32;
        let px_x = x + px as f32;
        let px_y = cy - val * (h / 2.0);

        if !started {
            path.move_to(vg::Point::new(px_x, px_y));
            started = true;
        } else {
            path.line_to(vg::Point::new(px_x, px_y));
        }
    }

    canvas.draw_path(&path, &paint);

    // Draw min/max envelope with reduced opacity
    let mut env_paint = vg::Paint::default();
    env_paint.set_color(vg::Color::from_argb(
        (color.a() as f32 * 0.3) as u8,
        color.r(),
        color.g(),
        color.b(),
    ));
    env_paint.set_stroke_width(1.0);
    env_paint.set_style(vg::paint::Style::Stroke);
    env_paint.set_anti_alias(true);

    for px in 0..pixels {
        let start_idx = (px as f64 * samples_per_pixel) as usize;
        let end_idx = (((px + 1) as f64 * samples_per_pixel) as usize).min(n);
        if start_idx >= end_idx {
            continue;
        }

        let mut mn = f64::MAX;
        let mut mx = f64::MIN;
        for &s in &samples[start_idx..end_idx] {
            mn = mn.min(s);
            mx = mx.max(s);
        }

        let mn = mn.clamp(-1.0, 1.0) as f32;
        let mx = mx.clamp(-1.0, 1.0) as f32;
        let px_x = x + px as f32;

        canvas.draw_line(
            vg::Point::new(px_x, cy - mx * (h / 2.0)),
            vg::Point::new(px_x, cy - mn * (h / 2.0)),
            &env_paint,
        );
    }
}
