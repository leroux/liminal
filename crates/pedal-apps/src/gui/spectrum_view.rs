/// Spectrum view — dry vs wet FFT magnitude curves.
use crate::gui::app_shell::AppState;
use vizia::prelude::*;
use vizia::vg;

/// Custom spectrum comparison view.
pub struct SpectrumView;

impl SpectrumView {
    pub fn new(cx: &mut Context) -> Handle<'_, Self> {
        Self.build(cx, |_| {})
    }
}

impl View for SpectrumView {
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

        // Grid
        let mut grid_paint = vg::Paint::default();
        grid_paint.set_color(vg::Color::from_argb(40, 60, 80, 120));
        grid_paint.set_stroke_width(1.0);
        grid_paint.set_style(vg::paint::Style::Stroke);

        let mut text_paint = vg::Paint::default();
        text_paint.set_color(vg::Color::from_rgb(80, 100, 140));
        let font = vg::Font::default();

        let db_min = -80.0_f32;
        let db_max = 0.0_f32;
        let db_range = db_max - db_min;

        // Horizontal dB lines
        for db in [-60.0_f32, -40.0, -20.0, 0.0] {
            let y = bounds.y + pad_top + plot_h * (1.0 - (db - db_min) / db_range);
            canvas.draw_line(
                vg::Point::new(bounds.x + pad_left, y),
                vg::Point::new(bounds.x + pad_left + plot_w, y),
                &grid_paint,
            );
            canvas.draw_str(
                format!("{db:.0}dB"),
                vg::Point::new(bounds.x + 4.0, y + 3.0),
                &font,
                &text_paint,
            );
        }

        // Vertical frequency lines (log scale)
        let log_min = 20.0_f32.log10();
        let log_max = 20000.0_f32.log10();
        let log_range = log_max - log_min;

        for &freq in &[100.0_f32, 1000.0, 10000.0] {
            let log_frac = (freq.log10() - log_min) / log_range;
            let x = bounds.x + pad_left + plot_w * log_frac;
            canvas.draw_line(
                vg::Point::new(x, bounds.y + pad_top),
                vg::Point::new(x, bounds.y + pad_top + plot_h),
                &grid_paint,
            );
            let label = if freq >= 1000.0 {
                format!("{:.0}k", freq / 1000.0)
            } else {
                format!("{freq:.0}")
            };
            canvas.draw_str(
                &label,
                vg::Point::new(x - 10.0, bounds.y + bounds.h - 8.0),
                &font,
                &text_paint,
            );
        }

        let state = cx.data::<AppState>();

        // Draw dry spectrum (source audio)
        if let Some(source) = state.and_then(|s| s.source_audio.as_ref()) {
            let spectrum = compute_spectrum(&source.to_mono(), source.sample_rate);
            draw_spectrum_curve(
                canvas,
                &spectrum,
                vg::Color::from_argb(150, 100, 100, 100),
                bounds.x + pad_left,
                bounds.y + pad_top,
                plot_w,
                plot_h,
                db_min,
                db_range,
                log_min,
                log_range,
            );
        }

        // Draw wet spectrum (rendered audio)
        if let Some(rendered) = state.and_then(|s| s.rendered_audio.as_ref()) {
            let spectrum = compute_spectrum(&rendered.to_mono(), rendered.sample_rate);
            draw_spectrum_curve(
                canvas,
                &spectrum,
                vg::Color::from_argb(220, 70, 150, 255),
                bounds.x + pad_left,
                bounds.y + pad_top,
                plot_w,
                plot_h,
                db_min,
                db_range,
                log_min,
                log_range,
            );
        }

        // Legend
        if state.and_then(|s| s.rendered_audio.as_ref()).is_some() {
            let mut dry_paint = vg::Paint::default();
            dry_paint.set_color(vg::Color::from_rgb(100, 100, 100));
            canvas.draw_str(
                "Dry",
                vg::Point::new(bounds.x + pad_left + plot_w - 80.0, bounds.y + pad_top + 14.0),
                &font,
                &dry_paint,
            );
            let mut wet_paint = vg::Paint::default();
            wet_paint.set_color(vg::Color::from_rgb(70, 150, 255));
            canvas.draw_str(
                "Wet",
                vg::Point::new(bounds.x + pad_left + plot_w - 40.0, bounds.y + pad_top + 14.0),
                &font,
                &wet_paint,
            );
        }
    }
}

/// Compute FFT magnitude spectrum as (freq_hz, magnitude_db) pairs.
fn compute_spectrum(mono: &[f64], sample_rate: u32) -> Vec<(f32, f32)> {
    use std::f64::consts::PI;

    let nfft = 4096;
    let n = mono.len().min(nfft);
    if n < 4 {
        return Vec::new();
    }

    // Apply Hann window
    let mut windowed: Vec<f64> = (0..n)
        .map(|i| {
            let w = 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos());
            mono[i] * w
        })
        .collect();
    windowed.resize(nfft, 0.0);

    // FFT via realfft
    let mut planner = realfft::RealFftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(nfft);
    let mut spectrum = fft.make_output_vec();
    let _ = fft.process(&mut windowed, &mut spectrum);

    let freq_step = sample_rate as f64 / nfft as f64;

    spectrum
        .iter()
        .enumerate()
        .skip(1) // skip DC
        .map(|(i, c)| {
            let freq = (i as f64 * freq_step) as f32;
            let mag = (c.re * c.re + c.im * c.im).sqrt();
            let db = (20.0 * (mag / nfft as f64).max(1e-10).log10()) as f32;
            (freq, db)
        })
        .filter(|(f, _)| *f >= 20.0 && *f <= 20000.0)
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn draw_spectrum_curve(
    canvas: &Canvas,
    spectrum: &[(f32, f32)],
    color: vg::Color,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    db_min: f32,
    db_range: f32,
    log_min: f32,
    log_range: f32,
) {
    if spectrum.is_empty() {
        return;
    }

    let mut paint = vg::Paint::default();
    paint.set_color(color);
    paint.set_stroke_width(1.5);
    paint.set_style(vg::paint::Style::Stroke);
    paint.set_anti_alias(true);

    let mut path = vg::Path::new();
    let mut started = false;

    for &(freq, db) in spectrum {
        let log_frac = (freq.log10() - log_min) / log_range;
        let db_frac = (db - db_min) / db_range;
        let px = x + w * log_frac;
        let py = y + h * (1.0 - db_frac);

        if !started {
            path.move_to(vg::Point::new(px, py));
            started = true;
        } else {
            path.line_to(vg::Point::new(px, py));
        }
    }

    canvas.draw_path(&path, &paint);
}
