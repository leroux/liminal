/// STFT spectrogram computation for visualization.
use realfft::RealFftPlanner;

/// Computed STFT spectrogram data.
#[derive(Debug, Clone)]
pub struct SpectrogramData {
    /// Magnitude bins, indexed as [time_frame][freq_bin].
    /// Values are in dB (normalized so max=0dB).
    pub bins: Vec<Vec<f32>>,
    /// Frequency values for each bin (Hz).
    pub freq_axis: Vec<f32>,
    /// Time values for each frame (seconds).
    pub time_axis: Vec<f32>,
    /// Number of frequency bins.
    pub n_freq: usize,
    /// Number of time frames.
    pub n_time: usize,
}

/// Compute STFT spectrogram from mono audio.
pub fn compute_stft(audio: &[f64], sr: u32, nfft: usize, hop: usize) -> SpectrogramData {
    let n = audio.len();
    if n < nfft {
        return SpectrogramData {
            bins: vec![],
            freq_axis: vec![],
            time_axis: vec![],
            n_freq: 0,
            n_time: 0,
        };
    }

    let n_freq = nfft / 2 + 1;
    let n_frames = (n - nfft) / hop + 1;

    // Precompute Hann window
    let window: Vec<f64> = (0..nfft)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / nfft as f64).cos()))
        .collect();

    let mut planner = RealFftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(nfft);

    let mut bins = Vec::with_capacity(n_frames);
    let mut global_max: f64 = 1e-30;

    for frame in 0..n_frames {
        let start = frame * hop;
        let mut input: Vec<f64> = audio[start..start + nfft]
            .iter()
            .zip(window.iter())
            .map(|(s, w)| s * w)
            .collect();

        let mut spectrum = fft.make_output_vec();
        fft.process(&mut input, &mut spectrum).ok();

        let magnitudes: Vec<f64> = spectrum.iter().map(|c| c.norm()).collect();
        let frame_max = magnitudes.iter().cloned().fold(0.0_f64, f64::max);
        global_max = global_max.max(frame_max);
        bins.push(magnitudes);
    }

    // Convert to dB, normalized to global max
    let db_bins: Vec<Vec<f32>> = bins
        .iter()
        .map(|frame| {
            frame
                .iter()
                .map(|&m| {
                    let db = 20.0 * (m.max(1e-30) / global_max).log10();
                    db.max(-80.0) as f32
                })
                .collect()
        })
        .collect();

    let freq_axis: Vec<f32> = (0..n_freq)
        .map(|i| (i as f64 * sr as f64 / nfft as f64) as f32)
        .collect();

    let time_axis: Vec<f32> = (0..n_frames)
        .map(|i| (i * hop) as f32 / sr as f32)
        .collect();

    SpectrogramData {
        bins: db_bins,
        freq_axis,
        time_axis,
        n_freq,
        n_time: n_frames,
    }
}

/// Magma colormap — 8-stop version matching Python waveform.py.
pub const MAGMA_STOPS: [[f64; 3]; 8] = [
    [0.001, 0.000, 0.014],
    [0.082, 0.046, 0.220],
    [0.280, 0.087, 0.475],
    [0.504, 0.110, 0.498],
    [0.721, 0.180, 0.378],
    [0.910, 0.337, 0.215],
    [0.986, 0.597, 0.170],
    [0.988, 0.999, 0.645],
];

/// Build a 256-entry magma LUT as [r, g, b] in 0..255.
pub fn magma_lut() -> Vec<[u8; 3]> {
    let n = 256;
    let stops = &MAGMA_STOPS;
    let n_stops = stops.len();

    (0..n)
        .map(|i| {
            let t = i as f64 / (n - 1) as f64;
            let seg = (t * (n_stops - 1) as f64).min((n_stops - 2) as f64);
            let idx = seg as usize;
            let frac = seg - idx as f64;

            let r = stops[idx][0] + frac * (stops[idx + 1][0] - stops[idx][0]);
            let g = stops[idx][1] + frac * (stops[idx + 1][1] - stops[idx][1]);
            let b = stops[idx][2] + frac * (stops[idx + 1][2] - stops[idx][2]);

            [
                (r * 255.0).round() as u8,
                (g * 255.0).round() as u8,
                (b * 255.0).round() as u8,
            ]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_stft_sine() {
        let sr = 44100;
        let n = sr * 2;
        let freq = 1000.0;
        let sine: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / sr as f64).sin())
            .collect();

        let spec = compute_stft(&sine, sr as u32, 2048, 512);
        assert!(spec.n_time > 0);
        assert_eq!(spec.n_freq, 1025);
        assert_eq!(spec.bins.len(), spec.n_time);
        assert_eq!(spec.bins[0].len(), spec.n_freq);
    }

    #[test]
    fn test_compute_stft_short() {
        let audio = vec![0.0; 100]; // shorter than nfft
        let spec = compute_stft(&audio, 44100, 2048, 512);
        assert_eq!(spec.n_time, 0);
    }

    #[test]
    fn test_magma_lut_range() {
        let lut = magma_lut();
        assert_eq!(lut.len(), 256);
        // First entry should be near-black
        assert!(lut[0][0] < 10);
        // Last entry should be bright
        assert!(lut[255][0] > 200);
    }
}
