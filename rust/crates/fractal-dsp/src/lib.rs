//! Fractal audio fractalization DSP engine.
//!
//! Port of the Python `fractal/engine/` package.
//!
//! Single entry point: `render_fractal(input_audio, params) -> output_audio`

pub mod chain;
pub mod core;
pub mod filters;
pub mod params;

pub use chain::{render_fractal, render_fractal_stereo};
pub use params::FractalParams;
