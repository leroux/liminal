//! 8-node Feedback Delay Network reverb DSP engine.
//!
//! Port of the Python `reverb/engine/` package.
//!
//! Single entry point: `render_fdn(input_audio, params) -> output_audio`

pub mod chain;
pub mod fdn;
pub mod fdn_mod;
pub mod matrix;
pub mod params;

pub use chain::{render_fdn, render_fdn_stereo};
pub use params::ReverbParams;
