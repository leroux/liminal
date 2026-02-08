//! Lossy codec emulation DSP engine.
//!
//! Port of the Python `lossy/engine/` package.
//!
//! Single entry point: `render_lossy(input_audio, params) -> output_audio`

pub mod bitcrush;
pub mod chain;
pub mod filters;
pub mod packets;
pub mod params;
pub mod rng;
pub mod spectral;

pub use chain::{render_lossy, render_lossy_stereo};
pub use params::LossyParams;
