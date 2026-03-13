//! AI-assisted sound design library.
//!
//! Provides parameter validation, audio metrics formatting, and an AI tuner
//! that uses `claudewire` to communicate with the Claude CLI for autonomous
//! sound design iteration.

pub mod metrics;
pub mod params;
pub mod tuner;
