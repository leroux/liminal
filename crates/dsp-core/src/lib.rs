//! Shared DSP utilities: filters, effects, safety, serde helpers.
//!
//! Pure functions and plain structs — no traits, no abstractions.

pub mod biquad;
pub mod effects;
pub mod one_pole;
pub mod safety;
pub mod serde_helpers;

/// Default sample rate used across all DSP engines.
pub const SR: f64 = 44100.0;

/// Flush denormalized f64 values to zero.
/// Denormals (subnormal floats) cause 10-100x CPU slowdowns on x86.
/// Use on filter state variables after processing silence or decay tails.
#[inline(always)]
pub fn flush_denormal(x: f64) -> f64 {
    // f64 denormals have absolute value < 2.2e-308.
    // This threshold is well above that but far below any audible signal.
    if x.abs() < 1e-30 {
        0.0
    } else {
        x
    }
}

/// Set FTZ+DAZ on x86_64 to prevent denormal CPU spikes in audio processing.
/// Call at the start of your audio processing callback.
/// Returns the previous MXCSR value for restoration if needed.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn enable_flush_to_zero() -> u32 {
    unsafe {
        let prev = std::arch::x86_64::_mm_getcsr();
        std::arch::x86_64::_mm_setcsr(prev | 0x8040); // FTZ (bit 15) + DAZ (bit 6)
        prev
    }
}

/// No-op on non-x86_64 architectures.
#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub fn enable_flush_to_zero() -> u32 {
    0
}
