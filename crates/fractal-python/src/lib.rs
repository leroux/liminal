//! Python bindings for the Fractal DSP engine via pyo3.
//!
//! Exposes `render_fractal(input_audio, params_json) -> output_audio`
//! as a drop-in replacement for the Python/Numba implementation.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Render mono audio through the fractal engine.
///
/// Args:
///     input_audio: 1D numpy array of float64 samples
///     params_json: JSON string of parameters (sparse OK, missing keys get defaults)
///
/// Returns:
///     1D numpy array of float64 processed samples
#[pyfunction]
fn render_fractal<'py>(
    py: Python<'py>,
    input_audio: PyReadonlyArray1<'py, f64>,
    params_json: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let params = fractal_dsp::FractalParams::from_json(params_json)
        .map_err(|e| PyValueError::new_err(format!("Invalid params JSON: {e}")))?;
    let input = input_audio.as_slice()?;
    let output = fractal_dsp::render_fractal(input, &params);
    Ok(PyArray1::from_vec(py, output))
}

/// Render stereo audio through the fractal engine.
///
/// Args:
///     left: 1D numpy array of float64 samples (left channel)
///     right: 1D numpy array of float64 samples (right channel)
///     params_json: JSON string of parameters
///
/// Returns:
///     Tuple of (left_out, right_out) numpy arrays
#[pyfunction]
fn render_fractal_stereo<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, f64>,
    right: PyReadonlyArray1<'py, f64>,
    params_json: &str,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let params = fractal_dsp::FractalParams::from_json(params_json)
        .map_err(|e| PyValueError::new_err(format!("Invalid params JSON: {e}")))?;
    let (out_l, out_r) = fractal_dsp::render_fractal_stereo(
        left.as_slice()?,
        right.as_slice()?,
        &params,
    );
    Ok((
        PyArray1::from_vec(py, out_l),
        PyArray1::from_vec(py, out_r),
    ))
}

/// Python module definition.
#[pymodule]
fn fractal_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(render_fractal, m)?)?;
    m.add_function(wrap_pyfunction!(render_fractal_stereo, m)?)?;
    Ok(())
}
