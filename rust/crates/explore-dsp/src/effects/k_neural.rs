//! K-series: Neural / ML-inspired effects (K001-K009).
//!
//! Random neural network waveshapers, autoencoders, echo state networks,
//! neural ODEs, weight interpolation, convnet filter banks, tiny RNNs,
//! overfit-then-corrupt, and random projections.

use std::collections::HashMap;
use serde_json::Value;
use crate::{AudioOutput, EffectEntry, pf, pi, pu, params};
use crate::primitives::*;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand::Rng;

// ============================================================================
// Shared helpers
// ============================================================================

/// Box-Muller transform: generate a standard normal sample from a uniform RNG.
fn randn(rng: &mut ChaCha8Rng) -> f64 {
    loop {
        let u1: f64 = rng.random::<f64>();
        let u2: f64 = rng.random::<f64>();
        if u1 > 1e-30 {
            let r = (-2.0 * u1.ln()).sqrt();
            return r * (2.0 * std::f64::consts::PI * u2).cos();
        }
    }
}

/// Generate a vector of standard normal random values.
fn randn_vec(rng: &mut ChaCha8Rng, n: usize) -> Vec<f64> {
    (0..n).map(|_| randn(rng)).collect()
}

/// Generate a matrix (row-major, rows x cols) of standard normal random values.
fn randn_mat(rng: &mut ChaCha8Rng, rows: usize, cols: usize) -> Vec<f64> {
    (0..rows * cols).map(|_| randn(rng)).collect()
}

/// Matrix-vector multiply: mat is row-major (rows x cols), vec has length cols.
/// Returns vector of length rows.
fn mat_vec_mul(mat: &[f64], rows: usize, cols: usize, v: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0f64; rows];
    for i in 0..rows {
        let mut sum = 0.0;
        for j in 0..cols {
            sum += mat[i * cols + j] * v[j];
        }
        out[i] = sum;
    }
    out
}

/// Matrix-matrix multiply: A (m x k) * B (k x n) -> C (m x n). All row-major.
fn mat_mat_mul(a: &[f64], m: usize, k: usize, b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

/// Transpose a row-major matrix (rows x cols) -> (cols x rows).
fn mat_transpose(mat: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            out[j * rows + i] = mat[i * cols + j];
        }
    }
    out
}

/// Add bias vector to each row of a matrix (num_rows x dim).
/// Returns modified matrix.
fn mat_add_bias(mat: &[f64], num_rows: usize, dim: usize, bias: &[f64]) -> Vec<f64> {
    let mut out = mat.to_vec();
    for i in 0..num_rows {
        for j in 0..dim {
            out[i * dim + j] += bias[j];
        }
    }
    out
}

/// Apply tanh element-wise to a matrix/vector.
fn tanh_vec(v: &[f64]) -> Vec<f64> {
    v.iter().map(|&x| x.tanh()).collect()
}

/// Element-wise: 1 - x^2 (tanh derivative).
fn tanh_deriv(tanh_output: &[f64]) -> Vec<f64> {
    tanh_output.iter().map(|&x| 1.0 - x * x).collect()
}

/// Element-wise multiply.
fn elem_mul(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect()
}

/// Element-wise subtract: a - b.
fn elem_sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x - y).collect()
}

/// Sum columns of a matrix (num_rows x dim) -> vector of length dim.
fn col_sum(mat: &[f64], num_rows: usize, dim: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; dim];
    for i in 0..num_rows {
        for j in 0..dim {
            out[j] += mat[i * dim + j];
        }
    }
    out
}

/// Scale a vector by a scalar.
fn vec_scale(v: &[f64], s: f64) -> Vec<f64> {
    v.iter().map(|&x| x * s).collect()
}

/// Add two vectors.
fn vec_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
}

/// Standard deviation of a slice.
fn std_dev(v: &[f64]) -> f64 {
    let n = v.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mean = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n;
    var.sqrt()
}

/// Overlap-add processing. process_fn receives a Hann-windowed chunk of length
/// chunk_size and must return a Vec<f64> of the same length.
fn overlap_add<F>(samples: &[f64], chunk_size: usize, hop_size: usize, mut process_fn: F) -> Vec<f64>
where
    F: FnMut(&[f64]) -> Vec<f64>,
{
    let n = samples.len();
    let window = hann_window_f64(chunk_size);
    let mut out = vec![0.0f64; n];
    let mut norm = vec![0.0f64; n];

    let mut pos = 0;
    while pos < n {
        let end = (pos + chunk_size).min(n);
        let seg_len = end - pos;
        let mut chunk = vec![0.0f64; chunk_size];
        chunk[..seg_len].copy_from_slice(&samples[pos..end]);
        for i in 0..chunk_size {
            chunk[i] *= window[i];
        }

        let processed = process_fn(&chunk);

        for i in 0..seg_len {
            out[pos + i] += processed[i] * window[i];
            norm[pos + i] += window[i] * window[i];
        }
        pos += hop_size;
    }

    // Normalise where we have overlap
    for i in 0..n {
        if norm[i] > 1e-8 {
            out[i] /= norm[i];
        }
    }
    out
}

/// Generate a Hann window of given size (f64).
fn hann_window_f64(size: usize) -> Vec<f64> {
    use std::f64::consts::PI;
    if size <= 1 {
        return vec![1.0; size];
    }
    (0..size)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / (size - 1) as f64).cos()))
        .collect()
}

/// Thin QR decomposition via modified Gram-Schmidt.
/// Input: matrix A (rows x cols), rows >= cols.
/// Returns Q (rows x cols) orthonormal columns, stored row-major.
fn qr_q(a: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    // Work column-major for convenience, then convert back
    let ncols = cols.min(rows);
    let mut q_cols: Vec<Vec<f64>> = Vec::with_capacity(ncols);

    for j in 0..ncols {
        // Extract column j of A
        let mut v: Vec<f64> = (0..rows).map(|i| a[i * cols + j]).collect();

        // Subtract projections onto previous q vectors
        for q_k in &q_cols {
            let dot: f64 = v.iter().zip(q_k.iter()).map(|(&a, &b)| a * b).sum();
            for i in 0..rows {
                v[i] -= dot * q_k[i];
            }
        }

        // Normalise
        let norm: f64 = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for x in &mut v {
                *x /= norm;
            }
        }
        q_cols.push(v);
    }

    // Convert to row-major (rows x ncols)
    let mut q = vec![0.0f64; rows * ncols];
    for j in 0..ncols {
        for i in 0..rows {
            q[i * ncols + j] = q_cols[j][i];
        }
    }
    q
}

// ---------------------------------------------------------------------------
// K001 -- Random Neural Network Waveshaper
// ---------------------------------------------------------------------------

fn process_k001(samples: &[f32], _sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let hidden_size = pi(params, "hidden_size", 32) as usize;
    let chunk_size = pi(params, "chunk_size", 64) as usize;
    let seed = pu(params, "seed", 42);
    let hop_size = chunk_size / 2;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Xavier-style init for stability
    let scale1 = (2.0 / (chunk_size + hidden_size) as f64).sqrt();
    let w1: Vec<f64> = randn_mat(&mut rng, hidden_size, chunk_size)
        .iter().map(|&x| x * scale1).collect();
    let b1: Vec<f64> = randn_vec(&mut rng, hidden_size)
        .iter().map(|&x| x * 0.01).collect();
    let scale2 = (2.0 / (hidden_size + hidden_size) as f64).sqrt();
    let w2: Vec<f64> = randn_mat(&mut rng, hidden_size, hidden_size)
        .iter().map(|&x| x * scale2).collect();
    let b2: Vec<f64> = randn_vec(&mut rng, hidden_size)
        .iter().map(|&x| x * 0.01).collect();
    let scale3 = (2.0 / (hidden_size + chunk_size) as f64).sqrt();
    let w3: Vec<f64> = randn_mat(&mut rng, chunk_size, hidden_size)
        .iter().map(|&x| x * scale3).collect();
    let b3: Vec<f64> = randn_vec(&mut rng, chunk_size)
        .iter().map(|&x| x * 0.01).collect();

    let x: Vec<f64> = samples.iter().map(|&s| s as f64).collect();

    let out = overlap_add(&x, chunk_size, hop_size, |chunk| {
        // h1 = tanh(W1 @ chunk + b1)
        let h1_pre = vec_add(&mat_vec_mul(&w1, hidden_size, chunk_size, chunk), &b1);
        let h1 = tanh_vec(&h1_pre);
        // h2 = tanh(W2 @ h1 + b2)
        let h2_pre = vec_add(&mat_vec_mul(&w2, hidden_size, hidden_size, &h1), &b2);
        let h2 = tanh_vec(&h2_pre);
        // y = W3 @ h2 + b3
        vec_add(&mat_vec_mul(&w3, chunk_size, hidden_size, &h2), &b3)
    });

    AudioOutput::Mono(out.iter().map(|&x| x as f32).collect())
}

fn variants_k001() -> Vec<HashMap<String, Value>> {
    vec![
        params!("hidden_size" => 16, "chunk_size" => 32, "seed" => 1),
        params!("hidden_size" => 32, "chunk_size" => 64, "seed" => 42),
        params!("hidden_size" => 64, "chunk_size" => 64, "seed" => 7),
        params!("hidden_size" => 128, "chunk_size" => 128, "seed" => 99),
        params!("hidden_size" => 32, "chunk_size" => 256, "seed" => 13),
        params!("hidden_size" => 64, "chunk_size" => 32, "seed" => 55),
    ]
}

// ---------------------------------------------------------------------------
// K002 -- Tiny Autoencoder
// ---------------------------------------------------------------------------

fn process_k002(samples: &[f32], _sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let latent_dim = pi(params, "latent_dim", 8) as usize;
    let chunk_size = pi(params, "chunk_size", 128) as usize;
    let corruption_type = params.get("corruption_type")
        .and_then(|v| v.as_str())
        .unwrap_or("noise")
        .to_string();
    let corruption_amount = pf(params, "corruption_amount", 0.5) as f64;
    let training_epochs = pi(params, "training_epochs", 100) as usize;
    let hop_size = chunk_size / 2;

    let x: Vec<f64> = samples.iter().map(|&s| s as f64).collect();

    // Collect training chunks
    let window = hann_window_f64(chunk_size);
    let mut chunks: Vec<Vec<f64>> = Vec::new();
    let mut pos = 0;
    while pos + chunk_size <= x.len() {
        let c: Vec<f64> = (0..chunk_size).map(|i| x[pos + i] * window[i]).collect();
        chunks.push(c);
        pos += hop_size;
    }
    if chunks.is_empty() {
        return AudioOutput::Mono(samples.to_vec());
    }

    let num_chunks = chunks.len();
    // Flatten chunks into data matrix (num_chunks x chunk_size), row-major
    let mut data = vec![0.0f64; num_chunks * chunk_size];
    for (i, c) in chunks.iter().enumerate() {
        data[i * chunk_size..(i + 1) * chunk_size].copy_from_slice(c);
    }

    // Init weights
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let scale_enc = (2.0 / (chunk_size + latent_dim) as f64).sqrt();
    let mut w_enc: Vec<f64> = randn_mat(&mut rng, latent_dim, chunk_size)
        .iter().map(|&x| x * scale_enc).collect();
    let mut b_enc = vec![0.0f64; latent_dim];
    let scale_dec = (2.0 / (latent_dim + chunk_size) as f64).sqrt();
    let mut w_dec: Vec<f64> = randn_mat(&mut rng, chunk_size, latent_dim)
        .iter().map(|&x| x * scale_dec).collect();
    let mut b_dec = vec![0.0f64; chunk_size];

    let lr = 0.001;

    // Training loop (batch gradient descent)
    for _epoch in 0..training_epochs {
        // Forward: Z = tanh(data @ W_enc.T + b_enc)  -> (N, latent_dim)
        let w_enc_t = mat_transpose(&w_enc, latent_dim, chunk_size); // (chunk_size, latent_dim)
        let z_pre = mat_mat_mul(&data, num_chunks, chunk_size, &w_enc_t, latent_dim);
        let z_pre = mat_add_bias(&z_pre, num_chunks, latent_dim, &b_enc);
        let z = tanh_vec(&z_pre); // (N * latent_dim)

        // recon = Z @ W_dec.T + b_dec  -> (N, chunk_size)
        let w_dec_t = mat_transpose(&w_dec, chunk_size, latent_dim); // (latent_dim, chunk_size)
        let recon = mat_mat_mul(&z, num_chunks, latent_dim, &w_dec_t, chunk_size);
        let recon = mat_add_bias(&recon, num_chunks, chunk_size, &b_dec);

        // error = recon - data
        let error = elem_sub(&recon, &data);

        // d_recon = (2/N) * error
        let d_recon = vec_scale(&error, 2.0 / num_chunks as f64);

        // Decoder grads: dW_dec = d_recon.T @ Z -> (chunk_size, latent_dim)
        let d_recon_t = mat_transpose(&d_recon, num_chunks, chunk_size); // (chunk_size, N)
        let dw_dec = mat_mat_mul(&d_recon_t, chunk_size, num_chunks, &z, latent_dim);
        let db_dec = col_sum(&d_recon, num_chunks, chunk_size);

        // Encoder grads: dZ = d_recon @ W_dec -> (N, latent_dim)
        let dz = mat_mat_mul(&d_recon, num_chunks, chunk_size, &w_dec, latent_dim);
        // dZ_pre = dZ * (1 - Z^2)
        let z_deriv = tanh_deriv(&z);
        let dz_pre = elem_mul(&dz, &z_deriv);
        // dW_enc = dZ_pre.T @ data -> (latent_dim, chunk_size)
        let dz_pre_t = mat_transpose(&dz_pre, num_chunks, latent_dim);
        let dw_enc = mat_mat_mul(&dz_pre_t, latent_dim, num_chunks, &data, chunk_size);
        let db_enc = col_sum(&dz_pre, num_chunks, latent_dim);

        // Update weights
        for i in 0..w_dec.len() { w_dec[i] -= lr * dw_dec[i]; }
        for i in 0..b_dec.len() { b_dec[i] -= lr * db_dec[i]; }
        for i in 0..w_enc.len() { w_enc[i] -= lr * dw_enc[i]; }
        for i in 0..b_enc.len() { b_enc[i] -= lr * db_enc[i]; }
    }

    // Inference with corruption
    let mut corrupt_rng = ChaCha8Rng::seed_from_u64(42);

    let out = overlap_add(&x, chunk_size, hop_size, |chunk| {
        // z = tanh(W_enc @ chunk + b_enc)
        let z_pre = vec_add(&mat_vec_mul(&w_enc, latent_dim, chunk_size, chunk), &b_enc);
        let mut z = tanh_vec(&z_pre);

        // Corrupt latent
        match corruption_type.as_str() {
            "noise" => {
                for i in 0..latent_dim {
                    z[i] += randn(&mut corrupt_rng) * corruption_amount;
                }
            }
            "dropout" => {
                for i in 0..latent_dim {
                    let r: f64 = corrupt_rng.random();
                    if r <= corruption_amount {
                        z[i] = 0.0;
                    }
                }
            }
            "scale" => {
                for i in 0..latent_dim {
                    z[i] *= corruption_amount;
                }
            }
            _ => {}
        }

        // recon = W_dec @ z + b_dec
        vec_add(&mat_vec_mul(&w_dec, chunk_size, latent_dim, &z), &b_dec)
    });

    AudioOutput::Mono(out.iter().map(|&x| x as f32).collect())
}

fn variants_k002() -> Vec<HashMap<String, Value>> {
    vec![
        params!("latent_dim" => 4, "chunk_size" => 64, "corruption_type" => "noise",
                "corruption_amount" => 0.3, "training_epochs" => 100),
        params!("latent_dim" => 8, "chunk_size" => 128, "corruption_type" => "noise",
                "corruption_amount" => 0.5, "training_epochs" => 100),
        params!("latent_dim" => 16, "chunk_size" => 128, "corruption_type" => "noise",
                "corruption_amount" => 1.0, "training_epochs" => 200),
        params!("latent_dim" => 8, "chunk_size" => 128, "corruption_type" => "dropout",
                "corruption_amount" => 0.5, "training_epochs" => 150),
        params!("latent_dim" => 8, "chunk_size" => 128, "corruption_type" => "scale",
                "corruption_amount" => 2.0, "training_epochs" => 100),
        params!("latent_dim" => 32, "chunk_size" => 256, "corruption_type" => "noise",
                "corruption_amount" => 0.2, "training_epochs" => 300),
        params!("latent_dim" => 4, "chunk_size" => 64, "corruption_type" => "dropout",
                "corruption_amount" => 0.8, "training_epochs" => 50),
    ]
}

// ---------------------------------------------------------------------------
// K003 -- Echo State Network
// ---------------------------------------------------------------------------

fn process_k003(samples: &[f32], _sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let reservoir_size = pi(params, "reservoir_size", 100) as usize;
    let spectral_radius = pf(params, "spectral_radius", 0.95) as f64;
    let input_scaling = pf(params, "input_scaling", 0.5) as f64;
    let leak_rate = pf(params, "leak_rate", 0.3) as f64;

    let x: Vec<f64> = samples.iter().map(|&s| s as f64).collect();
    let n = x.len();

    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Input weights (reservoir_size x 1)
    let w_in: Vec<f64> = randn_vec(&mut rng, reservoir_size)
        .iter().map(|&v| v * input_scaling).collect();

    // Reservoir weights -- sparse, scaled to desired spectral radius
    let mut w_res = randn_mat(&mut rng, reservoir_size, reservoir_size);
    for v in w_res.iter_mut() { *v *= 0.1; }

    // Make it sparse: zero out ~90% of connections
    for i in 0..reservoir_size {
        for j in 0..reservoir_size {
            let r: f64 = rng.random();
            if r >= 0.1 {
                w_res[i * reservoir_size + j] = 0.0;
            }
        }
    }

    // Approximate spectral radius using power iteration (since we can't do
    // full eigendecomposition easily). We do 100 iterations of power method
    // to estimate the largest eigenvalue magnitude.
    let max_eigval = power_iteration_spectral_radius(&w_res, reservoir_size, 100, &mut rng);
    if max_eigval > 1e-10 {
        let scale = spectral_radius / max_eigval;
        for v in w_res.iter_mut() { *v *= scale; }
    }

    // Run reservoir
    let mut states = vec![0.0f64; n * reservoir_size]; // (n, reservoir_size) row-major
    let mut h = vec![0.0f64; reservoir_size];

    for i in 0..n {
        let u = x[i];
        // h_new = tanh(w_in * u + w_res @ h)
        let wres_h = mat_vec_mul(&w_res, reservoir_size, reservoir_size, &h);
        let mut h_new = vec![0.0f64; reservoir_size];
        for j in 0..reservoir_size {
            h_new[j] = (w_in[j] * u + wres_h[j]).tanh();
        }
        // Leaky integration
        for j in 0..reservoir_size {
            h[j] = (1.0 - leak_rate) * h[j] + leak_rate * h_new[j];
        }
        states[i * reservoir_size..(i + 1) * reservoir_size].copy_from_slice(&h);
    }

    // Train linear readout via least squares: x_target = states @ W_out
    // Use normal equations: W_out = (states^T @ states)^-1 @ states^T @ x
    // But for numerical stability, use pseudo-inverse via Cholesky or SVD.
    // We'll use a simple regularised normal equations approach:
    // (S^T S + lambda I) W_out = S^T x
    let w_out = least_squares_solve(&states, n, reservoir_size, &x, 1e-6);

    // Generate output
    let mut out = vec![0.0f64; n];
    for i in 0..n {
        let mut y = 0.0;
        for j in 0..reservoir_size {
            y += states[i * reservoir_size + j] * w_out[j];
        }
        out[i] = y;
    }

    AudioOutput::Mono(out.iter().map(|&x| x as f32).collect())
}

/// Power iteration to estimate the spectral radius (largest |eigenvalue|).
fn power_iteration_spectral_radius(
    mat: &[f64], n: usize, iters: usize, rng: &mut ChaCha8Rng,
) -> f64 {
    // Random initial vector
    let mut v: Vec<f64> = randn_vec(rng, n);
    let norm: f64 = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
    if norm > 1e-30 {
        for x in &mut v { *x /= norm; }
    }

    let mut eigenvalue = 0.0f64;
    for _ in 0..iters {
        let w = mat_vec_mul(mat, n, n, &v);
        let norm: f64 = w.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if norm < 1e-30 { break; }
        eigenvalue = norm;
        v = w.iter().map(|&x| x / norm).collect();
    }
    eigenvalue
}

/// Solve least squares: find w such that A @ w ~= b.
/// A is (m x n), b is (m,). Uses regularised normal equations with Cholesky.
fn least_squares_solve(a: &[f64], m: usize, n: usize, b: &[f64], lambda: f64) -> Vec<f64> {
    // ATA = A^T @ A + lambda * I  (n x n)
    let mut ata = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..m {
                sum += a[k * n + i] * a[k * n + j];
            }
            ata[i * n + j] = sum;
        }
        ata[i * n + i] += lambda; // Regularisation
    }

    // ATb = A^T @ b  (n,)
    let mut atb = vec![0.0f64; n];
    for i in 0..n {
        let mut sum = 0.0;
        for k in 0..m {
            sum += a[k * n + i] * b[k];
        }
        atb[i] = sum;
    }

    // Solve ATA @ w = ATb using Cholesky decomposition
    // If Cholesky fails, fall back to simple Gaussian elimination
    cholesky_solve(&ata, n, &atb)
}

/// Cholesky decomposition and solve: solves A @ x = b where A is SPD (n x n).
fn cholesky_solve(a: &[f64], n: usize, b: &[f64]) -> Vec<f64> {
    // L @ L^T = A
    let mut l = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[i * n + k] * l[j * n + k];
            }
            if i == j {
                let val = a[i * n + i] - sum;
                l[i * n + j] = if val > 0.0 { val.sqrt() } else { 1e-10 };
            } else {
                l[i * n + j] = (a[i * n + j] - sum) / l[j * n + j];
            }
        }
    }

    // Forward substitution: L @ y = b
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..i {
            sum += l[i * n + j] * y[j];
        }
        y[i] = (b[i] - sum) / l[i * n + i];
    }

    // Back substitution: L^T @ x = y
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut sum = 0.0;
        for j in (i + 1)..n {
            sum += l[j * n + i] * x[j];
        }
        x[i] = (y[i] - sum) / l[i * n + i];
    }
    x
}

fn variants_k003() -> Vec<HashMap<String, Value>> {
    vec![
        params!("reservoir_size" => 50, "spectral_radius" => 0.9,
                "input_scaling" => 0.3, "leak_rate" => 0.2),
        params!("reservoir_size" => 100, "spectral_radius" => 0.95,
                "input_scaling" => 0.5, "leak_rate" => 0.3),
        params!("reservoir_size" => 200, "spectral_radius" => 1.0,
                "input_scaling" => 1.0, "leak_rate" => 0.5),
        params!("reservoir_size" => 100, "spectral_radius" => 1.1,
                "input_scaling" => 0.5, "leak_rate" => 0.1),
        params!("reservoir_size" => 300, "spectral_radius" => 0.85,
                "input_scaling" => 0.2, "leak_rate" => 0.8),
        params!("reservoir_size" => 500, "spectral_radius" => 1.2,
                "input_scaling" => 2.0, "leak_rate" => 1.0),
    ]
}

// ---------------------------------------------------------------------------
// K004 -- Neural ODE
// ---------------------------------------------------------------------------

fn process_k004(samples: &[f32], _sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let hidden_size = pi(params, "hidden_size", 32) as usize;
    let dt = pf(params, "dt", 0.05) as f64;
    let num_steps = pi(params, "num_steps", 10) as usize;
    let seed = pu(params, "seed", 42);
    let chunk_size = 64usize;
    let hop_size = chunk_size / 2;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Two-layer MLP for the dynamics function f
    let scale1 = (2.0 / (chunk_size + hidden_size) as f64).sqrt();
    let w1: Vec<f64> = randn_mat(&mut rng, hidden_size, chunk_size)
        .iter().map(|&x| x * scale1).collect();
    let b1: Vec<f64> = randn_vec(&mut rng, hidden_size)
        .iter().map(|&x| x * 0.01).collect();
    let scale2 = (2.0 / (hidden_size + chunk_size) as f64).sqrt();
    let w2: Vec<f64> = randn_mat(&mut rng, chunk_size, hidden_size)
        .iter().map(|&x| x * scale2).collect();
    let b2: Vec<f64> = randn_vec(&mut rng, chunk_size)
        .iter().map(|&x| x * 0.01).collect();

    let x: Vec<f64> = samples.iter().map(|&s| s as f64).collect();

    let out = overlap_add(&x, chunk_size, hop_size, |chunk| {
        let mut state = chunk.to_vec();
        for _ in 0..num_steps {
            // dynamics: f(x) = W2 @ tanh(W1 @ x + b1) + b2
            let h_pre = vec_add(&mat_vec_mul(&w1, hidden_size, chunk_size, &state), &b1);
            let h = tanh_vec(&h_pre);
            let dstate = vec_add(&mat_vec_mul(&w2, chunk_size, hidden_size, &h), &b2);
            // Euler step: state = state + dt * dynamics(state)
            for i in 0..chunk_size {
                state[i] += dt * dstate[i];
            }
        }
        state
    });

    AudioOutput::Mono(out.iter().map(|&x| x as f32).collect())
}

fn variants_k004() -> Vec<HashMap<String, Value>> {
    vec![
        params!("hidden_size" => 16, "dt" => 0.01, "num_steps" => 5, "seed" => 1),
        params!("hidden_size" => 32, "dt" => 0.05, "num_steps" => 10, "seed" => 42),
        params!("hidden_size" => 64, "dt" => 0.02, "num_steps" => 20, "seed" => 7),
        params!("hidden_size" => 32, "dt" => 0.1, "num_steps" => 5, "seed" => 99),
        params!("hidden_size" => 16, "dt" => 0.05, "num_steps" => 50, "seed" => 13),
        params!("hidden_size" => 64, "dt" => 0.1, "num_steps" => 30, "seed" => 55),
    ]
}

// ---------------------------------------------------------------------------
// K005 -- Weight Space Interpolation
// ---------------------------------------------------------------------------

fn process_k005(samples: &[f32], _sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let hidden_size = pi(params, "hidden_size", 32) as usize;
    let num_layers = pi(params, "num_layers", 3) as usize;
    let seed_a = pu(params, "seed_a", 42);
    let seed_b = pu(params, "seed_b", 123);
    let alpha = pf(params, "alpha", 0.5) as f64;
    let chunk_size = 64usize;
    let hop_size = chunk_size / 2;

    let mut rng_a = ChaCha8Rng::seed_from_u64(seed_a);
    let mut rng_b = ChaCha8Rng::seed_from_u64(seed_b);

    // Build layer specs for both networks
    let mut weights_a: Vec<Vec<f64>> = Vec::new();
    let mut biases_a: Vec<Vec<f64>> = Vec::new();
    let mut weights_b: Vec<Vec<f64>> = Vec::new();
    let mut biases_b: Vec<Vec<f64>> = Vec::new();
    let mut layer_dims: Vec<(usize, usize)> = Vec::new(); // (fan_in, fan_out)

    for i in 0..num_layers {
        let (fan_in, fan_out) = if i == 0 {
            (chunk_size, hidden_size)
        } else if i == num_layers - 1 {
            (hidden_size, chunk_size)
        } else {
            (hidden_size, hidden_size)
        };
        layer_dims.push((fan_in, fan_out));

        let scale = (2.0 / (fan_in + fan_out) as f64).sqrt();
        let wa: Vec<f64> = randn_mat(&mut rng_a, fan_out, fan_in)
            .iter().map(|&x| x * scale).collect();
        let ba: Vec<f64> = randn_vec(&mut rng_a, fan_out)
            .iter().map(|&x| x * 0.01).collect();
        let wb: Vec<f64> = randn_mat(&mut rng_b, fan_out, fan_in)
            .iter().map(|&x| x * scale).collect();
        let bb: Vec<f64> = randn_vec(&mut rng_b, fan_out)
            .iter().map(|&x| x * 0.01).collect();

        weights_a.push(wa);
        biases_a.push(ba);
        weights_b.push(wb);
        biases_b.push(bb);
    }

    // Interpolate
    let mut weights: Vec<Vec<f64>> = Vec::new();
    let mut biases: Vec<Vec<f64>> = Vec::new();
    for i in 0..num_layers {
        let w: Vec<f64> = weights_a[i].iter().zip(weights_b[i].iter())
            .map(|(&a, &b)| (1.0 - alpha) * a + alpha * b).collect();
        let b: Vec<f64> = biases_a[i].iter().zip(biases_b[i].iter())
            .map(|(&a, &b)| (1.0 - alpha) * a + alpha * b).collect();
        weights.push(w);
        biases.push(b);
    }

    let x: Vec<f64> = samples.iter().map(|&s| s as f64).collect();

    let out = overlap_add(&x, chunk_size, hop_size, |chunk| {
        let mut h = chunk.to_vec();
        for i in 0..num_layers {
            let (fan_in, fan_out) = layer_dims[i];
            let pre = vec_add(
                &mat_vec_mul(&weights[i], fan_out, fan_in, &h),
                &biases[i],
            );
            if i < num_layers - 1 {
                h = tanh_vec(&pre);
            } else {
                h = pre;
            }
        }
        h
    });

    AudioOutput::Mono(out.iter().map(|&x| x as f32).collect())
}

fn variants_k005() -> Vec<HashMap<String, Value>> {
    vec![
        params!("hidden_size" => 16, "num_layers" => 2, "seed_a" => 1, "seed_b" => 2, "alpha" => 0.0),
        params!("hidden_size" => 32, "num_layers" => 3, "seed_a" => 42, "seed_b" => 123, "alpha" => 0.5),
        params!("hidden_size" => 32, "num_layers" => 3, "seed_a" => 42, "seed_b" => 123, "alpha" => 1.0),
        params!("hidden_size" => 64, "num_layers" => 4, "seed_a" => 10, "seed_b" => 20, "alpha" => 0.25),
        params!("hidden_size" => 32, "num_layers" => 2, "seed_a" => 7, "seed_b" => 77, "alpha" => 0.75),
        params!("hidden_size" => 64, "num_layers" => 4, "seed_a" => 5, "seed_b" => 500, "alpha" => 0.1),
    ]
}

// ---------------------------------------------------------------------------
// K006 -- 1D Convnet Filter Bank
// ---------------------------------------------------------------------------

fn process_k006(samples: &[f32], _sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let num_kernels = pi(params, "num_kernels", 8) as usize;
    let kernel_size = pi(params, "kernel_size", 16) as usize;
    let seed = pu(params, "seed", 42);

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let x: Vec<f64> = samples.iter().map(|&s| s as f64).collect();
    let n = x.len();

    // Generate random kernels, normalised so each has unit energy
    let mut kernels: Vec<Vec<f64>> = Vec::with_capacity(num_kernels);
    for _ in 0..num_kernels {
        let mut k = randn_vec(&mut rng, kernel_size);
        let norm: f64 = k.iter().map(|&v| v * v).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for v in &mut k { *v /= norm; }
        }
        kernels.push(k);
    }

    let mut out = vec![0.0f64; n];

    for k in &kernels {
        // Convolution mode='same': output length = n
        let conv = convolve_same(&x, k);
        // ReLU and accumulate
        for i in 0..n {
            out[i] += conv[i].max(0.0);
        }
    }

    // Normalise to roughly match input level
    for v in out.iter_mut() { *v /= num_kernels as f64; }

    let rms_in = (x.iter().map(|&v| v * v).sum::<f64>() / n as f64).sqrt() + 1e-10;
    let rms_out = (out.iter().map(|&v| v * v).sum::<f64>() / n as f64).sqrt() + 1e-10;
    let scale = rms_in / rms_out;
    for v in out.iter_mut() { *v *= scale; }

    AudioOutput::Mono(out.iter().map(|&x| x as f32).collect())
}

/// 1D convolution with mode='same' (output length equals input length).
fn convolve_same(x: &[f64], kernel: &[f64]) -> Vec<f64> {
    let n = x.len();
    let k = kernel.len();
    let mut out = vec![0.0f64; n];
    let half = k / 2;

    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..k {
            let xi = i as isize + j as isize - half as isize;
            if xi >= 0 && (xi as usize) < n {
                sum += x[xi as usize] * kernel[j];
            }
        }
        out[i] = sum;
    }
    out
}

fn variants_k006() -> Vec<HashMap<String, Value>> {
    vec![
        params!("num_kernels" => 4, "kernel_size" => 3, "seed" => 1),
        params!("num_kernels" => 8, "kernel_size" => 16, "seed" => 42),
        params!("num_kernels" => 16, "kernel_size" => 8, "seed" => 7),
        params!("num_kernels" => 8, "kernel_size" => 64, "seed" => 99),
        params!("num_kernels" => 32, "kernel_size" => 32, "seed" => 13),
        params!("num_kernels" => 4, "kernel_size" => 64, "seed" => 55),
    ]
}

// ---------------------------------------------------------------------------
// K007 -- Tiny RNN Sample Processor
// ---------------------------------------------------------------------------

fn process_k007(samples: &[f32], _sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let hidden_size = pi(params, "hidden_size", 8) as usize;
    let seed = pu(params, "seed", 42);

    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Input->hidden weights (1 input)
    let scale_ih = (2.0 / (1 + hidden_size) as f64).sqrt();
    let w_ih: Vec<f64> = randn_vec(&mut rng, hidden_size)
        .iter().map(|&x| x * scale_ih).collect();
    // Hidden->hidden weights
    let scale_hh = (2.0 / (hidden_size + hidden_size) as f64).sqrt();
    let w_hh: Vec<f64> = randn_mat(&mut rng, hidden_size, hidden_size)
        .iter().map(|&x| x * scale_hh).collect();
    // Bias
    let b_h: Vec<f64> = randn_vec(&mut rng, hidden_size)
        .iter().map(|&x| x * 0.01).collect();
    // Hidden->output weights (1 output)
    let scale_ho = (2.0 / (hidden_size + 1) as f64).sqrt();
    let w_ho: Vec<f64> = randn_vec(&mut rng, hidden_size)
        .iter().map(|&x| x * scale_ho).collect();

    let x: Vec<f64> = samples.iter().map(|&s| s as f64).collect();
    let n = x.len();

    // RNN process sample by sample
    let mut out = vec![0.0f64; n];
    let mut h = vec![0.0f64; hidden_size];

    for i in 0..n {
        let xi = x[i];
        let mut h_new = vec![0.0f64; hidden_size];
        for j in 0..hidden_size {
            let mut val = w_ih[j] * xi + b_h[j];
            for k in 0..hidden_size {
                val += w_hh[j * hidden_size + k] * h[k];
            }
            h_new[j] = val.tanh();
        }
        h = h_new;

        let mut y = 0.0;
        for j in 0..hidden_size {
            y += w_ho[j] * h[j];
        }
        out[i] = y;
    }

    // Normalise output RMS to match input
    let rms_in = (x.iter().map(|&v| v * v).sum::<f64>() / n as f64).sqrt() + 1e-10;
    let rms_out = (out.iter().map(|&v| v * v).sum::<f64>() / n as f64).sqrt() + 1e-10;
    let scale = rms_in / rms_out;
    for v in out.iter_mut() { *v *= scale; }

    AudioOutput::Mono(out.iter().map(|&x| x as f32).collect())
}

fn variants_k007() -> Vec<HashMap<String, Value>> {
    vec![
        params!("hidden_size" => 4, "seed" => 1),
        params!("hidden_size" => 8, "seed" => 42),
        params!("hidden_size" => 16, "seed" => 7),
        params!("hidden_size" => 32, "seed" => 99),
        params!("hidden_size" => 8, "seed" => 13),
        params!("hidden_size" => 4, "seed" => 200),
    ]
}

// ---------------------------------------------------------------------------
// K008 -- Overfit-Then-Corrupt
// ---------------------------------------------------------------------------

fn process_k008(samples: &[f32], _sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let hidden_size = pi(params, "hidden_size", 32) as usize;
    let chunk_size = pi(params, "chunk_size", 64) as usize;
    let corruption_type = params.get("corruption_type")
        .and_then(|v| v.as_str())
        .unwrap_or("weight_noise")
        .to_string();
    let corruption_amount = pf(params, "corruption_amount", 0.5) as f64;
    let train_epochs = pi(params, "train_epochs", 100) as usize;
    let hop_size = chunk_size / 2;

    let x: Vec<f64> = samples.iter().map(|&s| s as f64).collect();

    // Collect training chunks
    let window = hann_window_f64(chunk_size);
    let mut chunks: Vec<Vec<f64>> = Vec::new();
    let mut pos = 0;
    while pos + chunk_size <= x.len() {
        let c: Vec<f64> = (0..chunk_size).map(|i| x[pos + i] * window[i]).collect();
        chunks.push(c);
        pos += hop_size;
    }
    if chunks.is_empty() {
        return AudioOutput::Mono(samples.to_vec());
    }

    let num_chunks = chunks.len();
    // Flatten chunks into data matrix (num_chunks x chunk_size), row-major
    let mut data = vec![0.0f64; num_chunks * chunk_size];
    for (i, c) in chunks.iter().enumerate() {
        data[i * chunk_size..(i + 1) * chunk_size].copy_from_slice(c);
    }

    // Init 2-layer MLP: chunk_size -> hidden -> chunk_size
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let scale1 = (2.0 / (chunk_size + hidden_size) as f64).sqrt();
    let mut w1: Vec<f64> = randn_mat(&mut rng, hidden_size, chunk_size)
        .iter().map(|&x| x * scale1).collect();
    let mut b1 = vec![0.0f64; hidden_size];
    let scale2 = (2.0 / (hidden_size + chunk_size) as f64).sqrt();
    let mut w2: Vec<f64> = randn_mat(&mut rng, chunk_size, hidden_size)
        .iter().map(|&x| x * scale2).collect();
    let mut b2 = vec![0.0f64; chunk_size];

    let lr = 0.001;

    // Train (overfit)
    for _epoch in 0..train_epochs {
        // Forward: H = tanh(data @ W1.T + b1), recon = H @ W2.T + b2
        let w1_t = mat_transpose(&w1, hidden_size, chunk_size); // (chunk_size, hidden_size)
        let h_pre = mat_mat_mul(&data, num_chunks, chunk_size, &w1_t, hidden_size);
        let h_pre = mat_add_bias(&h_pre, num_chunks, hidden_size, &b1);
        let h = tanh_vec(&h_pre); // (N * hidden_size)

        let w2_t = mat_transpose(&w2, chunk_size, hidden_size); // (hidden_size, chunk_size)
        let recon = mat_mat_mul(&h, num_chunks, hidden_size, &w2_t, chunk_size);
        let recon = mat_add_bias(&recon, num_chunks, chunk_size, &b2);

        let error = elem_sub(&recon, &data);
        let d_recon = vec_scale(&error, 2.0 / num_chunks as f64);

        // Decoder grads
        let d_recon_t = mat_transpose(&d_recon, num_chunks, chunk_size);
        let dw2 = mat_mat_mul(&d_recon_t, chunk_size, num_chunks, &h, hidden_size);
        let db2 = col_sum(&d_recon, num_chunks, chunk_size);

        // Encoder grads
        let dh = mat_mat_mul(&d_recon, num_chunks, chunk_size, &w2, hidden_size);
        let h_deriv = tanh_deriv(&h);
        let dh_pre = elem_mul(&dh, &h_deriv);
        let dh_pre_t = mat_transpose(&dh_pre, num_chunks, hidden_size);
        let dw1 = mat_mat_mul(&dh_pre_t, hidden_size, num_chunks, &data, chunk_size);
        let db1 = col_sum(&dh_pre, num_chunks, hidden_size);

        for i in 0..w2.len() { w2[i] -= lr * dw2[i]; }
        for i in 0..b2.len() { b2[i] -= lr * db2[i]; }
        for i in 0..w1.len() { w1[i] -= lr * dw1[i]; }
        for i in 0..b1.len() { b1[i] -= lr * db1[i]; }
    }

    // Corrupt weights
    let mut corrupt_rng = ChaCha8Rng::seed_from_u64(42);
    match corruption_type.as_str() {
        "weight_noise" => {
            let std_w1 = std_dev(&w1);
            let std_w2 = std_dev(&w2);
            let std_b1 = std_dev(&b1) + 1e-10;
            let std_b2 = std_dev(&b2) + 1e-10;
            for v in w1.iter_mut() {
                *v += randn(&mut corrupt_rng) * corruption_amount * std_w1;
            }
            for v in w2.iter_mut() {
                *v += randn(&mut corrupt_rng) * corruption_amount * std_w2;
            }
            for v in b1.iter_mut() {
                *v += randn(&mut corrupt_rng) * corruption_amount * std_b1;
            }
            for v in b2.iter_mut() {
                *v += randn(&mut corrupt_rng) * corruption_amount * std_b2;
            }
        }
        "quantize" => {
            let num_levels = (16.0 / corruption_amount).round().max(2.0) as usize;
            // Quantize each array
            quantize_array(&mut w1, num_levels);
            quantize_array(&mut w2, num_levels);
            quantize_array(&mut b1, num_levels);
            quantize_array(&mut b2, num_levels);
        }
        "prune" => {
            // Zero out a fraction of weights
            for v in w1.iter_mut() {
                let r: f64 = corrupt_rng.random();
                if r <= corruption_amount {
                    *v = 0.0;
                }
            }
            for v in w2.iter_mut() {
                let r: f64 = corrupt_rng.random();
                if r <= corruption_amount {
                    *v = 0.0;
                }
            }
        }
        _ => {}
    }

    let out = overlap_add(&x, chunk_size, hop_size, |chunk| {
        let h_pre = vec_add(&mat_vec_mul(&w1, hidden_size, chunk_size, chunk), &b1);
        let h = tanh_vec(&h_pre);
        vec_add(&mat_vec_mul(&w2, chunk_size, hidden_size, &h), &b2)
    });

    AudioOutput::Mono(out.iter().map(|&x| x as f32).collect())
}

/// Quantize array values to a fixed number of levels.
fn quantize_array(arr: &mut [f64], num_levels: usize) {
    let mn = arr.iter().cloned().fold(f64::INFINITY, f64::min);
    let mx = arr.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if mx - mn > 1e-10 {
        let nl = num_levels as f64;
        for v in arr.iter_mut() {
            let norm = (*v - mn) / (mx - mn);
            let q = (norm * nl).round() / nl;
            *v = q * (mx - mn) + mn;
        }
    }
}

fn variants_k008() -> Vec<HashMap<String, Value>> {
    vec![
        params!("hidden_size" => 16, "chunk_size" => 64, "corruption_type" => "weight_noise",
                "corruption_amount" => 0.3, "train_epochs" => 100),
        params!("hidden_size" => 32, "chunk_size" => 64, "corruption_type" => "weight_noise",
                "corruption_amount" => 0.5, "train_epochs" => 100),
        params!("hidden_size" => 32, "chunk_size" => 128, "corruption_type" => "weight_noise",
                "corruption_amount" => 1.5, "train_epochs" => 200),
        params!("hidden_size" => 32, "chunk_size" => 64, "corruption_type" => "quantize",
                "corruption_amount" => 0.5, "train_epochs" => 100),
        params!("hidden_size" => 32, "chunk_size" => 64, "corruption_type" => "prune",
                "corruption_amount" => 0.7, "train_epochs" => 100),
        params!("hidden_size" => 64, "chunk_size" => 256, "corruption_type" => "quantize",
                "corruption_amount" => 2.0, "train_epochs" => 150),
        params!("hidden_size" => 16, "chunk_size" => 64, "corruption_type" => "prune",
                "corruption_amount" => 0.9, "train_epochs" => 50),
    ]
}

// ---------------------------------------------------------------------------
// K009 -- Random Projection
// ---------------------------------------------------------------------------

fn process_k009(samples: &[f32], _sr: u32, params: &HashMap<String, Value>) -> AudioOutput {
    let chunk_size = pi(params, "chunk_size", 64) as usize;
    let bottleneck_dim = pi(params, "bottleneck_dim", 8) as usize;
    let seed = pu(params, "seed", 42);
    let hop_size = chunk_size / 2;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Random projection matrices (orthogonalised for stability)
    // A_raw: (bottleneck_dim x chunk_size)
    let a_raw = randn_mat(&mut rng, bottleneck_dim, chunk_size);
    // Orthogonalise via QR on A_raw.T (chunk_size x bottleneck_dim)
    let a_raw_t = mat_transpose(&a_raw, bottleneck_dim, chunk_size);
    let q1 = qr_q(&a_raw_t, chunk_size, bottleneck_dim);
    // A = Q1[:, :bottleneck_dim].T -> (bottleneck_dim, chunk_size)
    let a = mat_transpose(&q1, chunk_size, bottleneck_dim);

    // B_raw: (chunk_size x bottleneck_dim)
    let b_raw = randn_mat(&mut rng, chunk_size, bottleneck_dim);
    let q2 = qr_q(&b_raw, chunk_size, bottleneck_dim);
    // B = Q2[:, :bottleneck_dim] -> (chunk_size, bottleneck_dim)
    let b = q2; // already (chunk_size x bottleneck_dim)

    let x: Vec<f64> = samples.iter().map(|&s| s as f64).collect();

    let out = overlap_add(&x, chunk_size, hop_size, |chunk| {
        // latent = A @ chunk  (bottleneck_dim)
        let latent = mat_vec_mul(&a, bottleneck_dim, chunk_size, chunk);
        // result = B @ latent (chunk_size)
        mat_vec_mul(&b, chunk_size, bottleneck_dim, &latent)
    });

    AudioOutput::Mono(out.iter().map(|&x| x as f32).collect())
}

fn variants_k009() -> Vec<HashMap<String, Value>> {
    vec![
        params!("chunk_size" => 32, "bottleneck_dim" => 2, "seed" => 1),
        params!("chunk_size" => 64, "bottleneck_dim" => 8, "seed" => 42),
        params!("chunk_size" => 64, "bottleneck_dim" => 4, "seed" => 7),
        params!("chunk_size" => 128, "bottleneck_dim" => 16, "seed" => 99),
        params!("chunk_size" => 256, "bottleneck_dim" => 32, "seed" => 13),
        params!("chunk_size" => 64, "bottleneck_dim" => 2, "seed" => 55),
    ]
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

pub fn register() -> Vec<EffectEntry> {
    vec![
        EffectEntry {
            id: "K001",
            process: process_k001,
            variants: variants_k001,
            category: "neural",
        },
        EffectEntry {
            id: "K002",
            process: process_k002,
            variants: variants_k002,
            category: "neural",
        },
        EffectEntry {
            id: "K003",
            process: process_k003,
            variants: variants_k003,
            category: "neural",
        },
        EffectEntry {
            id: "K004",
            process: process_k004,
            variants: variants_k004,
            category: "neural",
        },
        EffectEntry {
            id: "K005",
            process: process_k005,
            variants: variants_k005,
            category: "neural",
        },
        EffectEntry {
            id: "K006",
            process: process_k006,
            variants: variants_k006,
            category: "neural",
        },
        EffectEntry {
            id: "K007",
            process: process_k007,
            variants: variants_k007,
            category: "neural",
        },
        EffectEntry {
            id: "K008",
            process: process_k008,
            variants: variants_k008,
            category: "neural",
        },
        EffectEntry {
            id: "K009",
            process: process_k009,
            variants: variants_k009,
            category: "neural",
        },
    ]
}
