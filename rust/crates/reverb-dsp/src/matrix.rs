//! Feedback matrix constructors for the FDN reverb.
//!
//! All matrices are unitary (energy-preserving) unless noted.
//! Ported from `reverb/primitives/matrix.py`.

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Build a named matrix type. Returns flattened row-major N×N.
pub fn get_matrix(name: &str, n: usize, seed: i32) -> Vec<f64> {
    match name {
        "householder" => householder(n),
        "hadamard" => hadamard(n),
        "diagonal" => diagonal(n),
        "random_orthogonal" => random_orthogonal(n, seed),
        "circulant" => circulant_shift(n),
        "stautner_puckette" => stautner_puckette(n),
        "zero" => vec![0.0; n * n],
        _ => householder(n),
    }
}

/// Householder reflection: A = I - (2/N) * ones * ones^T
///
/// Uniform coupling. Standard FDN choice since Jot 1991.
pub fn householder(n: usize) -> Vec<f64> {
    let mut m = vec![0.0; n * n];
    let scale = 2.0 / n as f64;
    for i in 0..n {
        for j in 0..n {
            m[i * n + j] = if i == j { 1.0 - scale } else { -scale };
        }
    }
    m
}

/// Hadamard matrix (normalized to be unitary). Requires n = power of 2.
pub fn hadamard(n: usize) -> Vec<f64> {
    assert!(n > 0 && (n & (n - 1)) == 0, "n must be power of 2, got {n}");
    // The recursive construction already normalizes at each level
    hadamard_recursive(n)
}

fn hadamard_recursive(n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![1.0];
    }
    let half = n / 2;
    let h = hadamard_recursive(half);
    let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
    let mut m = vec![0.0; n * n];
    for i in 0..half {
        for j in 0..half {
            let v = h[i * half + j] * inv_sqrt2;
            m[i * n + j] = v;           // top-left: H
            m[i * n + half + j] = v;    // top-right: H
            m[(half + i) * n + j] = v;  // bottom-left: H
            m[(half + i) * n + half + j] = -v; // bottom-right: -H
        }
    }
    m
}

/// Identity matrix — no coupling.
pub fn diagonal(n: usize) -> Vec<f64> {
    let mut m = vec![0.0; n * n];
    for i in 0..n {
        m[i * n + i] = 1.0;
    }
    m
}

/// Random orthogonal matrix via QR decomposition with determinant correction.
///
/// Uses a ChaCha8 PRNG seeded deterministically, then does a Householder QR
/// decomposition and corrects the sign to ensure det = +1.
///
/// Note: Python uses numpy.random.RandomState(seed) which is MT19937, then
/// QR via LAPACK. We replicate the same *algorithm* (QR + sign correction)
/// but not bit-for-bit output since the RNG differs. For preset compatibility
/// this is fine — random_orthogonal matrices are only used for their structural
/// properties (unitarity), not exact element values.
pub fn random_orthogonal(n: usize, seed: i32) -> Vec<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed as u64);

    // Generate random N×N matrix with normal distribution (Box-Muller)
    let mut a = vec![0.0f64; n * n];
    for i in 0..n * n {
        let u1: f64 = loop {
            let v: f64 = rng.random();
            if v > 0.0 { break v; }
        };
        let u2: f64 = rng.random();
        a[i] = (-2.0_f64 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    }

    // Householder QR decomposition
    let (q, r) = qr_decomposition(&a, n);

    // Sign correction: Q *= sign(diag(R)) so that det(Q) = +1
    let mut result = q;
    for j in 0..n {
        let d = r[j * n + j];
        if d < 0.0 {
            for i in 0..n {
                result[i * n + j] = -result[i * n + j];
            }
        }
    }
    result
}

/// Householder QR decomposition. Returns (Q, R) as flat row-major n×n.
fn qr_decomposition(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut r = a.to_vec();
    let mut q = diagonal(n);

    for k in 0..n {
        // Extract column k below diagonal
        let mut x = vec![0.0; n - k];
        for i in k..n {
            x[i - k] = r[i * n + k];
        }

        // Compute Householder vector
        let norm_x = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm_x < 1e-15 {
            continue;
        }
        let sign = if x[0] >= 0.0 { 1.0 } else { -1.0 };
        x[0] += sign * norm_x;
        let norm_v = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm_v < 1e-15 {
            continue;
        }
        for v in x.iter_mut() {
            *v /= norm_v;
        }

        // Apply H = I - 2*v*v^T to R from left (rows k..n)
        for j in k..n {
            let mut dot = 0.0;
            for i in k..n {
                dot += x[i - k] * r[i * n + j];
            }
            for i in k..n {
                r[i * n + j] -= 2.0 * x[i - k] * dot;
            }
        }

        // Apply H to Q from right (all rows, columns k..n)
        for i in 0..n {
            let mut dot = 0.0;
            for j in k..n {
                dot += q[i * n + j] * x[j - k];
            }
            for j in k..n {
                q[i * n + j] -= 2.0 * dot * x[j - k];
            }
        }
    }

    (q, r)
}

/// Circulant permutation — each node feeds into the next in a ring.
pub fn circulant_shift(n: usize) -> Vec<f64> {
    let mut m = vec![0.0; n * n];
    for i in 0..n {
        m[((i + 1) % n) * n + i] = 1.0;
    }
    m
}

/// Stautner-Puckette style matrix — pairs of nodes cross-coupled at π/4.
pub fn stautner_puckette(n: usize) -> Vec<f64> {
    assert!(n % 2 == 0, "n must be even, got {n}");
    let mut m = vec![0.0; n * n];
    let angle = std::f64::consts::FRAC_PI_4;
    let c = angle.cos();
    let s = angle.sin();
    for i in (0..n).step_by(2) {
        m[i * n + i] = c;
        m[i * n + i + 1] = s;
        m[(i + 1) * n + i] = -s;
        m[(i + 1) * n + i + 1] = c;
    }
    m
}

/// Check if flattened N×N matrix `m` is approximately householder.
/// Used to select the O(N) fast path.
pub fn is_householder(m: &[f64], n: usize) -> bool {
    let expected = householder(n);
    m.iter()
        .zip(expected.iter())
        .all(|(a, b)| (a - b).abs() < 1e-10)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mat_mul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
        let mut c = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    c[i * n + j] += a[i * n + k] * b[k * n + j];
                }
            }
        }
        c
    }

    fn is_approx_identity(m: &[f64], n: usize, tol: f64) -> bool {
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                if (m[i * n + j] - expected).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    fn transpose(m: &[f64], n: usize) -> Vec<f64> {
        let mut t = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                t[j * n + i] = m[i * n + j];
            }
        }
        t
    }

    #[test]
    fn test_householder_unitary() {
        let m = householder(8);
        let mt = transpose(&m, 8);
        let product = mat_mul(&m, &mt, 8);
        assert!(is_approx_identity(&product, 8, 1e-10));
    }

    #[test]
    fn test_hadamard_unitary() {
        let m = hadamard(8);
        let mt = transpose(&m, 8);
        let product = mat_mul(&m, &mt, 8);
        assert!(is_approx_identity(&product, 8, 1e-10));
    }

    #[test]
    fn test_random_orthogonal_unitary() {
        let m = random_orthogonal(8, 42);
        let mt = transpose(&m, 8);
        let product = mat_mul(&m, &mt, 8);
        assert!(is_approx_identity(&product, 8, 1e-6));
    }

    #[test]
    fn test_circulant_permutation() {
        let m = circulant_shift(4);
        // Node 0 feeds node 1, node 1 feeds node 2, etc.
        assert_eq!(m[1 * 4 + 0], 1.0); // (1,0) = 1
        assert_eq!(m[2 * 4 + 1], 1.0); // (2,1) = 1
        assert_eq!(m[3 * 4 + 2], 1.0); // (3,2) = 1
        assert_eq!(m[0 * 4 + 3], 1.0); // (0,3) = 1
    }

    #[test]
    fn test_stautner_puckette_unitary() {
        let m = stautner_puckette(8);
        let mt = transpose(&m, 8);
        let product = mat_mul(&m, &mt, 8);
        assert!(is_approx_identity(&product, 8, 1e-10));
    }

    #[test]
    fn test_is_householder() {
        assert!(is_householder(&householder(8), 8));
        assert!(!is_householder(&hadamard(8), 8));
    }

    #[test]
    fn test_deterministic_random_orthogonal() {
        let m1 = random_orthogonal(8, 42);
        let m2 = random_orthogonal(8, 42);
        assert_eq!(m1, m2);
        let m3 = random_orthogonal(8, 137);
        assert_ne!(m1, m3);
    }
}
