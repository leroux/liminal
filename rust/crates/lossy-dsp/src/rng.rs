//! Numpy-compatible Mersenne Twister RNG.
//!
//! Wraps `rand_mt::Mt` (MT19937) to produce the same float64 sequence
//! as `numpy.random.RandomState(seed)`.

use rand_mt::Mt;

/// Numpy-compatible MT19937 RNG.
///
/// Produces the same sequence as `numpy.random.RandomState(seed)`.
pub struct NumpyRng {
    mt: Mt,
}

impl NumpyRng {
    /// Create a new RNG with the given seed, matching `np.random.RandomState(seed)`.
    pub fn new(seed: u32) -> Self {
        Self { mt: Mt::new(seed) }
    }

    /// Generate a uniform f64 in [0, 1), matching `rng.random()`.
    ///
    /// Uses genrand_res53: `(a >> 5) * 2^26 + (b >> 6) / 2^53`
    pub fn random(&mut self) -> f64 {
        let a = self.mt.next_u32() >> 5;
        let b = self.mt.next_u32() >> 6;
        (a as f64 * 67108864.0 + b as f64) / 9007199254740992.0
    }

    /// Generate a uniform f64 in [low, high), matching `rng.uniform(low, high)`.
    pub fn uniform(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.random()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matches_numpy_random() {
        let mut rng = NumpyRng::new(42);
        let expected = [
            0.37454011884736249094_f64,
            0.95071430640991616556,
            0.73199394181140509108,
            0.59865848419703659999,
            0.15601864044243651808,
            0.15599452033620264668,
            0.05808361216819946105,
            0.86617614577493518002,
            0.60111501174320880470,
            0.70807257779604548809,
        ];
        for (i, &exp) in expected.iter().enumerate() {
            let val = rng.random();
            assert!(
                (val - exp).abs() < 1e-18,
                "mismatch at [{i}]: rust={val:.20} numpy={exp:.20}"
            );
        }
    }

    #[test]
    fn test_matches_numpy_uniform() {
        let mut rng = NumpyRng::new(42);
        let pi = std::f64::consts::PI;
        let expected = [
            -0.78828768189874909300_f64,
            2.83192150777042339627,
            1.45766092654409629148,
            0.61988953833542981275,
            -2.16129862431574126802,
        ];
        for (i, &exp) in expected.iter().enumerate() {
            let val = rng.uniform(-pi, pi);
            assert!(
                (val - exp).abs() < 1e-12,
                "mismatch at [{i}]: rust={val:.20} numpy={exp:.20}"
            );
        }
    }
}
