//! Scale definitions for parameter and stats grid dimensions.
//!
//! Scales define how to generate values for grid search dimensions.
//! They support various progressions useful for benchmarking:
//! - Logarithmic (powers of 2, 10, etc.) for size parameters
//! - Linear for evenly-spaced values
//! - Steps for floating-point ranges (like density 0.0 to 1.0)
//! - Explicit for custom value lists

use serde::Deserialize;
use serde::Serialize;

/// Defines how to generate values for a grid dimension.
///
/// # Example
///
/// ```
/// use vortex_threshold_traits::Scale;
///
/// // Powers of 2 from 64 to 1M
/// let size_scale = Scale::log2(6, 20);
/// assert_eq!(size_scale.values().collect::<Vec<_>>(),
///     vec![64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0,
///          16384.0, 32768.0, 65536.0, 131072.0, 262144.0, 524288.0, 1048576.0]);
///
/// // Density from 0.0 to 1.0 in 0.1 steps
/// let density_scale = Scale::steps(0.0, 1.0, 11);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Scale {
    /// Powers of 2: 2^start_exp, 2^(start_exp+1), ..., 2^end_exp
    Log2 {
        /// Starting exponent (inclusive)
        start_exp: u32,
        /// Ending exponent (inclusive)
        end_exp: u32,
    },

    /// Powers of arbitrary base: base^start_exp, ..., base^end_exp
    Log {
        /// Base for exponentiation
        base: f64,
        /// Starting exponent (inclusive)
        start_exp: i32,
        /// Ending exponent (inclusive)
        end_exp: i32,
    },

    /// Linear progression: start, start+step, start+2*step, ..., end
    Linear {
        /// Starting value (inclusive)
        start: f64,
        /// Ending value (inclusive)
        end: f64,
        /// Step size between values
        step: f64,
    },

    /// Fixed number of evenly-spaced steps between start and end.
    /// Useful for floating-point ranges like density.
    Steps {
        /// Starting value (inclusive)
        start: f64,
        /// Ending value (inclusive)
        end: f64,
        /// Number of values to generate
        count: usize,
    },

    /// Explicit list of values.
    Explicit(Vec<f64>),
}

impl Scale {
    /// Creates a log2 scale (powers of 2).
    ///
    /// Generates: 2^start_exp, 2^(start_exp+1), ..., 2^end_exp
    ///
    /// # Example
    ///
    /// ```
    /// use vortex_threshold_traits::Scale;
    ///
    /// let scale = Scale::log2(6, 10);
    /// let values: Vec<_> = scale.values().collect();
    /// assert_eq!(values, vec![64.0, 128.0, 256.0, 512.0, 1024.0]);
    /// ```
    #[must_use]
    pub fn log2(start_exp: u32, end_exp: u32) -> Self {
        Self::Log2 { start_exp, end_exp }
    }

    /// Creates a logarithmic scale with arbitrary base.
    ///
    /// Generates: base^start_exp, base^(start_exp+1), ..., base^end_exp
    #[must_use]
    pub fn log(base: f64, start_exp: i32, end_exp: i32) -> Self {
        Self::Log {
            base,
            start_exp,
            end_exp,
        }
    }

    /// Creates a linear scale.
    ///
    /// Generates: start, start+step, start+2*step, ..., up to end
    #[must_use]
    pub fn linear(start: f64, end: f64, step: f64) -> Self {
        Self::Linear { start, end, step }
    }

    /// Creates a scale with a fixed number of evenly-spaced steps.
    ///
    /// Useful for floating-point ranges like density (0.0 to 1.0).
    ///
    /// # Example
    ///
    /// ```
    /// use vortex_threshold_traits::Scale;
    ///
    /// let scale = Scale::steps(0.0, 1.0, 5);
    /// let values: Vec<_> = scale.values().collect();
    /// assert_eq!(values, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
    /// ```
    #[must_use]
    pub fn steps(start: f64, end: f64, count: usize) -> Self {
        Self::Steps { start, end, count }
    }

    /// Creates a scale from explicit values.
    #[must_use]
    pub fn explicit(values: Vec<f64>) -> Self {
        Self::Explicit(values)
    }

    /// Creates a scale from explicit integer values.
    #[must_use]
    pub fn explicit_usize(values: Vec<usize>) -> Self {
        Self::Explicit(values.into_iter().map(|v| v as f64).collect())
    }

    /// Returns an iterator over all values in this scale.
    pub fn values(&self) -> Box<dyn Iterator<Item = f64> + '_> {
        match self {
            Self::Log2 { start_exp, end_exp } => {
                let start = *start_exp;
                let end = *end_exp;
                Box::new((start..=end).map(|exp| 2f64.powi(exp as i32)))
            }
            Self::Log {
                base,
                start_exp,
                end_exp,
            } => {
                let base = *base;
                let start = *start_exp;
                let end = *end_exp;
                Box::new((start..=end).map(move |exp| base.powi(exp)))
            }
            Self::Linear { start, end, step } => {
                let start = *start;
                let end = *end;
                let step = *step;
                Box::new(std::iter::successors(Some(start), move |&prev| {
                    let next = prev + step;
                    // tolerance for float comparison
                    (next <= end + step * 0.5).then_some(next)
                }))
            }
            Self::Steps { start, end, count } => {
                let start = *start;
                let end = *end;
                let count = *count;
                if count <= 1 {
                    Box::new(std::iter::once(start))
                } else {
                    let step = (end - start) / (count - 1) as f64;
                    Box::new((0..count).map(move |i| start + step * i as f64))
                }
            }
            Self::Explicit(values) => Box::new(values.iter().copied()),
        }
    }

    /// Returns an iterator over all values as usize (truncated).
    #[allow(clippy::cast_possible_truncation)]
    pub fn values_usize(&self) -> Box<dyn Iterator<Item = usize> + '_> {
        Box::new(self.values().map(|v| v as usize))
    }

    /// Returns the number of values in this scale.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn len(&self) -> usize {
        match self {
            Self::Log2 { start_exp, end_exp } => (end_exp - start_exp + 1) as usize,
            Self::Log {
                start_exp, end_exp, ..
            } => (end_exp - start_exp + 1) as usize,
            Self::Linear { start, end, step } => {
                if *step <= 0.0 {
                    0
                } else {
                    ((end - start) / step).floor() as usize + 1
                }
            }
            Self::Steps { count, .. } => *count,
            Self::Explicit(values) => values.len(),
        }
    }

    /// Returns true if this scale has no values.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log2_scale() {
        let scale = Scale::log2(6, 10);
        let values: Vec<_> = scale.values().collect();
        assert_eq!(values, vec![64.0, 128.0, 256.0, 512.0, 1024.0]);
        assert_eq!(scale.len(), 5);
    }

    #[test]
    fn test_log_scale() {
        let scale = Scale::log(10.0, 0, 3);
        let values: Vec<_> = scale.values().collect();
        assert_eq!(values, vec![1.0, 10.0, 100.0, 1000.0]);
    }

    #[test]
    fn test_linear_scale() {
        let scale = Scale::linear(0.0, 100.0, 25.0);
        let values: Vec<_> = scale.values().collect();
        assert_eq!(values, vec![0.0, 25.0, 50.0, 75.0, 100.0]);
        assert_eq!(scale.len(), 5);
    }

    #[test]
    fn test_steps_scale() {
        let scale = Scale::steps(0.0, 1.0, 5);
        let values: Vec<_> = scale.values().collect();
        assert_eq!(values, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
        assert_eq!(scale.len(), 5);
    }

    #[test]
    fn test_steps_scale_single() {
        let scale = Scale::steps(0.5, 0.5, 1);
        let values: Vec<_> = scale.values().collect();
        assert_eq!(values, vec![0.5]);
    }

    #[test]
    fn test_explicit_scale() {
        let scale = Scale::explicit(vec![1.0, 10.0, 100.0, 1000.0]);
        let values: Vec<_> = scale.values().collect();
        assert_eq!(values, vec![1.0, 10.0, 100.0, 1000.0]);
    }

    #[test]
    fn test_values_usize() {
        let scale = Scale::log2(6, 8);
        let values: Vec<_> = scale.values_usize().collect();
        assert_eq!(values, vec![64, 128, 256]);
    }
}
