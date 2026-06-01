// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Max-Lloyd centroid computation for TurboQuant scalar quantizers.
//!
//! Pre-computes and caches optimal scalar quantizer centroids for the marginal distribution of
//! coordinates after a random orthogonal transform of a unit-norm vector.
//!
//! In high dimensions, each coordinate of a randomly transformed unit vector follows a
//! distribution proportional to `(1 - x^2)^((d-3)/2)` on `[-1, 1]`, which converges to
//! `N(0, 1/d)`.
//!
//! The Max-Lloyd algorithm finds optimal quantization centroids that minimize MSE for this
//! distribution.
//!
//! Centroids are not stored in TurboQuant arrays. They are deterministically derived from
//! `(padded_dim, bit_width)` and cached process-locally.
//!
//! The centroid model follows the random orthogonal transform marginal used by the TurboQuant
//! paper. This encoder applies a SORF-style structured transform instead of a dense random Gaussian
//! or orthogonal matrix, so paper-level error bounds should not be treated as verified for this
//! implementation without separate empirical validation.

use std::sync::LazyLock;

use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_utils::aliases::dash_map::DashMap;

use crate::encodings::turboquant::MAX_BIT_WIDTH;
use crate::encodings::turboquant::MIN_DIMENSION;

// NB: All of these constants were chosen arbitrarily.

/// The maximum iterations for Max-Lloyd algorithm when computing centroids.
const MAX_ITERATIONS: usize = 200;

/// The Max-Lloyd convergence threshold for stopping early when computing centroids.
const CONVERGENCE_EPSILON: f64 = 1e-12;

/// Number of trapezoids used for numerical integration when computing conditional expectations.
///
/// The trapezoidal rule evaluates the integrand at `INTEGRATION_TRAPEZOIDS + 1` points.
const INTEGRATION_TRAPEZOIDS: usize = 1000;

/// Global centroid cache keyed by (dimension, bit_width).
static CENTROID_CACHE: LazyLock<DashMap<(u32, u8), Buffer<f32>>> = LazyLock::new(DashMap::default);

/// Get or compute cached centroids for the given dimension and bit width.
///
/// Returns `2^bit_width` centroids sorted in ascending order, representing optimal scalar
/// quantization levels for the coordinate distribution after a random orthogonal transform in
/// `dimension`-dimensional space.
pub(crate) fn compute_or_get_centroids(dimension: u32, bit_width: u8) -> VortexResult<Buffer<f32>> {
    vortex_ensure!(
        (1..=MAX_BIT_WIDTH).contains(&bit_width),
        "TurboQuant bit_width must be 1-{}, got {bit_width}",
        MAX_BIT_WIDTH
    );
    vortex_ensure!(
        dimension >= MIN_DIMENSION,
        "TurboQuant dimension must be >= {}, got {dimension}",
        MIN_DIMENSION
    );

    if let Some(centroids) = CENTROID_CACHE.get(&(dimension, bit_width)) {
        return Ok(centroids.clone());
    }

    let centroids = max_lloyd_centroids(dimension, bit_width);
    CENTROID_CACHE.insert((dimension, bit_width), centroids.clone());

    Ok(centroids)
}

// TODO(connor): It would potentially be more performant if this was modelled as const generic
// parameters to functions.
/// Half-integer exponent: represents `int_part + (if has_half { 0.5 } else { 0.0 })`.
///
/// The marginal distribution exponent `(d-3)/2` is always an integer (when `d` is odd) or a
/// half-integer (when `d` is even).
///
/// This type makes that invariant explicit and avoids floating-point comparison in the hot path.
#[derive(Clone, Copy, Debug)]
struct HalfIntExponent {
    int_part: i32,
    has_half: bool,
}

impl HalfIntExponent {
    /// Compute `(numerator) / 2` as a half-integer exponent.
    ///
    /// `numerator` is `d - 3` where `d` is the dimension (>= 2), so it can be negative.
    fn from_numerator(numerator: i32) -> Self {
        // Use Euclidean division to get floor division toward negative infinity.
        let int_part = numerator.div_euclid(2);
        let has_half = numerator.rem_euclid(2) != 0;
        Self { int_part, has_half }
    }
}

/// Compute optimal centroids via the Max-Lloyd (Lloyd-Max) algorithm.
///
/// Operates on the marginal distribution of a single coordinate of a randomly transformed unit
/// vector in d dimensions.
///
/// The probability distribution function is:
///   `f(x) = C_d * (1 - x^2)^((d-3)/2)` on `[-1, 1]`
/// where `C_d` is the normalizing constant.
///
/// Centroids are seeded uniformly on `[±sqrt(bit_width) * sigma]` (where `sigma` is the standard
/// deviation of the normal distribution that hypershere dimension values take, and specifically
/// `sigma = 1/sqrt(dimension)`) rather than across the full `[-1, 1]`, which strands most of the
/// centroids in the near-zero-mass tails.
///
/// Note that the `sqrt(bit_width)` is mostly empirically derived, we do not have a theoretical
/// basis for choosing this other than the fact that it seems to produce good results.
fn max_lloyd_centroids(dimension: u32, bit_width: u8) -> Buffer<f32> {
    debug_assert!((1..=MAX_BIT_WIDTH).contains(&bit_width));
    let num_centroids = 1usize << bit_width;

    // For the marginal distribution on [-1, 1], we use the exponent (d-3)/2.
    let exponent = HalfIntExponent::from_numerator(dimension as i32 - 3);

    // The coordinate marginal concentrates around 0 with this standard deviation.
    let sigma = 1.0 / f64::from(dimension).sqrt();
    let init_half = (f64::from(bit_width).sqrt() * sigma).min(1.0);

    // Initialize centroids uniformly on [-init_half, init_half], where the mass lives, so no cell
    // starts in a zero-mass region and freezes.
    let mut centroids: Vec<f64> = (0..num_centroids)
        .map(|idx| -init_half + (2.0 * (idx as f64) + 1.0) * init_half / (num_centroids as f64))
        .collect();

    let mut boundaries: Vec<f64> = vec![0.0; num_centroids + 1];
    for _ in 0..MAX_ITERATIONS {
        // Compute decision boundaries (midpoints between adjacent centroids).
        boundaries[0] = -1.0;
        for idx in 0..num_centroids - 1 {
            boundaries[idx + 1] = (centroids[idx] + centroids[idx + 1]) / 2.0;
        }
        boundaries[num_centroids] = 1.0;

        // Update each centroid to the conditional mean within its Voronoi cell.
        let mut max_change = 0.0f64;
        for idx in 0..num_centroids {
            let lo = boundaries[idx];
            let hi = boundaries[idx + 1];
            let new_centroid = mean_between_centroids(lo, hi, exponent);
            max_change = max_change.max((new_centroid - centroids[idx]).abs());
            centroids[idx] = new_centroid;
        }

        if max_change < CONVERGENCE_EPSILON {
            break;
        }
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "all values are in [-1, 1] so this just loses precision"
    )]
    centroids.into_iter().map(|val| val as f32).collect()
}

/// Compute the conditional mean of the coordinate distribution on interval [lo, hi].
///
/// Returns `E[X | lo <= X <= hi]` where X has PDF proportional to `(1 - x^2)^exponent` on [-1, 1].
///
/// Since there is no closed form for the integrals, we compute this numerically.
fn mean_between_centroids(lo: f64, hi: f64, exponent: HalfIntExponent) -> f64 {
    if (hi - lo).abs() < 1e-15 {
        return (lo + hi) / 2.0;
    }

    let dx = (hi - lo) / INTEGRATION_TRAPEZOIDS as f64;

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for step in 0..=INTEGRATION_TRAPEZOIDS {
        let x_val = lo + (step as f64) * dx;
        let weight = pdf_unnormalized(x_val, exponent);

        let trap_weight = if step == 0 || step == INTEGRATION_TRAPEZOIDS {
            0.5
        } else {
            1.0
        };

        numerator += trap_weight * x_val * weight;
        denominator += trap_weight * weight;
    }

    if denominator.abs() < 1e-30 {
        (lo + hi) / 2.0
    } else {
        numerator / denominator
    }
}

/// Unnormalized PDF of the coordinate distribution: `(1 - x^2)^exponent`.
///
/// Uses `powi` + `sqrt` instead of `powf` for the half-integer exponents that arise from `(d-3)/2`.
/// This is significantly faster than the general `powf` which goes through
/// `exp(exponent * ln(base))`.
fn pdf_unnormalized(x_val: f64, exponent: HalfIntExponent) -> f64 {
    let base = (1.0 - x_val * x_val).max(0.0);

    if exponent.has_half {
        // Half-integer exponent: base^(int_part) * sqrt(base).
        base.powi(exponent.int_part) * base.sqrt()
    } else {
        // Integer exponent: use powi directly.
        base.powi(exponent.int_part)
    }
}

/// Precompute decision boundaries (midpoints between adjacent centroids).
///
/// For `k` centroids, returns `k-1` boundaries. A value below `boundaries[0]` maps to centroid 0, a
/// value in `[boundaries[i-1], boundaries[i])` maps to centroid `i`, and a
/// value `>= boundaries[k-2]` maps to centroid `k-1`.
pub(crate) fn compute_centroid_boundaries(centroids: &[f32]) -> Vec<f32> {
    centroids.windows(2).map(|w| (w[0] + w[1]) * 0.5).collect()
}

/// Find the index of the nearest centroid using precomputed decision boundaries.
///
/// `boundaries` must be the output of [`compute_centroid_boundaries`] for the corresponding
/// centroids. Uses binary search on the midpoints, avoiding distance comparisons
/// in the inner loop.
#[inline]
pub(crate) fn find_nearest_centroid(value: f32, boundaries: &[f32]) -> u8 {
    debug_assert!(
        boundaries.windows(2).all(|w| w[0] <= w[1]),
        "boundaries must be sorted"
    );
    debug_assert!(
        boundaries.len() <= 256, // 1 << 8
        "too many boundaries"
    );

    #[expect(
        clippy::cast_possible_truncation,
        reason = "num_centroids <= 256 and partition_point will return at most 255"
    )]
    (boundaries.partition_point(|&b| b < value) as u8)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_error::VortexResult;

    use super::*;

    #[rstest]
    #[case(128, 1, 2)]
    #[case(128, 2, 4)]
    #[case(128, 3, 8)]
    #[case(128, 4, 16)]
    #[case(768, 2, 4)]
    #[case(1536, 3, 8)]
    fn centroids_have_correct_count(
        #[case] dim: u32,
        #[case] bits: u8,
        #[case] expected: usize,
    ) -> VortexResult<()> {
        let centroids = compute_or_get_centroids(dim, bits)?;
        assert_eq!(centroids.len(), expected);
        Ok(())
    }

    #[rstest]
    #[case(128, 1)]
    #[case(128, 2)]
    #[case(128, 3)]
    #[case(128, 4)]
    #[case(768, 2)]
    fn centroids_are_sorted(#[case] dim: u32, #[case] bits: u8) -> VortexResult<()> {
        let centroids = compute_or_get_centroids(dim, bits)?;
        for window in centroids.windows(2) {
            assert!(
                window[0] < window[1],
                "centroids not sorted: {:?}",
                centroids
            );
        }
        Ok(())
    }

    #[rstest]
    #[case(128, 1)]
    #[case(128, 2)]
    #[case(256, 2)]
    #[case(768, 2)]
    fn centroids_are_symmetric(#[case] dim: u32, #[case] bits: u8) -> VortexResult<()> {
        let centroids = compute_or_get_centroids(dim, bits)?;
        let count = centroids.len();
        for idx in 0..count / 2 {
            let diff = (centroids[idx] + centroids[count - 1 - idx]).abs();
            assert!(
                diff < 1e-5,
                "centroids not symmetric: c[{idx}]={}, c[{}]={}",
                centroids[idx],
                count - 1 - idx,
                centroids[count - 1 - idx]
            );
        }
        Ok(())
    }

    #[rstest]
    #[case(128, 1)]
    #[case(128, 4)]
    fn centroids_within_bounds(#[case] dim: u32, #[case] bits: u8) -> VortexResult<()> {
        let centroids = compute_or_get_centroids(dim, bits)?;
        for &val in centroids.iter() {
            assert!(
                (-1.0..=1.0).contains(&val),
                "centroid out of [-1, 1]: {val}",
            );
        }
        Ok(())
    }

    #[test]
    fn centroids_cached() -> VortexResult<()> {
        let c1 = compute_or_get_centroids(128, 2)?;
        let c2 = compute_or_get_centroids(128, 2)?;
        assert_eq!(c1, c2);
        Ok(())
    }

    #[test]
    fn find_nearest_basic() -> VortexResult<()> {
        let centroids = compute_or_get_centroids(128, 2)?;
        let boundaries = compute_centroid_boundaries(&centroids);
        assert_eq!(find_nearest_centroid(-1.0, &boundaries), 0);

        #[expect(clippy::cast_possible_truncation)]
        let last_idx = (centroids.len() - 1) as u8;
        assert_eq!(find_nearest_centroid(1.0, &boundaries), last_idx);
        for (idx, &cv) in centroids.iter().enumerate() {
            #[expect(clippy::cast_possible_truncation)]
            let expected = idx as u8;
            assert_eq!(find_nearest_centroid(cv, &boundaries), expected);
        }
        Ok(())
    }

    #[test]
    fn rejects_invalid_params() {
        assert!(compute_or_get_centroids(128, 0).is_err());
        assert!(compute_or_get_centroids(128, 9).is_err());
        assert!(compute_or_get_centroids(1, 2).is_err());
        assert!(compute_or_get_centroids(127, 2).is_err());
    }
}
