// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Max-Lloyd centroid computation for TurboQuant scalar quantizers.
//!
//! Pre-computes optimal scalar quantizer centroids for the marginal distribution of coordinates
//! after random rotation of a unit-norm vector. In high dimensions, each coordinate of a randomly
//! rotated unit vector follows a distribution proportional to `(1 - x^2)^((d-3)/2)` on `[-1, 1]`,
//! which converges to `N(0, 1/d)`. The Max-Lloyd algorithm finds optimal quantization centroids
//! that minimize MSE for this distribution.

use std::sync::LazyLock;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_utils::aliases::dash_map::DashMap;

/// Number of numerical integration points for computing conditional expectations.
const INTEGRATION_POINTS: usize = 1000;

/// Max-Lloyd convergence threshold.
const CONVERGENCE_EPSILON: f64 = 1e-12;

/// Maximum iterations for Max-Lloyd algorithm.
const MAX_ITERATIONS: usize = 200;

/// Global centroid cache keyed by (dimension, bit_width).
static CENTROID_CACHE: LazyLock<DashMap<(u32, u8), Vec<f32>>> = LazyLock::new(DashMap::default);

/// Get or compute cached centroids for the given dimension and bit width.
///
/// Returns `2^bit_width` centroids sorted in ascending order, representing
/// optimal scalar quantization levels for the coordinate distribution after
/// random rotation in `dimension`-dimensional space.
pub fn get_centroids(dimension: u32, bit_width: u8) -> VortexResult<Vec<f32>> {
    if !(1..=8).contains(&bit_width) {
        vortex_bail!("TurboQuant bit_width must be 1-8, got {bit_width}");
    }
    if dimension < 2 {
        vortex_bail!("TurboQuant dimension must be >= 2, got {dimension}");
    }

    if let Some(centroids) = CENTROID_CACHE.get(&(dimension, bit_width)) {
        return Ok(centroids.clone());
    }

    let centroids = max_lloyd_centroids(dimension, bit_width);
    CENTROID_CACHE.insert((dimension, bit_width), centroids.clone());
    Ok(centroids)
}

/// Compute optimal centroids via the Max-Lloyd (Lloyd-Max) algorithm.
///
/// Operates on the marginal distribution of a single coordinate of a randomly
/// rotated unit vector in d dimensions. The PDF is:
///   `f(x) = C_d * (1 - x^2)^((d-3)/2)` on `[-1, 1]`
/// where `C_d` is the normalizing constant.
fn max_lloyd_centroids(dimension: u32, bit_width: u8) -> Vec<f32> {
    let num_centroids = 1usize << bit_width;
    let dim = dimension as f64;

    // For the marginal distribution on [-1, 1], we use the exponent (d-3)/2.
    let exponent = (dim - 3.0) / 2.0;

    // Initialize centroids uniformly on [-1, 1].
    let mut centroids: Vec<f64> = (0..num_centroids)
        .map(|idx| -1.0 + (2.0 * (idx as f64) + 1.0) / (num_centroids as f64))
        .collect();

    for _ in 0..MAX_ITERATIONS {
        // Compute decision boundaries (midpoints between adjacent centroids).
        let mut boundaries = Vec::with_capacity(num_centroids + 1);
        boundaries.push(-1.0);
        for idx in 0..num_centroids - 1 {
            boundaries.push((centroids[idx] + centroids[idx + 1]) / 2.0);
        }
        boundaries.push(1.0);

        // Update each centroid to the conditional mean within its Voronoi cell.
        let mut max_change = 0.0f64;
        for idx in 0..num_centroids {
            let lo = boundaries[idx];
            let hi = boundaries[idx + 1];
            let new_centroid = conditional_mean(lo, hi, exponent);
            max_change = max_change.max((new_centroid - centroids[idx]).abs());
            centroids[idx] = new_centroid;
        }

        if max_change < CONVERGENCE_EPSILON {
            break;
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    centroids.iter().map(|&val| val as f32).collect()
}

/// Compute the conditional mean of the coordinate distribution on interval [lo, hi].
///
/// Returns `E[X | lo <= X <= hi]` where X has PDF proportional to `(1 - x^2)^exponent`
/// on [-1, 1].
fn conditional_mean(lo: f64, hi: f64, exponent: f64) -> f64 {
    if (hi - lo).abs() < 1e-15 {
        return (lo + hi) / 2.0;
    }

    let num_points = INTEGRATION_POINTS;
    let dx = (hi - lo) / num_points as f64;

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for step in 0..=num_points {
        let x_val = lo + (step as f64) * dx;
        let weight = pdf_unnormalized(x_val, exponent);

        let trap_weight = if step == 0 || step == num_points {
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
#[inline]
fn pdf_unnormalized(x_val: f64, exponent: f64) -> f64 {
    (1.0 - x_val * x_val).max(0.0).powf(exponent)
}

/// Precompute decision boundaries (midpoints between adjacent centroids).
///
/// For `k` centroids, returns `k-1` boundaries. A value below `boundaries[0]` maps
/// to centroid 0, a value in `[boundaries[i-1], boundaries[i])` maps to centroid `i`,
/// and a value >= `boundaries[k-2]` maps to centroid `k-1`.
pub fn compute_boundaries(centroids: &[f32]) -> Vec<f32> {
    centroids.windows(2).map(|w| (w[0] + w[1]) * 0.5).collect()
}

/// Find the index of the nearest centroid using precomputed decision boundaries.
///
/// `boundaries` must be the output of [`compute_boundaries`] for the corresponding
/// centroids. Uses binary search on the midpoints, avoiding distance comparisons
/// in the inner loop.
#[inline]
#[allow(clippy::cast_possible_truncation)]
pub fn find_nearest_centroid(value: f32, boundaries: &[f32]) -> u8 {
    debug_assert!(
        boundaries.windows(2).all(|w| w[0] <= w[1]),
        "boundaries must be sorted"
    );
    boundaries.partition_point(|&b| b < value) as u8
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
        let centroids = get_centroids(dim, bits)?;
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
        let centroids = get_centroids(dim, bits)?;
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
        let centroids = get_centroids(dim, bits)?;
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
        let centroids = get_centroids(dim, bits)?;
        for &val in &centroids {
            assert!(
                (-1.0..=1.0).contains(&val),
                "centroid out of [-1, 1]: {val}",
            );
        }
        Ok(())
    }

    #[test]
    fn centroids_cached() -> VortexResult<()> {
        let c1 = get_centroids(128, 2)?;
        let c2 = get_centroids(128, 2)?;
        assert_eq!(c1, c2);
        Ok(())
    }

    #[test]
    fn find_nearest_basic() -> VortexResult<()> {
        let centroids = get_centroids(128, 2)?;
        let boundaries = compute_boundaries(&centroids);
        assert_eq!(find_nearest_centroid(-1.0, &boundaries), 0);
        #[allow(clippy::cast_possible_truncation)]
        let last_idx = (centroids.len() - 1) as u8;
        assert_eq!(find_nearest_centroid(1.0, &boundaries), last_idx);
        for (idx, &cv) in centroids.iter().enumerate() {
            #[allow(clippy::cast_possible_truncation)]
            let expected = idx as u8;
            assert_eq!(find_nearest_centroid(cv, &boundaries), expected);
        }
        Ok(())
    }

    #[test]
    fn rejects_invalid_params() {
        assert!(get_centroids(128, 0).is_err());
        assert!(get_centroids(128, 9).is_err());
        assert!(get_centroids(1, 2).is_err());
    }
}
