// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Largest-Triangle-Three-Buckets (LTTB) downsampling for chart series.
//!
//! Reduces a `(timestamp, value)` series to a target point count while
//! preserving visible peaks and troughs. Applied per-series so different
//! series in the same chart can have independent gap structures.
//!
//! See <https://skemman.is/handle/1946/15343> for the original paper.

/// Default target point count for `?max_points` when the caller omits it.
pub const DEFAULT_MAX_POINTS: u32 = 600;

/// Hard upper bound on `?max_points`. Anything larger is clamped.
pub const MAX_POINTS_LIMIT: u32 = 5000;

/// One sample of a chart series. `value` may be `None` to denote a gap.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sample {
    /// X-coordinate. The caller picks the units (seconds since epoch is fine).
    pub x: f64,
    /// Y-coordinate, or `None` for a missing point.
    pub y: Option<f64>,
}

/// Run LTTB on `series`, returning at most `max_points` samples.
///
/// If the input has `max_points` or fewer non-gap points, the input is
/// returned unchanged so the chart shows raw data when it's short enough
/// to fit. Gaps (`None` y-values) are preserved by always emitting them
/// to the output if they fall in the kept buckets, and by skipping them
/// when computing bucket centroids.
pub fn lttb(series: &[Sample], max_points: u32) -> Vec<Sample> {
    let n = series.len();
    let target = max_points as usize;
    if target < 3 || n <= target {
        return series.to_vec();
    }

    let first_idx = 0usize;
    let last_idx = n - 1;
    let mut out: Vec<Sample> = Vec::with_capacity(target);
    out.push(series[first_idx]);

    // We pick `target - 2` middle points and always keep the endpoints.
    let middle = target - 2;
    let bucket_size = (n - 2) as f64 / middle as f64;

    let mut a = first_idx;
    for i in 0..middle {
        let next_start = 1 + ((i + 1) as f64 * bucket_size).floor() as usize;
        let next_end = (1 + ((i + 2) as f64 * bucket_size).floor() as usize).min(n);
        let bucket_start = 1 + (i as f64 * bucket_size).floor() as usize;
        let bucket_end = next_start.min(n);

        let (avg_x, avg_y) = bucket_average(&series[next_start..next_end]);

        let prev = series[a];
        let mut best: Option<(usize, f64)> = None;
        for (j, s) in series[bucket_start..bucket_end].iter().enumerate() {
            let area = triangle_area(prev, *s, avg_x, avg_y);
            let real_idx = bucket_start + j;
            match best {
                None => best = Some((real_idx, area)),
                Some((_, prev_area)) if area > prev_area => best = Some((real_idx, area)),
                _ => {}
            }
        }
        if let Some((idx, _)) = best {
            out.push(series[idx]);
            a = idx;
        }
    }
    out.push(series[last_idx]);
    out
}

/// Average of (x, y) ignoring `None` y-values. Falls back to the bucket's
/// midpoint x and a `None` y if the whole bucket is gaps.
fn bucket_average(bucket: &[Sample]) -> (f64, Option<f64>) {
    if bucket.is_empty() {
        return (0.0, None);
    }
    let mut sum_x = 0.0f64;
    let mut sum_y = 0.0f64;
    let mut count = 0usize;
    for s in bucket {
        if let Some(y) = s.y {
            sum_x += s.x;
            sum_y += y;
            count += 1;
        }
    }
    if count == 0 {
        let mid = bucket[bucket.len() / 2];
        return (mid.x, None);
    }
    let n = count as f64;
    (sum_x / n, Some(sum_y / n))
}

/// Triangle area between `prev`, `point`, and the next-bucket centroid.
/// Treats gaps as zero-area, which makes LTTB prefer real points whenever a
/// non-gap option exists.
fn triangle_area(prev: Sample, point: Sample, avg_x: f64, avg_y: Option<f64>) -> f64 {
    let (Some(py), Some(y), Some(ay)) = (prev.y, point.y, avg_y) else {
        return 0.0;
    };
    ((prev.x - avg_x) * (y - py) - (prev.x - point.x) * (ay - py)).abs() * 0.5
}

/// Resolve a user-supplied `max_points` into the value to actually use.
/// `None` means "use [`DEFAULT_MAX_POINTS`]"; a `0` value means "do not
/// downsample" and is signalled by returning `None`. Anything above
/// [`MAX_POINTS_LIMIT`] is clamped to it.
pub fn resolve_max_points(requested: Option<u32>) -> Option<u32> {
    match requested {
        Some(0) => None,
        Some(n) => Some(n.min(MAX_POINTS_LIMIT)),
        None => Some(DEFAULT_MAX_POINTS),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(n: usize) -> Vec<Sample> {
        (0..n)
            .map(|i| Sample {
                x: i as f64,
                y: Some(i as f64),
            })
            .collect()
    }

    #[test]
    fn passthrough_when_input_smaller_than_target() {
        let s = ramp(50);
        let out = lttb(&s, 100);
        assert_eq!(out, s);
    }

    #[test]
    fn passthrough_when_max_points_is_tiny() {
        let s = ramp(50);
        // Anything below 3 is degenerate; we just return the input.
        assert_eq!(lttb(&s, 0), s);
        assert_eq!(lttb(&s, 1), s);
        assert_eq!(lttb(&s, 2), s);
    }

    #[test]
    fn endpoints_are_preserved() {
        let s = ramp(1000);
        let out = lttb(&s, 100);
        assert_eq!(out.len(), 100);
        assert_eq!(out.first().copied(), s.first().copied());
        assert_eq!(out.last().copied(), s.last().copied());
    }

    #[test]
    fn output_count_matches_target() {
        let s: Vec<Sample> = (0..10_000)
            .map(|i| Sample {
                x: i as f64,
                y: Some(((i as f64) / 50.0).sin()),
            })
            .collect();
        let out = lttb(&s, 600);
        assert_eq!(out.len(), 600);
    }

    #[test]
    fn output_is_monotonic_in_x() {
        let s: Vec<Sample> = (0..2000)
            .map(|i| Sample {
                x: i as f64 * 0.5,
                y: Some((i as f64).sqrt()),
            })
            .collect();
        let out = lttb(&s, 200);
        for w in out.windows(2) {
            assert!(w[0].x <= w[1].x, "lttb must preserve x order: {w:?}");
        }
    }

    #[test]
    fn gaps_pass_through_without_panicking() {
        let mut s = ramp(2000);
        for i in (100..200).chain(500..505) {
            s[i].y = None;
        }
        let out = lttb(&s, 300);
        assert_eq!(out.len(), 300);
    }

    #[test]
    fn preserves_a_clear_spike() {
        let mut s = ramp(5000);
        s[2500].y = Some(1.0e9);
        let out = lttb(&s, 300);
        let max_y = out.iter().filter_map(|s| s.y).fold(0.0_f64, f64::max);
        assert!(
            max_y >= 1.0e9,
            "LTTB should retain a 1e9 spike, got max {max_y}"
        );
    }

    #[test]
    fn resolve_max_points_default_and_cap() {
        assert_eq!(resolve_max_points(None), Some(DEFAULT_MAX_POINTS));
        assert_eq!(resolve_max_points(Some(0)), None);
        assert_eq!(resolve_max_points(Some(100)), Some(100));
        assert_eq!(resolve_max_points(Some(999_999)), Some(MAX_POINTS_LIMIT));
    }
}
