// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::min_ident_chars, reason = "model coefficients use short names")]

use super::FitResult;
use super::Model;
use super::ModelKind;

/// `y = a + b*t`. Uses ordinary least squares against the relative offset `t = i - start`.
///
/// The PGM-Index hull-based incremental fit gives O(n) amortised cost during piece extension,
/// but for v1 we fit each candidate span from scratch in O(n); the synthetic and real bench
/// numbers should still be informative, and the implementation is small.
pub struct LinearModel;

impl Model for LinearModel {
    fn fit(values: &[f64], start: usize, end: usize, scale: f64) -> Option<FitResult> {
        let n = end - start;
        if n < 2 {
            // Degenerate: a single point. Fall through to constant fit upstream.
            return None;
        }
        let n_f = n as f64;
        let mut sum_t = 0.0;
        let mut sum_y = 0.0;
        let mut sum_tt = 0.0;
        let mut sum_ty = 0.0;
        for (k, &y) in values[start..end].iter().enumerate() {
            if y.is_nan() {
                return None;
            }
            let t = k as f64;
            sum_t += t;
            sum_y += y;
            sum_tt += t * t;
            sum_ty += t * y;
        }
        let denom = n_f * sum_tt - sum_t * sum_t;
        if denom == 0.0 {
            return None;
        }
        let b = (n_f * sum_ty - sum_t * sum_y) / denom;
        let a = (sum_y - b * sum_t) / n_f;

        let mut max_abs = 0.0_f64;
        for (k, &y) in values[start..end].iter().enumerate() {
            let t = k as f64;
            let r = (y - (a + b * t)).abs();
            if r > max_abs {
                max_abs = r;
            }
        }
        Some(FitResult {
            kind: ModelKind::Linear,
            a,
            b,
            c: 0.0,
            max_abs_residual: max_abs / scale,
        })
    }
}
