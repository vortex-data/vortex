// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::min_ident_chars, reason = "model coefficients use short names")]

use super::FitResult;
use super::Model;
use super::ModelKind;

/// `y = exp(a + b*t)`. We fit in log-space by linear-regressing `log(y)` against `t`. The
/// residual returned by `fit` is measured back in the original (non-log) value space so the
/// caller can quantize uniformly with all other families.
///
/// We refuse to fit if any value in the span is non-positive (no real log).
pub struct ExponentialModel;

impl Model for ExponentialModel {
    fn fit(values: &[f64], start: usize, end: usize, scale: f64) -> Option<FitResult> {
        let n = end - start;
        if n < 2 {
            return None;
        }
        let n_f = n as f64;

        let mut sum_t = 0.0;
        let mut sum_ly = 0.0;
        let mut sum_tt = 0.0;
        let mut sum_tly = 0.0;
        for (k, &y) in values[start..end].iter().enumerate() {
            if !y.is_finite() || y <= 0.0 {
                return None;
            }
            let t = k as f64;
            let ly = y.ln();
            sum_t += t;
            sum_ly += ly;
            sum_tt += t * t;
            sum_tly += t * ly;
        }
        let denom = n_f * sum_tt - sum_t * sum_t;
        if denom == 0.0 {
            return None;
        }
        let b = (n_f * sum_tly - sum_t * sum_ly) / denom;
        let a = (sum_ly - b * sum_t) / n_f;

        let mut max_abs = 0.0_f64;
        for (k, &y) in values[start..end].iter().enumerate() {
            let t = k as f64;
            let pred = (a + b * t).exp();
            if !pred.is_finite() {
                return None;
            }
            let r = (y - pred).abs();
            if r > max_abs {
                max_abs = r;
            }
        }
        Some(FitResult {
            kind: ModelKind::Exponential,
            a,
            b,
            c: 0.0,
            max_abs_residual: max_abs / scale,
        })
    }
}
