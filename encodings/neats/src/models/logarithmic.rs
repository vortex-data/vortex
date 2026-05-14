// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::min_ident_chars, reason = "model coefficients use short names")]

use super::FitResult;
use super::Model;
use super::ModelKind;

/// `y = a + b * ln(t + 1)`. Linear in parameters with `u = ln(t + 1)` so we use OLS over (u, y).
///
/// The `+1` ensures `ln` is finite at `t = 0`. This family fits log-shaped growth (e.g.
/// information-content vs sample-size, or many decaying-rate processes).
pub struct LogarithmicModel;

impl Model for LogarithmicModel {
    fn fit(values: &[f64], start: usize, end: usize, scale: f64) -> Option<FitResult> {
        let n = end - start;
        if n < 2 {
            return None;
        }
        let n_f = n as f64;
        let mut sum_u = 0.0;
        let mut sum_y = 0.0;
        let mut sum_uu = 0.0;
        let mut sum_uy = 0.0;
        for (k, &y) in values[start..end].iter().enumerate() {
            if y.is_nan() {
                return None;
            }
            let u = ((k as f64) + 1.0).ln();
            sum_u += u;
            sum_y += y;
            sum_uu += u * u;
            sum_uy += u * y;
        }
        let denom = n_f * sum_uu - sum_u * sum_u;
        if denom == 0.0 {
            return None;
        }
        let b = (n_f * sum_uy - sum_u * sum_y) / denom;
        let a = (sum_y - b * sum_u) / n_f;

        let mut max_abs = 0.0_f64;
        for (k, &y) in values[start..end].iter().enumerate() {
            let u = ((k as f64) + 1.0).ln();
            let r = (y - (a + b * u)).abs();
            if r > max_abs {
                max_abs = r;
            }
        }
        Some(FitResult {
            kind: ModelKind::Logarithmic,
            a,
            b,
            c: 0.0,
            max_abs_residual: max_abs / scale,
        })
    }
}
