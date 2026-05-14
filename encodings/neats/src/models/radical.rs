// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::min_ident_chars, reason = "model coefficients use short names")]

use super::FitResult;
use super::Model;
use super::ModelKind;

/// `y = a + b * sqrt(t)`. Linear in parameters with `u = sqrt(t)` so we use OLS over (u, y).
///
/// This family fits cumulative-distribution-like and decaying-response signals well — sqrt is the
/// natural growth law for diffusion and many physical processes.
pub struct RadicalModel;

impl Model for RadicalModel {
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
            let u = (k as f64).sqrt();
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
            let u = (k as f64).sqrt();
            let r = (y - (a + b * u)).abs();
            if r > max_abs {
                max_abs = r;
            }
        }
        Some(FitResult {
            kind: ModelKind::Radical,
            a,
            b,
            c: 0.0,
            max_abs_residual: max_abs / scale,
        })
    }
}
