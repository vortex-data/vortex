// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::min_ident_chars, reason = "model coefficients use short names")]

use super::FitResult;
use super::Model;
use super::ModelKind;

/// `y = a + b*t + c*t*t`. Closed-form least-squares solution over `t = i - start`.
pub struct QuadraticModel;

impl Model for QuadraticModel {
    fn fit(values: &[f64], start: usize, end: usize, scale: f64) -> Option<FitResult> {
        let n = end - start;
        if n < 3 {
            return None;
        }
        let n_f = n as f64;

        // Power sums of t.
        let mut s0 = 0.0; // sum t^0 = n
        let mut s1 = 0.0;
        let mut s2 = 0.0;
        let mut s3 = 0.0;
        let mut s4 = 0.0;
        // Cross sums.
        let mut sy = 0.0;
        let mut sty = 0.0;
        let mut stty = 0.0;
        for (k, &y) in values[start..end].iter().enumerate() {
            if y.is_nan() {
                return None;
            }
            let t = k as f64;
            let tt = t * t;
            s0 += 1.0;
            s1 += t;
            s2 += tt;
            s3 += tt * t;
            s4 += tt * tt;
            sy += y;
            sty += t * y;
            stty += tt * y;
        }
        debug_assert_eq!(s0, n_f);

        // Solve the 3x3 normal-equation system
        // [s0 s1 s2] [a]   [sy]
        // [s1 s2 s3] [b] = [sty]
        // [s2 s3 s4] [c]   [stty]
        // via cofactor expansion.
        let det = s0 * (s2 * s4 - s3 * s3) - s1 * (s1 * s4 - s2 * s3) + s2 * (s1 * s3 - s2 * s2);
        if det.abs() < 1e-30 {
            return None;
        }
        let a = (sy * (s2 * s4 - s3 * s3) - s1 * (sty * s4 - stty * s3)
            + s2 * (sty * s3 - stty * s2))
            / det;
        let b = (s0 * (sty * s4 - stty * s3) - sy * (s1 * s4 - s2 * s3)
            + s2 * (s1 * stty - s2 * sty))
            / det;
        let c = (s0 * (s2 * stty - s3 * sty) - s1 * (s1 * stty - s2 * sty)
            + sy * (s1 * s3 - s2 * s2))
            / det;

        let mut max_abs = 0.0_f64;
        for (k, &y) in values[start..end].iter().enumerate() {
            let t = k as f64;
            let r = (y - (a + b * t + c * t * t)).abs();
            if r > max_abs {
                max_abs = r;
            }
        }
        Some(FitResult {
            kind: ModelKind::Quadratic,
            a,
            b,
            c,
            max_abs_residual: max_abs / scale,
        })
    }
}
