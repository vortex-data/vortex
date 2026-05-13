// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::min_ident_chars, reason = "model coefficients use short names")]

use super::FitResult;
use super::Model;
use super::ModelKind;

/// `y = a`. Picks `a` as the midpoint of `min(values)..max(values)`, which minimises the
/// L-infinity residual.
pub struct ConstantModel;

impl Model for ConstantModel {
    fn fit(values: &[f64], start: usize, end: usize, scale: f64) -> Option<FitResult> {
        let slice = &values[start..end];
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for &v in slice {
            if v.is_nan() {
                return None;
            }
            if v < lo {
                lo = v;
            }
            if v > hi {
                hi = v;
            }
        }
        let a = 0.5 * (lo + hi);
        let max_abs = 0.5 * (hi - lo);
        Some(FitResult {
            kind: ModelKind::Constant,
            a,
            b: 0.0,
            c: 0.0,
            max_abs_residual: max_abs / scale,
        })
    }
}
