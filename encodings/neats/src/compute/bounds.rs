// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::min_ident_chars,
    reason = "model coefficients use mathematical short names (a, b, c, t)"
)]

//! Per-piece bounds for NeaTS.
//!
//! For a piece with `model_p` and the residuals `r[piece_start..piece_end]`, the decoded values
//! lie in
//!
//! ```text
//!   [min_t model_p(t) + min_r r * scale, max_t model_p(t) + max_r r * scale]
//! ```
//!
//! These bounds are exact in lossy mode (modulo `scale/2 = epsilon`, which the caller already
//! accepted) and exact in lossless mode (residuals are the exact integer correction).
//!
//! Bounds enable:
//!
//! - **min/max aggregate**: reduce over `P` bounds instead of `N` values; O(P + N).
//! - **predicate pushdown**: a piece whose bounds prove `predicate(x) = false` (or always true)
//!   for all `x` in the bound can be skipped or replaced with a constant mask.

use crate::models::ModelKind;

/// Bounds on `model_p(t)` for `t in [0, piece_len)`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ModelBounds {
    pub min: f64,
    pub max: f64,
}

/// Compute exact bounds on the piece's model output without iterating residuals.
pub(crate) fn model_bounds(
    kind: ModelKind,
    a: f64,
    b: f64,
    c: f64,
    piece_len: usize,
) -> ModelBounds {
    if piece_len == 0 {
        return ModelBounds { min: a, max: a };
    }
    let last = (piece_len - 1) as f64;
    match kind {
        ModelKind::Constant => ModelBounds { min: a, max: a },
        ModelKind::Linear => {
            let v0 = a;
            let v1 = a + b * last;
            ModelBounds {
                min: v0.min(v1),
                max: v0.max(v1),
            }
        }
        ModelKind::Quadratic => {
            // y = a + b*t + c*t^2. Vertex at t* = -b/(2c) (if c != 0).
            let v0 = a;
            let v1 = a + b * last + c * last * last;
            let mut lo = v0.min(v1);
            let mut hi = v0.max(v1);
            if c != 0.0 {
                let t_star = -b / (2.0 * c);
                if t_star > 0.0 && t_star < last {
                    let v_star = a + b * t_star + c * t_star * t_star;
                    lo = lo.min(v_star);
                    hi = hi.max(v_star);
                }
            }
            ModelBounds { min: lo, max: hi }
        }
        ModelKind::Exponential => {
            // y = exp(a + b*t) is monotonic in t (sign of b decides direction).
            let v0 = a.exp();
            let v1 = (a + b * last).exp();
            ModelBounds {
                min: v0.min(v1),
                max: v0.max(v1),
            }
        }
        ModelKind::Radical => {
            // y = a + b * sqrt(t) is monotonic in t (sign of b decides direction).
            let v0 = a;
            let v1 = a + b * last.sqrt();
            ModelBounds {
                min: v0.min(v1),
                max: v0.max(v1),
            }
        }
        ModelKind::Logarithmic => {
            // y = a + b * ln(t+1) is monotonic in t.
            let v0 = a;
            let v1 = a + b * (last + 1.0).ln();
            ModelBounds {
                min: v0.min(v1),
                max: v0.max(v1),
            }
        }
        ModelKind::Gaussian => {
            // y = a * exp(-(t-μ)²/(2σ²)). The peak is at t = μ with value a; values fall off
            // monotonically with |t-μ|. We bound by checking endpoints and (if inside the
            // piece) the centre μ.
            let v_at = |t: f64| -> f64 {
                let dt = t - b;
                a * (-0.5 * (dt * dt) / (c * c)).exp()
            };
            let v0 = v_at(0.0);
            let v_last = v_at(last);
            let mut lo = v0.min(v_last);
            let mut hi = v0.max(v_last);
            if b > 0.0 && b < last {
                let v_peak = a; // exp(0) = 1
                lo = lo.min(v_peak);
                hi = hi.max(v_peak);
            }
            // Far from the centre the value goes to 0; the data span only covers a finite
            // distance so the above three points give a valid bound (Gaussian is unimodal).
            ModelBounds { min: lo, max: hi }
        }
    }
}

/// Combined value bounds for a piece: model bounds plus residual extrema scaled.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ValueBounds {
    pub min: f64,
    pub max: f64,
}

/// Combine model bounds with the piece's residual extrema (`min_r`/`max_r` from the residual
/// slot) to produce inclusive bounds on the decoded values of the piece.
pub(crate) fn value_bounds(model: ModelBounds, min_r: i64, max_r: i64, scale: f64) -> ValueBounds {
    let r_min_contribution = (min_r as f64) * scale;
    let r_max_contribution = (max_r as f64) * scale;
    // `scale` is positive, so min_r * scale <= max_r * scale.
    ValueBounds {
        min: model.min + r_min_contribution,
        max: model.max + r_max_contribution,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_bounds() {
        let b = model_bounds(ModelKind::Constant, 3.5, 0.0, 0.0, 100);
        assert_eq!(b.min, 3.5);
        assert_eq!(b.max, 3.5);
    }

    #[test]
    fn linear_bounds() {
        let b = model_bounds(ModelKind::Linear, 1.0, 0.5, 0.0, 10);
        // values 1.0, 1.5, ..., 1.0 + 0.5*9 = 5.5
        assert!((b.min - 1.0).abs() < 1e-12);
        assert!((b.max - 5.5).abs() < 1e-12);
    }

    #[test]
    fn quadratic_bounds_vertex_inside() {
        // y = 0 + 4t + (-1)t^2. Vertex at t=2, value=4. Endpoints: t=0→0, t=4→0.
        let b = model_bounds(ModelKind::Quadratic, 0.0, 4.0, -1.0, 5);
        assert!((b.min - 0.0).abs() < 1e-12);
        assert!((b.max - 4.0).abs() < 1e-12);
    }

    #[test]
    fn quadratic_bounds_vertex_outside() {
        // y = 0 + 4t + 1*t^2 over t in [0, 5). Vertex at t=-2 (outside). Monotonic increasing.
        let b = model_bounds(ModelKind::Quadratic, 0.0, 4.0, 1.0, 5);
        assert!((b.min - 0.0).abs() < 1e-12);
        // t=4: 0 + 16 + 16 = 32
        assert!((b.max - 32.0).abs() < 1e-12);
    }

    #[test]
    fn exponential_bounds() {
        // y = exp(0 + 1*t) over t in [0, 3). Monotonic increasing.
        let b = model_bounds(ModelKind::Exponential, 0.0, 1.0, 0.0, 3);
        assert!((b.min - 1.0).abs() < 1e-12);
        assert!((b.max - 2.0_f64.exp() * std::f64::consts::E.powi(0)).abs() < 1e-3);
    }

    #[test]
    fn value_bounds_combine() {
        let model = ModelBounds { min: 2.0, max: 5.0 };
        let vb = value_bounds(model, -3, 7, 0.1);
        assert!((vb.min - 1.7).abs() < 1e-12);
        assert!((vb.max - 5.7).abs() < 1e-12);
    }
}
