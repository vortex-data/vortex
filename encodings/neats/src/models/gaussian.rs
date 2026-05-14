// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::min_ident_chars, reason = "model coefficients use short names")]

use super::FitResult;
use super::Model;
use super::ModelKind;

/// `y = a * exp(-(t - μ)² / (2σ²))`. Three parameters packed into the slot triple:
///
/// - `a` ↦ peak amplitude
/// - `b` ↦ centre μ
/// - `c` ↦ width σ (positive)
///
/// Gaussian fits are non-convex; we don't run a full Levenberg–Marquardt. Instead:
///
/// 1. Method-of-moments seed: treat `|y|` as a probability density, compute the weighted mean
///    (μ) and weighted second moment about the mean (σ²), and use the absolute maximum of `y`
///    as `a`.
/// 2. Two Gauss–Newton refinement steps on the linearised log-residual.
///
/// The refinement is cheap and converges quickly when the seed is reasonable (which it is for
/// peak-shaped signals like ECG-style spikes). For data that doesn't look Gaussian the fit
/// returns large residuals, the partitioner will pick a different family, and the seed-and-
/// refine cost is bounded.
pub struct GaussianModel;

impl Model for GaussianModel {
    fn fit(values: &[f64], start: usize, end: usize, scale: f64) -> Option<FitResult> {
        let n = end - start;
        if n < 3 {
            return None;
        }
        let slice = &values[start..end];

        // All values must be the same sign for the Gaussian envelope to make sense; pick the
        // dominant sign and work with absolute values.
        let mut max_abs = 0.0_f64;
        let mut max_idx = 0usize;
        for (k, &y) in slice.iter().enumerate() {
            if !y.is_finite() {
                return None;
            }
            let ay = y.abs();
            if ay > max_abs {
                max_abs = ay;
                max_idx = k;
            }
        }
        if max_abs == 0.0 {
            return None;
        }
        let sign = if slice[max_idx] >= 0.0 { 1.0 } else { -1.0 };

        // Method-of-moments seed using |y_k| as weights.
        let mut sum_w = 0.0;
        let mut sum_tw = 0.0;
        for (k, &y) in slice.iter().enumerate() {
            let w = y.abs();
            sum_w += w;
            sum_tw += (k as f64) * w;
        }
        if sum_w == 0.0 {
            return None;
        }
        let mut mu = sum_tw / sum_w;
        let mut sum_var = 0.0;
        for (k, &y) in slice.iter().enumerate() {
            let w = y.abs();
            let d = (k as f64) - mu;
            sum_var += w * d * d;
        }
        let mut sigma = (sum_var / sum_w).sqrt();
        if !sigma.is_finite() || sigma < 1e-9 {
            sigma = 1.0;
        }
        let mut a = sign * max_abs;

        // Two Gauss–Newton refinement steps. Each step solves a 3x3 normal equations system
        // for δ = (δa, δμ, δσ) minimising ‖f(θ) - y‖² locally.
        for _ in 0..2 {
            // f(t) = a * exp(-(t-μ)²/(2σ²))
            // ∂f/∂a = exp(...) = f/a (when a != 0)
            // ∂f/∂μ = f * (t-μ) / σ²
            // ∂f/∂σ = f * (t-μ)² / σ³
            let s2 = sigma * sigma;
            let s3 = s2 * sigma;
            let mut jtj = [[0.0_f64; 3]; 3];
            let mut jtr = [0.0_f64; 3];
            let mut all_zero = true;
            for (k, &y) in slice.iter().enumerate() {
                let t = k as f64;
                let dt = t - mu;
                let exp_term = (-0.5 * (dt * dt) / s2).exp();
                let pred = a * exp_term;
                if pred != 0.0 {
                    all_zero = false;
                }
                let dfda = exp_term;
                let dfdmu = pred * dt / s2;
                let dfds = pred * (dt * dt) / s3;
                let r = y - pred;
                jtj[0][0] += dfda * dfda;
                jtj[0][1] += dfda * dfdmu;
                jtj[0][2] += dfda * dfds;
                jtj[1][1] += dfdmu * dfdmu;
                jtj[1][2] += dfdmu * dfds;
                jtj[2][2] += dfds * dfds;
                jtr[0] += dfda * r;
                jtr[1] += dfdmu * r;
                jtr[2] += dfds * r;
            }
            if all_zero {
                return None;
            }
            jtj[1][0] = jtj[0][1];
            jtj[2][0] = jtj[0][2];
            jtj[2][1] = jtj[1][2];
            // Tikhonov regularization: add a tiny ridge so the solve is numerically stable.
            for i in 0..3 {
                jtj[i][i] += 1e-12;
            }
            let Some(delta) = solve_3x3(jtj, jtr) else {
                break;
            };
            a += delta[0];
            mu += delta[1];
            sigma += delta[2];
            if sigma < 1e-9 {
                sigma = 1e-9;
            }
        }

        // Compute max residual at the final (a, μ, σ).
        let s2 = sigma * sigma;
        let mut max_abs_residual = 0.0_f64;
        for (k, &y) in slice.iter().enumerate() {
            let dt = (k as f64) - mu;
            let pred = a * (-0.5 * (dt * dt) / s2).exp();
            let r = (y - pred).abs();
            if r > max_abs_residual {
                max_abs_residual = r;
            }
        }
        if !max_abs_residual.is_finite() {
            return None;
        }
        Some(FitResult {
            kind: ModelKind::Gaussian,
            a,
            b: mu,
            c: sigma,
            max_abs_residual: max_abs_residual / scale,
        })
    }
}

/// Solve `M · x = b` for `x` where `M` is a 3×3 matrix using Cramer's rule. Returns `None` when
/// the determinant is too small to invert reliably.
fn solve_3x3(m: [[f64; 3]; 3], b: [f64; 3]) -> Option<[f64; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() < 1e-30 {
        return None;
    }
    let inv_det = 1.0 / det;
    let mut x = [0.0; 3];
    for col in 0..3 {
        let mut mc = m;
        for row in 0..3 {
            mc[row][col] = b[row];
        }
        let dc = mc[0][0] * (mc[1][1] * mc[2][2] - mc[1][2] * mc[2][1])
            - mc[0][1] * (mc[1][0] * mc[2][2] - mc[1][2] * mc[2][0])
            + mc[0][2] * (mc[1][0] * mc[2][1] - mc[1][1] * mc[2][0]);
        x[col] = dc * inv_det;
    }
    Some(x)
}
