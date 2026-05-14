// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::min_ident_chars,
    reason = "model coefficients use mathematical short names (a, b, c, t)"
)]

//! Model families used by NeaTS pieces.
//!
//! A piece is `(start, model_id, [a, b, c])`. The model is evaluated at the relative offset
//! `t = global_index - start` so coefficients are interpretable per-piece.

mod constant;
mod exponential;
mod gaussian;
mod linear;
mod logarithmic;
mod quadratic;
mod radical;

pub use constant::ConstantModel;
pub use exponential::ExponentialModel;
pub use gaussian::GaussianModel;
pub use linear::LinearModel;
pub use logarithmic::LogarithmicModel;
pub use quadratic::QuadraticModel;
pub use radical::RadicalModel;

/// The model family identifier stored per piece. Values are stable on disk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ModelKind {
    Constant = 0,
    Linear = 1,
    Quadratic = 2,
    Exponential = 3,
    Radical = 4,
    Logarithmic = 5,
    Gaussian = 6,
}

impl ModelKind {
    /// Decode a model kind from its on-disk byte.
    pub fn from_u8(byte: u8) -> Option<Self> {
        match byte {
            0 => Some(Self::Constant),
            1 => Some(Self::Linear),
            2 => Some(Self::Quadratic),
            3 => Some(Self::Exponential),
            4 => Some(Self::Radical),
            5 => Some(Self::Logarithmic),
            6 => Some(Self::Gaussian),
            _ => None,
        }
    }
}

/// Evaluate a piece's model at offset `t`. Used by both decompress and random access.
#[inline]
pub fn eval(kind: ModelKind, a: f64, b: f64, c: f64, t: f64) -> f64 {
    match kind {
        ModelKind::Constant => a,
        ModelKind::Linear => a + b * t,
        ModelKind::Quadratic => a + b * t + c * t * t,
        // Exponential is stored as (log(a), b, _) so we don't accumulate large numbers; the
        // canonical form is `exp(log(a) + b*t)`. We use `a` to hold `log(a_real)`.
        ModelKind::Exponential => (a + b * t).exp(),
        ModelKind::Radical => a + b * t.sqrt(),
        ModelKind::Logarithmic => a + b * (t + 1.0).ln(),
        // Gaussian: a is amplitude, b is centre μ, c is width σ.
        ModelKind::Gaussian => {
            let dt = t - b;
            a * (-0.5 * (dt * dt) / (c * c)).exp()
        }
    }
}

/// Try to fit any of the enabled families to `values[start..end]` and return the best one (the
/// family that minimises the bit-width of the residuals after quantization at `scale`).
///
/// `values` are the full array of f64 source values. `start..end` is the half-open piece.
///
/// Returns `(kind, a, b, c, max_abs_residual)`. The caller chooses the piece length by extending
/// while `max_abs_residual` stays bounded; a return of `None` means no family fits.
pub fn fit_best(values: &[f64], start: usize, end: usize, scale: f64) -> Option<FitResult> {
    debug_assert!(start < end);

    let mut best: Option<FitResult> = None;
    let candidates = [
        ConstantModel::fit(values, start, end, scale),
        LinearModel::fit(values, start, end, scale),
        QuadraticModel::fit(values, start, end, scale),
        ExponentialModel::fit(values, start, end, scale),
        RadicalModel::fit(values, start, end, scale),
        LogarithmicModel::fit(values, start, end, scale),
        GaussianModel::fit(values, start, end, scale),
    ];
    for cand in candidates.into_iter().flatten() {
        match best {
            None => best = Some(cand),
            Some(b) if cand.max_abs_residual < b.max_abs_residual => best = Some(cand),
            _ => {}
        }
    }
    best
}

/// The output of fitting a single model family to a contiguous span of values.
#[derive(Clone, Copy, Debug)]
pub struct FitResult {
    pub kind: ModelKind,
    pub a: f64,
    pub b: f64,
    pub c: f64,
    /// The maximum absolute residual `|x - model(t)|` over the span, expressed in units of
    /// `scale` (i.e., divided by `scale`). The caller uses this to decide if the piece's
    /// residuals fit in a target bit width.
    pub max_abs_residual: f64,
}

/// Trait every NeaTS model family implements.
pub trait Model {
    /// Fit this family to `values[start..end]`. Return `None` if the family cannot fit (for
    /// example exponential against non-positive values).
    fn fit(values: &[f64], start: usize, end: usize, scale: f64) -> Option<FitResult>;
}
