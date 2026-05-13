// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "Numeric conversions are bounded by domain checks: residuals are range-clamped \
    before being cast to u32, index conversions are guarded by `len <= u32::MAX`, and \
    f64-to-u64 sum accumulators are intentional."
)]

//! Multi-stage homomorphic compression for scientific float columns.
//!
//! `Hsz` encodes a sequence of `f64` values as three independently decodable
//! stages:
//!
//! 1. A **predictor stage** holding per-block summaries (min, max, sum, count).
//!    All zone-map skipping and exact aggregates are answered here.
//! 2. A **residual stage** holding integer residuals from the per-block
//!    predictor, quantised to a user-supplied error bound `eps`.
//! 3. An **outlier stage** holding sparse, exact `(index, value)` pairs for
//!    elements whose residual would overflow the quantiser range.
//!
//! Operators are *homomorphic*: each one consumes the smallest set of stages
//! that suffices to answer the query. For example, [`Hsz::between_mask`]
//! returns a mask using only Stage 0 for blocks that are fully inside or
//! fully outside the range, descending into Stage 1 only for the boundary
//! blocks. [`Hsz::sum`] is answered entirely from Stage 0.
//!
//! Reconstruction is approximate within the configured `eps`:
//! `|decompress(i) - original(i)| <= eps` for every element that is not an
//! outlier, and exact for outliers.

mod compress;
mod compute;
mod decompress;
mod stage;

pub use compress::HszConfig;
pub use compute::BetweenStats;
pub use stage::BlockSummary;
pub use stage::Hsz;

#[cfg(test)]
mod tests;
