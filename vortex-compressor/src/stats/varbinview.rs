// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Variable-length byte/string compression statistics.

use vortex_array::ExecutionCtx;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::varbinview::BinaryView;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::AllOr;

use super::GenerateStatsOptions;
use super::cardinality::CardinalityEstimator;
use super::cardinality::estimate_could_be_at_most;

/// Array of variable-length byte/string values, and relevant stats for compression.
#[derive(Clone, Debug)]
pub struct StringStats {
    /// The estimated number of distinct values, or `None` if not computed.
    /// This _must_ be non-zero.
    estimated_distinct_count: Option<usize>,
    /// The number of non-null values.
    value_count: usize,
    /// The number of null values.
    null_count: usize,
}

/// Estimate the number of distinct strings in the var bin view array using Cloudflare's
/// cardinality estimator.
///
/// Every non-null value is hashed in full, so the estimate's only error is the estimator's own:
/// it is exact for small cardinalities and transitions to HyperLogLog++ for larger ones. Null
/// entries are skipped, matching the integer and float distinct-count stats.
fn estimate_distinct_count(
    strings: &VarBinViewArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<usize> {
    let views = strings.views();
    let validity = strings
        .as_ref()
        .validity()?
        .execute_mask(strings.len(), ctx)?;
    let mut estimator: CardinalityEstimator<[u8]> = CardinalityEstimator::new();
    let buffers = strings
        .data_buffers()
        .iter()
        .map(|b| b.as_host())
        .collect::<Vec<_>>();

    match validity.bit_buffer() {
        AllOr::All => {
            for view in views {
                estimator.insert(view_bytes(&buffers, view));
            }
        }
        // Every value is null, so there is nothing to count.
        AllOr::None => {}
        AllOr::Some(is_valid) => {
            for (idx, view) in views.iter().enumerate() {
                if is_valid.value(idx) {
                    estimator.insert(view_bytes(&buffers, view));
                }
            }
        }
    }

    Ok(estimator.estimate())
}

/// Returns the full bytes backing a single view, reading from the array's data buffers when the
/// value is not inlined.
///
/// Only call this for non-null positions: a null slot's view may hold arbitrary bytes whose buffer
/// index and offset are not safe to dereference.
fn view_bytes<'a>(buffers: &[&'a ByteBuffer], view: &'a BinaryView) -> &'a [u8] {
    if view.is_inlined() {
        view.as_inlined().value()
    } else {
        let r = view.as_view();
        &buffers[r.buffer_index as usize][r.as_range()]
    }
}

impl StringStats {
    /// Generates stats, returning an error on failure.
    fn generate_opts_fallible(
        input: &VarBinViewArray,
        opts: GenerateStatsOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Self> {
        let null_count = input
            .statistics()
            .compute_null_count(ctx)
            .ok_or_else(|| vortex_err!("Failed to compute null_count"))?;
        let value_count = input.len() - null_count;
        let estimated_distinct_count = opts
            .count_distinct_values
            .then(|| estimate_distinct_count(input, ctx))
            .transpose()?;

        Ok(Self {
            estimated_distinct_count,
            value_count,
            null_count,
        })
    }
}

impl StringStats {
    /// Generates stats with default options.
    pub fn generate(input: &VarBinViewArray, ctx: &mut ExecutionCtx) -> Self {
        Self::generate_opts(input, GenerateStatsOptions::default(), ctx)
    }

    /// Generates stats with provided options.
    pub fn generate_opts(
        input: &VarBinViewArray,
        opts: GenerateStatsOptions,
        ctx: &mut ExecutionCtx,
    ) -> Self {
        Self::generate_opts_fallible(input, opts, ctx)
            .vortex_expect("StringStats::generate_opts should not fail")
    }

    /// Returns the estimated number of distinct values, or `None` if not computed.
    ///
    /// The estimate is exact for small cardinalities and an approximation (which may be slightly
    /// above or below the true count) for larger ones.
    pub fn estimated_distinct_count(&self) -> Option<usize> {
        self.estimated_distinct_count
    }

    /// Returns true if the true distinct count could plausibly be at most `count`.
    pub fn estimated_distinct_count_could_be_at_most(&self, count: usize) -> bool {
        let Some(distinct_count) = self.estimated_distinct_count else {
            return true;
        };

        estimate_could_be_at_most(distinct_count, count)
    }

    /// Returns the number of non-null values.
    pub fn value_count(&self) -> usize {
        self.value_count
    }

    /// Returns the number of null values.
    pub fn null_count(&self) -> usize {
        self.null_count
    }
}
