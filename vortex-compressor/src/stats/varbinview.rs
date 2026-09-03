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
use vortex_utils::aliases::hash_set::HashSet;

use super::GenerateStatsOptions;

/// Array of variable-length byte/string values, and relevant stats for compression.
#[derive(Clone, Debug)]
pub struct StringStats {
    /// The number of distinct values, or `None` if not computed.
    /// This _must_ be non-zero.
    distinct_count: Option<u32>,
    /// The visible bytes across the distinct values.
    distinct_value_bytes: Option<u64>,
    /// The visible bytes across all values.
    value_bytes: u64,
    /// The number of non-null values.
    value_count: u32,
    /// The number of null values.
    null_count: u32,
}

/// Returns the bytes referenced by a variable-width view.
fn view_bytes<'a>(buffers: &[&'a ByteBuffer], view: &'a BinaryView) -> &'a [u8] {
    if view.is_inlined() {
        view.as_inlined().value()
    } else {
        let reference = view.as_view();
        &buffers[reference.buffer_index as usize][reference.as_range()]
    }
}

/// Counts distinct values and their visible bytes.
fn count_distinct_values(
    varbinview: &VarBinViewArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(u32, u64)> {
    let views = varbinview.views();
    let buffers = varbinview
        .data_buffers()
        .iter()
        .map(|buffer| buffer.as_host())
        .collect::<Vec<_>>();
    let validity = varbinview
        .as_ref()
        .validity()?
        .execute_mask(varbinview.len(), ctx)?;
    let mut distinct = HashSet::with_capacity(views.len() / 2);
    let mut distinct_value_bytes = 0u64;

    for (index, view) in views.iter().enumerate() {
        if validity.value(index) {
            let bytes = view_bytes(&buffers, view);
            if distinct.insert(bytes) {
                distinct_value_bytes += bytes.len() as u64;
            }
        }
    }

    Ok((u32::try_from(distinct.len())?, distinct_value_bytes))
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
        let distinct_values = opts
            .count_distinct_values
            .then(|| count_distinct_values(input, ctx))
            .transpose()?;
        let (distinct_count, distinct_value_bytes) = distinct_values
            .map(|(count, bytes)| (Some(count), Some(bytes)))
            .unwrap_or((None, None));
        let value_bytes = input.views().iter().map(|view| u64::from(view.len())).sum();

        Ok(Self {
            value_count: u32::try_from(value_count)?,
            null_count: u32::try_from(null_count)?,
            distinct_count,
            distinct_value_bytes,
            value_bytes,
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

    /// Returns the number of distinct values, or `None` if not computed.
    pub fn distinct_count(&self) -> Option<u32> {
        self.distinct_count
    }

    /// Returns the visible bytes across the distinct values.
    pub fn distinct_value_bytes(&self) -> Option<u64> {
        self.distinct_value_bytes
    }

    /// Returns the visible bytes across all values.
    pub fn value_bytes(&self) -> u64 {
        self.value_bytes
    }

    /// Returns the number of non-null values.
    pub fn value_count(&self) -> u32 {
        self.value_count
    }

    /// Returns the number of null values.
    pub fn null_count(&self) -> u32 {
        self.null_count
    }
}
