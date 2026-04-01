// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! String compression statistics.

use vortex_array::arrays::VarBinViewArray;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_utils::aliases::hash_set::HashSet;

use super::GenerateStatsOptions;

/// Array of variable-length byte arrays, and relevant stats for compression.
#[derive(Clone, Debug)]
pub struct StringStats {
    /// The underlying source array.
    src: VarBinViewArray,
    /// The estimated number of distinct strings, or `None` if not computed.
    estimated_distinct_count: Option<u32>,
    /// The number of non-null values.
    value_count: u32,
    /// The number of null values.
    null_count: u32,
}

/// Estimate the number of distinct strings in the var bin view array.
fn estimate_distinct_count(strings: &VarBinViewArray) -> VortexResult<u32> {
    let views = strings.views();
    // Iterate the views. Two strings which are equal must have the same first 8-bytes.
    // NOTE: there are cases where this performs pessimally, e.g. when we have strings that all
    // share a 4-byte prefix and have the same length.
    let mut distinct = HashSet::with_capacity(views.len() / 2);
    views.iter().for_each(|&view| {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "approximate uniqueness with view prefix"
        )]
        let len_and_prefix = view.as_u128() as u64;
        distinct.insert(len_and_prefix);
    });

    Ok(u32::try_from(distinct.len())?)
}

impl StringStats {
    /// Generates stats, returning an error on failure.
    fn generate_opts_fallible(
        input: &VarBinViewArray,
        opts: GenerateStatsOptions,
    ) -> VortexResult<Self> {
        let null_count = input
            .statistics()
            .compute_null_count()
            .ok_or_else(|| vortex_err!("Failed to compute null_count"))?;
        let value_count = input.len() - null_count;
        let estimated_distinct_count = opts
            .count_distinct_values
            .then(|| estimate_distinct_count(input))
            .transpose()?;

        Ok(Self {
            src: input.clone(),
            value_count: u32::try_from(value_count)?,
            null_count: u32::try_from(null_count)?,
            estimated_distinct_count,
        })
    }
}

impl StringStats {
    /// Generates stats with default options.
    pub fn generate(input: &VarBinViewArray) -> Self {
        Self::generate_opts(input, GenerateStatsOptions::default())
    }

    /// Generates stats with provided options.
    pub fn generate_opts(input: &VarBinViewArray, opts: GenerateStatsOptions) -> Self {
        Self::generate_opts_fallible(input, opts)
            .vortex_expect("StringStats::generate_opts should not fail")
    }

    /// Returns the underlying source array.
    pub fn source(&self) -> &VarBinViewArray {
        &self.src
    }

    /// Returns the estimated number of distinct strings, or `None` if not computed.
    pub fn estimated_distinct_count(&self) -> Option<u32> {
        self.estimated_distinct_count
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
