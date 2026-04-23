// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! String compression statistics.

use vortex_array::ExecutionCtx;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::varbinview::VarBinViewArrayExt;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_utils::aliases::hash_set::HashSet;

use super::GenerateStatsOptions;

/// Array of variable-length byte arrays, and relevant stats for compression.
#[derive(Clone, Debug)]
pub struct StringStats {
    /// The estimated number of distinct strings, or `None` if not computed.
    /// This _must_ be non-zero.
    estimated_distinct_count: Option<u32>,
    /// The number of non-null values.
    value_count: u32,
    /// The number of null values.
    null_count: u32,
    /// The total byte length of all non-null string values.
    total_value_bytes: u64,
}

/// Returns the length-plus-prefix key currently used for approximate string distinct counts.
fn view_len_and_prefix(view: u128) -> u64 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "approximate uniqueness with view prefix"
    )]
    {
        view as u64
    }
}

/// Visits the varbin views that correspond to non-null string values.
fn for_each_valid_view(strings: &VarBinViewArray, validity: &Mask, mut visit: impl FnMut(u64)) {
    let views = strings.views();

    match validity {
        Mask::AllTrue(_) => {
            views
                .iter()
                .for_each(|view| visit(view_len_and_prefix(view.as_u128())));
        }
        Mask::AllFalse(_) => {}
        Mask::Values(values) => {
            values
                .indices()
                .iter()
                .for_each(|&idx| visit(view_len_and_prefix(views[idx].as_u128())));
        }
    }
}

impl StringStats {
    /// Generates stats, returning an error on failure.
    fn generate_opts_fallible(
        input: &VarBinViewArray,
        opts: GenerateStatsOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Self> {
        let validity = input.varbinview_validity().execute_mask(input.len(), ctx)?;
        let value_count = validity.true_count();
        let null_count = input.len() - value_count;
        let mut total_value_bytes = 0u64;
        let mut distinct_values = opts
            .count_distinct_values
            .then(|| HashSet::with_capacity(value_count / 2));

        for_each_valid_view(input, &validity, |view| {
            if let Some(distinct_values) = &mut distinct_values {
                distinct_values.insert(view);
            }

            let len = view & u64::from(u32::MAX);
            total_value_bytes += len;
        });

        let estimated_distinct_count = distinct_values
            .map(|distinct_values| u32::try_from(distinct_values.len()))
            .transpose()?;

        Ok(Self {
            value_count: u32::try_from(value_count)?,
            null_count: u32::try_from(null_count)?,
            total_value_bytes,
            estimated_distinct_count,
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

    /// Returns the estimated number of distinct strings, or `None` if not computed.
    ///
    /// This estimation is always going to be less than or equal to the actual distinct count.
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

    /// Returns the total byte length of all non-null string values.
    pub fn total_value_bytes(&self) -> u64 {
        self.total_value_bytes
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;

    use super::*;

    #[test]
    fn string_stats_only_count_non_null_value_bytes() {
        let strings = VarBinViewArray::from_iter(
            [Some("alpha"), None, Some("beta"), Some("alpha")],
            DType::Utf8(Nullability::Nullable),
        );
        let mut ctx = LEGACY_SESSION.create_execution_ctx();

        let stats = StringStats::generate_opts(
            &strings,
            GenerateStatsOptions {
                count_distinct_values: true,
            },
            &mut ctx,
        );

        assert_eq!(stats.value_count(), 3);
        assert_eq!(stats.null_count(), 1);
        assert_eq!(stats.total_value_bytes(), 14);
        assert_eq!(stats.estimated_distinct_count(), Some(2));
    }
}
