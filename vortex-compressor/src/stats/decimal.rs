// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decimal compression statistics.

use std::hash::Hash;

use rustc_hash::FxBuildHasher;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::decimal::DecimalArrayExt;
use vortex_array::dtype::NativeDecimalType;
use vortex_array::match_each_decimal_value_type;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::AllOr;
use vortex_utils::aliases::hash_set::HashSet;

use super::GenerateStatsOptions;

/// Decimal array statistics for compression.
#[derive(Debug, Clone)]
pub struct DecimalStats {
    /// Cache for `validity.false_count()`.
    null_count: u32,
    /// Cache for `validity.true_count()`.
    value_count: u32,
    /// The count of distinct valid values, or `None` if not computed.
    distinct_count: Option<u32>,
}

impl DecimalStats {
    /// Generates stats with default options.
    pub fn generate(input: &DecimalArray, ctx: &mut ExecutionCtx) -> Self {
        Self::generate_opts(input, GenerateStatsOptions::default(), ctx)
    }

    /// Generates stats with provided options.
    pub fn generate_opts(
        input: &DecimalArray,
        opts: GenerateStatsOptions,
        ctx: &mut ExecutionCtx,
    ) -> Self {
        Self::generate_opts_fallible(input, opts, ctx)
            .vortex_expect("DecimalStats::generate_opts should not fail")
    }

    /// Generates stats, returning an error on failure.
    fn generate_opts_fallible(
        input: &DecimalArray,
        opts: GenerateStatsOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Self> {
        let mask = input.validity()?.execute_mask(input.len(), ctx)?;
        let value_count = u32::try_from(mask.true_count())?;
        let null_count = u32::try_from(input.len())? - value_count;

        let distinct_count = if opts.count_distinct_values {
            Some(match_each_decimal_value_type!(input.values_type(), |T| {
                count_distinct(input.buffer::<T>().as_slice(), mask.indices())?
            }))
        } else {
            None
        };

        Ok(Self {
            null_count,
            value_count,
            distinct_count,
        })
    }

    /// Returns the number of null values.
    pub fn null_count(&self) -> u32 {
        self.null_count
    }

    /// Returns the number of non-null values.
    pub fn value_count(&self) -> u32 {
        self.value_count
    }

    /// Returns the count of distinct valid values, if computed.
    pub fn distinct_count(&self) -> Option<u32> {
        self.distinct_count
    }
}

/// Counts the distinct values at the valid positions.
fn count_distinct<T: NativeDecimalType + Hash>(
    values: &[T],
    valid: AllOr<&[usize]>,
) -> VortexResult<u32> {
    let mut distinct = HashSet::with_hasher(FxBuildHasher);
    match valid {
        AllOr::All => distinct.extend(values.iter().copied()),
        AllOr::None => {}
        AllOr::Some(indices) => distinct.extend(indices.iter().map(|&idx| values[idx])),
    }
    Ok(u32::try_from(distinct.len())?)
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::DecimalArray;
    use vortex_array::dtype::DecimalDType;
    use vortex_array::dtype::i256;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_session::VortexSession;

    use super::DecimalStats;
    use crate::stats::GenerateStatsOptions;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    const COUNT_DISTINCT: GenerateStatsOptions = GenerateStatsOptions {
        count_distinct_values: true,
    };

    #[rstest]
    #[case::i64(DecimalArray::new(
        buffer![5i64, 7, 5, 9, 7],
        DecimalDType::new(18, 2),
        Validity::NonNullable,
    ), 5, 0, 3)]
    #[case::i128_with_nulls(DecimalArray::new(
        buffer![1i128 << 100, 3, 1i128 << 100, 4, 3],
        DecimalDType::new(38, 2),
        Validity::from_iter([true, true, true, false, false]),
    ), 3, 2, 2)]
    #[case::i256(DecimalArray::new(
        Buffer::copy_from([i256::from_parts(1, 1), i256::from_parts(1, 2), i256::from_parts(1, 1)]),
        DecimalDType::new(76, 0),
        Validity::NonNullable,
    ), 3, 0, 2)]
    #[case::all_null(DecimalArray::new(
        buffer![1i32, 2],
        DecimalDType::new(9, 2),
        Validity::AllInvalid,
    ), 0, 2, 0)]
    fn counts_distinct_valid_values(
        #[case] array: DecimalArray,
        #[case] value_count: u32,
        #[case] null_count: u32,
        #[case] distinct_count: u32,
    ) {
        let mut ctx = SESSION.create_execution_ctx();
        let stats = DecimalStats::generate_opts(&array, COUNT_DISTINCT, &mut ctx);
        assert_eq!(stats.value_count(), value_count);
        assert_eq!(stats.null_count(), null_count);
        assert_eq!(stats.distinct_count(), Some(distinct_count));
    }

    #[test]
    fn skips_distinct_values_unless_requested() {
        let mut ctx = SESSION.create_execution_ctx();
        let array = DecimalArray::new(
            buffer![5i64, 7, 5],
            DecimalDType::new(18, 2),
            Validity::NonNullable,
        );
        assert_eq!(
            DecimalStats::generate(&array, &mut ctx).distinct_count(),
            None
        );
    }
}
