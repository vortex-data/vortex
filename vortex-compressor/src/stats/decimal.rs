// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decimal compression statistics.
//!
//! Decimals stored in integer types take the integer statistics pass. When the distinct values
//! are counted, `i128` decimals take the chunked scan of that pass, without its bounds, and `i256`
//! decimals, which are not primitive integers, take a scalar pass. Both count the runs with the
//! distinct values, which the dictionary cost model reads, as the integer statistics do.

use std::hash::Hash;

use rustc_hash::FxBuildHasher;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::decimal::DecimalArrayExt;
use vortex_array::dtype::DecimalType;
use vortex_array::dtype::NativeDecimalType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::i256;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::AllOr;
use vortex_mask::Mask;
use vortex_utils::aliases::hash_set::HashSet;

use super::GenerateStatsOptions;
use super::IntegerStats;
use super::integer::Counter;
use super::integer::scan;

/// Decimal array statistics for compression.
#[derive(Debug, Clone)]
pub struct DecimalStats {
    /// Cache for `validity.false_count()`.
    null_count: u32,
    /// Cache for `validity.true_count()`.
    value_count: u32,
    /// The distinct valid values and runs, or `None` if not computed.
    distinct: Option<Distinct>,
}

/// The distinct valid values of a decimal array and its runs.
#[derive(Debug, Clone, Copy)]
struct Distinct {
    /// The count of distinct valid values.
    count: u32,
    /// The average length of runs of equal consecutive valid values.
    average_run_length: u32,
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
        match input.values_type() {
            DecimalType::I8 => integer_stats::<i8>(input, opts, ctx),
            DecimalType::I16 => integer_stats::<i16>(input, opts, ctx),
            DecimalType::I32 => integer_stats::<i32>(input, opts, ctx),
            DecimalType::I64 => integer_stats::<i64>(input, opts, ctx),
            DecimalType::I128 => wide_stats(input, opts, ctx, i128_distinct),
            DecimalType::I256 => wide_stats(input, opts, ctx, i256_distinct),
        }
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
        self.distinct.map(|distinct| distinct.count)
    }

    /// Returns the average length of runs of equal consecutive valid values, if computed with the
    /// distinct values.
    pub fn average_run_length(&self) -> Option<u32> {
        self.distinct.map(|distinct| distinct.average_run_length)
    }
}

/// Computes the stats of decimals stored in the integer type `T` with the integer statistics,
/// over the same buffer and validity.
fn integer_stats<T: NativeDecimalType + NativePType>(
    input: &DecimalArray,
    opts: GenerateStatsOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<DecimalStats> {
    let integers = PrimitiveArray::new(input.buffer::<T>(), input.validity()?);
    let stats = IntegerStats::generate_opts(&integers, opts, ctx);
    Ok(DecimalStats {
        null_count: stats.null_count(),
        value_count: stats.value_count(),
        distinct: stats.distinct_count().map(|count| Distinct {
            count,
            average_run_length: stats.average_run_length(),
        }),
    })
}

/// Computes the stats of decimals stored wider than 64 bits, counting the distinct values and
/// runs with `distinct` when requested.
fn wide_stats(
    input: &DecimalArray,
    opts: GenerateStatsOptions,
    ctx: &mut ExecutionCtx,
    distinct: fn(&DecimalArray, &Mask) -> VortexResult<(u32, u32)>,
) -> VortexResult<DecimalStats> {
    let validity = input.validity()?.execute_mask(input.len(), ctx)?;
    let value_count = u32::try_from(validity.true_count())?;
    let null_count = u32::try_from(validity.false_count())?;

    let distinct = if !opts.count_distinct_values {
        None
    } else if value_count == 0 {
        Some(Distinct {
            count: 0,
            average_run_length: 0,
        })
    } else {
        let (count, runs) = distinct(input, &validity)?;
        Some(Distinct {
            count,
            average_run_length: value_count / runs,
        })
    };

    Ok(DecimalStats {
        null_count,
        value_count,
        distinct,
    })
}

/// Returns the count of distinct valid `i128` values and the runs, with the chunked scan of the
/// integer statistics. `validity` has at least one valid value.
fn i128_distinct(input: &DecimalArray, validity: &Mask) -> VortexResult<(u32, u32)> {
    let buffer = input.buffer::<i128>();
    let mut distinct = DistinctSet::with_capacity(buffer.len());
    let runs = scan::<i128, _, false, true>(buffer.as_slice(), validity, &mut distinct).runs;
    Ok((u32::try_from(distinct.0.len())?, runs))
}

/// Returns the count of distinct valid `i256` values and the runs. `validity` has at least one
/// valid value.
///
/// `i256` is not a primitive integer, so it cannot take the chunked scan of the integer
/// statistics, which keeps integer bounds. Making that scan generic over the bounds slowed the
/// integer statistics, so `i256` takes this scalar pass, which also visits the set once per run.
fn i256_distinct(input: &DecimalArray, validity: &Mask) -> VortexResult<(u32, u32)> {
    let buffer = input.buffer::<i256>();
    let values = buffer.as_slice();
    let mut distinct = DistinctSet::with_capacity(values.len());
    let mut runs = 0u32;
    let mut prev = None;
    let mut visit = |value: i256| {
        if prev != Some(value) {
            runs += 1;
            prev = Some(value);
            distinct.add(value, 1);
        }
    };
    match validity.indices() {
        AllOr::All => values.iter().copied().for_each(&mut visit),
        AllOr::None => {}
        AllOr::Some(indices) => indices.iter().for_each(|&idx| visit(values[idx])),
    }
    Ok((u32::try_from(distinct.0.len())?, runs))
}

/// The distinct values of a wide decimal array.
struct DistinctSet<T>(HashSet<T, FxBuildHasher>);

impl<T> DistinctSet<T> {
    /// Creates a set sized like the integer statistics' hash counter.
    fn with_capacity(len: usize) -> Self {
        Self(HashSet::with_capacity_and_hasher(len / 2, FxBuildHasher))
    }
}

impl<T: Eq + Hash> Counter<T> for DistinctSet<T> {
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn add(&mut self, value: T, _count: u32) {
        self.0.insert(value);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::DecimalArray;
    use vortex_array::dtype::BigCast;
    use vortex_array::dtype::DecimalDType;
    use vortex_array::dtype::DecimalType;
    use vortex_array::dtype::i256;
    use vortex_array::match_each_decimal_value_type;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
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

    /// The `i128` and `i256` passes agree with the integer pass on the same values, across chunk
    /// boundaries, runs and nulls.
    #[rstest]
    fn wide_passes_match_integer_pass(
        #[values(1, 63, 64, 65, 200, 1000)] len: usize,
        #[values(1, 3, 70)] run_len: usize,
        #[values(false, true)] nullable: bool,
        #[values(DecimalType::I128, DecimalType::I256)] wide_type: DecimalType,
    ) {
        let mut ctx = SESSION.create_execution_ctx();
        let validity = || {
            if nullable {
                Validity::from_iter((0..len).map(|i| i % 5 != 0 && i % 67 != 3))
            } else {
                Validity::NonNullable
            }
        };
        let values = (0..len)
            .map(|i| ((i / run_len) % 7) as i64 - 3)
            .collect::<Vec<_>>();
        let decimal = |values_type: DecimalType| {
            match_each_decimal_value_type!(values_type, |T| {
                DecimalArray::new(
                    values
                        .iter()
                        .map(|&v| <T as BigCast>::from(v).vortex_expect("fits"))
                        .collect::<Buffer<T>>(),
                    DecimalDType::new(18, 2),
                    validity(),
                )
            })
        };

        let narrow =
            DecimalStats::generate_opts(&decimal(DecimalType::I64), COUNT_DISTINCT, &mut ctx);
        let wide = DecimalStats::generate_opts(&decimal(wide_type), COUNT_DISTINCT, &mut ctx);
        assert_eq!(wide.null_count(), narrow.null_count());
        assert_eq!(wide.value_count(), narrow.value_count());
        assert_eq!(wide.average_run_length(), narrow.average_run_length());
        assert_eq!(wide.distinct_count(), narrow.distinct_count());
    }

    #[rstest]
    #[case::i64(DecimalArray::new(
        buffer![1i64, 1, 1, 2, 2, 3, 3, 3],
        DecimalDType::new(18, 2),
        Validity::NonNullable,
    ))]
    #[case::i256(DecimalArray::new(
        Buffer::copy_from(
            [1, 1, 1, 2, 2, 3, 3, 3].map(|v| i256::from_parts(v, 1)),
        ),
        DecimalDType::new(76, 2),
        Validity::NonNullable,
    ))]
    #[case::i128(DecimalArray::new(
        buffer![1i128, 1, 1, 2, 2, 3, 3, 3],
        DecimalDType::new(38, 2),
        Validity::NonNullable,
    ))]
    fn counts_runs(#[case] array: DecimalArray) {
        let mut ctx = SESSION.create_execution_ctx();
        let stats = DecimalStats::generate_opts(&array, COUNT_DISTINCT, &mut ctx);
        // 8 values in 3 runs.
        assert_eq!(stats.average_run_length(), Some(2));
        assert_eq!(stats.distinct_count(), Some(3));
    }

    #[rstest]
    #[case::i128(DecimalArray::new(
        buffer![5i128, 7, 5],
        DecimalDType::new(38, 2),
        Validity::NonNullable,
    ))]
    #[case::i256(DecimalArray::new(
        Buffer::copy_from([5, 7, 5].map(|v| i256::from_parts(v, 1))),
        DecimalDType::new(76, 2),
        Validity::NonNullable,
    ))]
    fn skips_distinct_values_unless_requested(#[case] array: DecimalArray) {
        let mut ctx = SESSION.create_execution_ctx();
        let stats = DecimalStats::generate(&array, &mut ctx);
        assert_eq!(stats.value_count(), 3);
        assert_eq!(stats.distinct_count(), None);
        assert_eq!(stats.average_run_length(), None);
    }
}
