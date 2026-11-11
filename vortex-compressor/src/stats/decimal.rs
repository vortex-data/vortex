// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decimal compression statistics.
//!
//! Decimals stored in integer types take the integer statistics pass. `i128` decimals take the
//! chunked scan of that pass for runs and distinct values, without its bounds. `i256` decimals,
//! which are not primitive integers, take a scalar pass.

use std::hash::Hash;

use num_traits::PrimInt;
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
use vortex_utils::aliases::hash_map::HashMap;

use super::GenerateStatsOptions;
use super::IntegerStats;
use super::integer::Counter;
use super::integer::NoCounter;
use super::integer::scan;

/// Decimal array statistics for compression.
#[derive(Debug, Clone)]
pub struct DecimalStats {
    /// Cache for `validity.false_count()`.
    null_count: u32,
    /// Cache for `validity.true_count()`.
    value_count: u32,
    /// The average run length.
    average_run_length: u32,
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
        match input.values_type() {
            DecimalType::I8 => integer_stats::<i8>(input, opts, ctx),
            DecimalType::I16 => integer_stats::<i16>(input, opts, ctx),
            DecimalType::I32 => integer_stats::<i32>(input, opts, ctx),
            DecimalType::I64 => integer_stats::<i64>(input, opts, ctx),
            DecimalType::I128 => wide_stats::<i128>(input, opts, ctx),
            DecimalType::I256 => i256_stats(input, opts, ctx),
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

    /// Returns the average run length.
    pub fn average_run_length(&self) -> u32 {
        self.average_run_length
    }

    /// Returns the count of distinct valid values, if computed.
    pub fn distinct_count(&self) -> Option<u32> {
        self.distinct_count
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
        average_run_length: stats.average_run_length(),
        distinct_count: stats.distinct_count(),
    })
}

/// Computes the stats of decimals wider than any integer type with the chunked scan of the
/// integer statistics, which counts their runs and, if requested, their distinct values.
fn wide_stats<T: NativeDecimalType + Hash + PrimInt>(
    input: &DecimalArray,
    opts: GenerateStatsOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<DecimalStats> {
    let validity = input.validity()?.execute_mask(input.len(), ctx)?;
    let value_count = u32::try_from(validity.true_count())?;
    let null_count = u32::try_from(validity.false_count())?;

    if value_count == 0 {
        return Ok(DecimalStats {
            null_count,
            value_count,
            average_run_length: 0,
            distinct_count: opts.count_distinct_values.then_some(0),
        });
    }

    let buffer = input.buffer::<T>();
    let values = buffer.as_slice();
    let (runs, distinct_count) = if opts.count_distinct_values {
        let mut counter = DistinctCounter::<T>::new(values.len());
        let runs = scan::<T, _, false, true>(values, &validity, &mut counter).runs;
        (runs, Some(u32::try_from(counter.0.len())?))
    } else {
        (
            scan::<T, _, false, true>(values, &validity, &mut NoCounter).runs,
            None,
        )
    };

    Ok(DecimalStats {
        null_count,
        value_count,
        average_run_length: value_count / runs,
        distinct_count,
    })
}

/// Computes the stats of `i256` decimals.
///
/// `i256` is not a primitive integer, so it cannot take the chunked integer pass, which keeps
/// integer bounds. Making that pass generic over the bounds slowed the integer statistics, so
/// `i256` takes this scalar pass, which also counts each run once.
fn i256_stats(
    input: &DecimalArray,
    opts: GenerateStatsOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<DecimalStats> {
    let validity = input.validity()?.execute_mask(input.len(), ctx)?;
    let value_count = u32::try_from(validity.true_count())?;
    let null_count = u32::try_from(validity.false_count())?;
    let buffer = input.buffer::<i256>();
    let values = buffer.as_slice();

    let mut counter = opts
        .count_distinct_values
        .then(|| DistinctCounter::<i256>::new(values.len()));
    let mut runs = 0u32;
    let mut prev = None;
    let mut visit = |value: i256| {
        if prev != Some(value) {
            runs += 1;
            prev = Some(value);
            if let Some(counter) = counter.as_mut() {
                counter.add(value, 1);
            }
        }
    };
    match validity.indices() {
        AllOr::All => values.iter().copied().for_each(&mut visit),
        AllOr::None => {}
        AllOr::Some(indices) => indices.iter().for_each(|&idx| visit(values[idx])),
    }

    Ok(DecimalStats {
        null_count,
        value_count,
        average_run_length: value_count.checked_div(runs).unwrap_or(0),
        distinct_count: counter
            .map(|counter| u32::try_from(counter.0.len()))
            .transpose()?,
    })
}

/// Counts the occurrences of each distinct wide decimal value.
struct DistinctCounter<T>(HashMap<T, u32, FxBuildHasher>);

impl<T> DistinctCounter<T> {
    /// Creates a counter sized like the integer statistics' hash counter.
    fn new(len: usize) -> Self {
        Self(HashMap::with_capacity_and_hasher(len / 2, FxBuildHasher))
    }
}

impl<T: Eq + Hash> Counter<T> for DistinctCounter<T> {
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn add(&mut self, value: T, count: u32) {
        *self.0.entry(value).or_insert(0) += count;
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
    fn counts_runs(#[case] array: DecimalArray) {
        let mut ctx = SESSION.create_execution_ctx();
        let stats = DecimalStats::generate(&array, &mut ctx);
        // 8 values in 3 runs.
        assert_eq!(stats.average_run_length(), 2);
        assert_eq!(stats.distinct_count(), None);
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
