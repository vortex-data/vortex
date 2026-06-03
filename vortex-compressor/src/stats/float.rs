// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Float compression statistics.

use std::hash::Hash;
use std::marker::PhantomData;

use itertools::Itertools;
use num_traits::Float;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::NativeValue;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_array::dtype::half::f16;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_mask::AllOr;

use super::GenerateStatsOptions;
use super::cardinality::CardinalityEstimator;

/// Information about the distinct values in a float array.
///
/// The distinct count is an estimate produced by Cloudflare's cardinality estimator, which is
/// exact for small cardinalities and approximate beyond that.
#[derive(Debug, Clone)]
pub struct DistinctInfo<T> {
    /// The estimated count of unique values. This _must_ be non-zero.
    distinct_count: usize,
    /// Phantom marker for the float element type.
    _marker: PhantomData<T>,
}

/// Typed statistics for a specific float type.
#[derive(Debug, Clone)]
pub struct TypedStats<T> {
    /// Distinct value information, or `None` if not computed.
    distinct: Option<DistinctInfo<T>>,
}

impl<T> TypedStats<T> {
    /// Returns the distinct value information, if computed.
    pub fn distinct(&self) -> Option<&DistinctInfo<T>> {
        self.distinct.as_ref()
    }
}

/// Type-erased container for one of the [`TypedStats`] variants.
#[derive(Debug, Clone)]
pub enum ErasedStats {
    /// Stats for `f16` arrays.
    F16(TypedStats<f16>),
    /// Stats for `f32` arrays.
    F32(TypedStats<f32>),
    /// Stats for `f64` arrays.
    F64(TypedStats<f64>),
}

impl ErasedStats {
    /// Get the count of distinct values, if we have computed it already.
    fn distinct_count(&self) -> Option<usize> {
        match self {
            ErasedStats::F16(x) => x.distinct.as_ref().map(|d| d.distinct_count),
            ErasedStats::F32(x) => x.distinct.as_ref().map(|d| d.distinct_count),
            ErasedStats::F64(x) => x.distinct.as_ref().map(|d| d.distinct_count),
        }
    }
}

/// Implements `From<TypedStats<$T>>` for [`ErasedStats`].
macro_rules! impl_from_typed {
    ($T:ty, $variant:path) => {
        impl From<TypedStats<$T>> for ErasedStats {
            fn from(typed: TypedStats<$T>) -> Self {
                $variant(typed)
            }
        }
    };
}

impl_from_typed!(f16, ErasedStats::F16);
impl_from_typed!(f32, ErasedStats::F32);
impl_from_typed!(f64, ErasedStats::F64);

/// Array of floating-point numbers and relevant stats for compression.
#[derive(Debug, Clone)]
pub struct FloatStats {
    /// Cache for `validity.false_count()`.
    null_count: usize,
    /// Cache for `validity.true_count()`.
    value_count: usize,
    /// The average run length.
    average_run_length: usize,
    /// Type-erased typed statistics.
    erased: ErasedStats,
}

impl FloatStats {
    /// Generates stats, returning an error on failure.
    fn generate_opts_fallible(
        input: &PrimitiveArray,
        opts: GenerateStatsOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Self> {
        match input.ptype() {
            PType::F16 => typed_float_stats::<f16>(input, opts.count_distinct_values, ctx),
            PType::F32 => typed_float_stats::<f32>(input, opts.count_distinct_values, ctx),
            PType::F64 => typed_float_stats::<f64>(input, opts.count_distinct_values, ctx),
            _ => vortex_panic!("cannot generate FloatStats from ptype {}", input.ptype()),
        }
    }

    /// Get the count of distinct values, if we have computed it already.
    pub fn distinct_count(&self) -> Option<usize> {
        self.erased.distinct_count()
    }
}

impl FloatStats {
    /// Generates stats with default options.
    pub fn generate(input: &PrimitiveArray, ctx: &mut ExecutionCtx) -> Self {
        Self::generate_opts(input, GenerateStatsOptions::default(), ctx)
    }

    /// Generates stats with provided options.
    pub fn generate_opts(
        input: &PrimitiveArray,
        opts: GenerateStatsOptions,
        ctx: &mut ExecutionCtx,
    ) -> Self {
        Self::generate_opts_fallible(input, opts, ctx)
            .vortex_expect("FloatStats::generate_opts should not fail")
    }

    /// Returns the number of null values.
    pub fn null_count(&self) -> usize {
        self.null_count
    }

    /// Returns the number of non-null values.
    pub fn value_count(&self) -> usize {
        self.value_count
    }

    /// Returns the average run length.
    pub fn average_run_length(&self) -> usize {
        self.average_run_length
    }

    /// Returns the type-erased typed statistics.
    pub fn erased(&self) -> &ErasedStats {
        &self.erased
    }
}

/// Computes typed float statistics for a specific float type.
fn typed_float_stats<T: NativePType + Float>(
    array: &PrimitiveArray,
    count_distinct_values: bool,
    ctx: &mut ExecutionCtx,
) -> VortexResult<FloatStats>
where
    NativeValue<T>: Hash + Eq,
    TypedStats<T>: Into<ErasedStats>,
{
    // Special case: empty array.
    if array.is_empty() {
        return Ok(FloatStats {
            null_count: 0,
            value_count: 0,
            average_run_length: 0,
            erased: TypedStats { distinct: None }.into(),
        });
    }

    if array.all_invalid(ctx)? {
        return Ok(FloatStats {
            null_count: array.len(),
            value_count: 0,
            average_run_length: 0,
            erased: TypedStats {
                distinct: Some(DistinctInfo {
                    distinct_count: 0,
                    _marker: PhantomData,
                }),
            }
            .into(),
        });
    }

    let null_count = array
        .statistics()
        .compute_null_count(ctx)
        .ok_or_else(|| vortex_err!("Failed to compute null_count"))?;
    let value_count = array.len() - null_count;

    // Cloudflare's cardinality estimator gives us a bounded-memory approximation of the
    // number of distinct values, replacing the previous exact `HashSet`.
    let mut estimator: CardinalityEstimator<NativeValue<T>> = CardinalityEstimator::new();

    let validity = array
        .as_ref()
        .validity()?
        .execute_mask(array.as_ref().len(), ctx)?;

    let mut runs = 1;
    let head_idx = validity
        .first()
        .vortex_expect("All null masks have been handled before");
    let buff = array.to_buffer::<T>();
    let mut prev = buff[head_idx];

    let first_valid_buff = buff.slice(head_idx..array.len());
    match validity.bit_buffer() {
        AllOr::All => {
            for value in first_valid_buff {
                if count_distinct_values {
                    estimator.insert(&NativeValue(value));
                }

                if value != prev {
                    prev = value;
                    runs += 1;
                }
            }
        }
        AllOr::None => unreachable!("All invalid arrays have been handled earlier"),
        AllOr::Some(v) => {
            for (&value, valid) in first_valid_buff
                .iter()
                .zip_eq(v.slice(head_idx..array.len()).iter())
            {
                if valid {
                    if count_distinct_values {
                        estimator.insert(&NativeValue(value));
                    }

                    if value != prev {
                        prev = value;
                        runs += 1;
                    }
                }
            }
        }
    }

    let distinct = count_distinct_values.then(|| DistinctInfo {
        distinct_count: estimator.estimate().max(1),
        _marker: PhantomData,
    });

    Ok(FloatStats {
        null_count,
        value_count,
        average_run_length: value_count / runs,
        erased: TypedStats { distinct }.into(),
    })
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::session::ArraySession;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::FloatStats;
    use crate::stats::GenerateStatsOptions;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    #[test]
    fn test_float_stats() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let floats = buffer![0.0f32, 1.0f32, 2.0f32].into_array();
        let floats = floats.execute::<PrimitiveArray>(&mut ctx)?;

        let stats = FloatStats::generate_opts(
            &floats,
            GenerateStatsOptions {
                count_distinct_values: true,
            },
            &mut ctx,
        );

        assert_eq!(stats.value_count, 3);
        assert_eq!(stats.null_count, 0);
        assert_eq!(stats.average_run_length, 1);
        assert_eq!(stats.distinct_count().unwrap(), 3);
        Ok(())
    }

    #[test]
    fn test_float_stats_leading_nulls() {
        let mut ctx = SESSION.create_execution_ctx();
        let floats = PrimitiveArray::new(
            buffer![0.0f32, 1.0f32, 2.0f32],
            Validity::from_iter([false, true, true]),
        );

        let stats = FloatStats::generate_opts(
            &floats,
            GenerateStatsOptions {
                count_distinct_values: true,
            },
            &mut ctx,
        );

        assert_eq!(stats.value_count, 2);
        assert_eq!(stats.null_count, 1);
        assert_eq!(stats.average_run_length, 1);
        assert_eq!(stats.distinct_count().unwrap(), 2);
    }
}
