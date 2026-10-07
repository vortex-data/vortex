// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Float compression statistics.

use std::hash::Hash;

use num_traits::Float;
use rustc_hash::FxBuildHasher;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::NativeValue;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_array::dtype::half::f16;
use vortex_compute::lane_kernels::for_each_chunk;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_mask::AllOr;
use vortex_utils::aliases::hash_set::HashSet;

use super::GenerateStatsOptions;

/// Information about the distinct values in a float array.
#[derive(Debug, Clone)]
pub struct DistinctInfo<T> {
    /// The set of distinct float values.
    distinct_values: HashSet<NativeValue<T>, FxBuildHasher>,
    /// The count of unique values. This _must_ be non-zero.
    distinct_count: u32,
}

impl<T> DistinctInfo<T> {
    /// Returns a reference to the distinct values set.
    pub fn distinct_values(&self) -> &HashSet<NativeValue<T>, FxBuildHasher> {
        &self.distinct_values
    }
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
    fn distinct_count(&self) -> Option<u32> {
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
    null_count: u32,
    /// Cache for `validity.true_count()`.
    value_count: u32,
    /// The average run length.
    average_run_length: u32,
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
    pub fn distinct_count(&self) -> Option<u32> {
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
            null_count: u32::try_from(array.len())?,
            value_count: 0,
            average_run_length: 0,
            erased: TypedStats {
                distinct: Some(DistinctInfo {
                    distinct_values: HashSet::with_capacity_and_hasher(0, FxBuildHasher),
                    distinct_count: 0,
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

    // Keep a HashMap of T, then convert the keys into PValue afterward since value is
    // so much more efficient to hash and search for.
    let mut distinct_values = if count_distinct_values {
        HashSet::with_capacity_and_hasher(array.len() / 2, FxBuildHasher)
    } else {
        HashSet::with_hasher(FxBuildHasher)
    };

    let validity = array
        .as_ref()
        .validity()?
        .execute_mask(array.as_ref().len(), ctx)?;

    let mut runs: u32 = 1;
    let head_idx = validity
        .first()
        .vortex_expect("All null masks have been handled before");
    let buff = array.to_buffer::<T>();
    let mut prev = buff[head_idx];

    let validity_bits = match validity.bit_buffer() {
        AllOr::All => None,
        AllOr::None => unreachable!("All invalid arrays have been handled before"),
        AllOr::Some(bits) => Some(bits),
    };
    // The nulls before the head are skipped, so the loop can start at 0.
    for_each_chunk(buff.as_slice(), validity_bits, |chunk, valid| match valid {
        // All nulls -> no stats to update.
        0 => {}
        u64::MAX => {
            if count_distinct_values {
                distinct_values.extend(chunk.iter().map(|&value| NativeValue(value)));
            }
            // Branch-free count of value changes, including the change from the previous chunk.
            // At most 64, so a `u8` accumulator lets the comparison use full-width byte lanes.
            let transitions = u8::from(chunk[0] != prev)
                + chunk
                    .iter()
                    .zip(&chunk[1..])
                    .map(|(a, b)| u8::from(a != b))
                    .sum::<u8>();
            runs += u32::from(transitions);
            prev = chunk[63];
        }
        // Floats are not forward filled like integers, since a filled NaN would add a run.
        _ => {
            // Locals keep the run state in registers across the bit walk.
            let (mut chunk_prev, mut chunk_runs) = (prev, runs);
            let mut valid = valid;
            while valid != 0 {
                let value = chunk[valid.trailing_zeros() as usize];
                if count_distinct_values {
                    distinct_values.insert(NativeValue(value));
                }
                if value != chunk_prev {
                    chunk_prev = value;
                    chunk_runs += 1;
                }
                valid &= valid - 1;
            }
            (prev, runs) = (chunk_prev, chunk_runs);
        }
    });

    let null_count = u32::try_from(null_count)?;
    let value_count = u32::try_from(value_count)?;

    let distinct = count_distinct_values.then(|| DistinctInfo {
        distinct_count: u32::try_from(distinct_values.len())
            .vortex_expect("more than u32::MAX distinct values"),
        distinct_values,
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
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use super::FloatStats;
    use crate::stats::GenerateStatsOptions;

    #[test]
    fn test_float_stats() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
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
        let mut ctx = array_session().create_execution_ctx();
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

    /// Checks the chunked loop against a naive reference. The run lengths cover aligned constant
    /// chunks (64), runs that cross chunk boundaries (3, 100), and chunks where every value
    /// changes (1). With nulls, `null_chunks` also nulls two whole chunks.
    #[rstest]
    fn test_matches_naive_reference(
        #[values(1, 63, 64, 65, 64 * 20 + 17)] len: u32,
        #[values(1, 3, 64, 100)] run_len: u32,
        #[values(None, Some(97), Some(3))] null_every: Option<u32>,
        #[values(false, true)] null_chunks: bool,
    ) {
        let values: Vec<f64> = (0..len).map(|i| f64::from((i / run_len) % 16)).collect();
        let valid: Vec<bool> = (0..len)
            .map(|i| {
                null_every.is_none_or(|n| i % n != 0 && !(null_chunks && (64..192).contains(&i)))
            })
            .collect();

        let mut distinct = Vec::new();
        let mut runs = 0;
        let mut prev = None;
        for (&value, _) in values.iter().zip(&valid).filter(|(_, ok)| **ok) {
            if !distinct.contains(&value) {
                distinct.push(value);
            }
            if prev.replace(value) != Some(value) {
                runs += 1;
            }
        }
        let value_count = u32::try_from(valid.iter().filter(|ok| **ok).count()).unwrap();

        let validity = match null_every {
            None => Validity::NonNullable,
            Some(_) => Validity::from(BitBuffer::from(valid)),
        };
        let array = PrimitiveArray::new(Buffer::from(values), validity);
        let mut ctx = array_session().create_execution_ctx();
        let stats = FloatStats::generate_opts(
            &array,
            GenerateStatsOptions {
                count_distinct_values: true,
            },
            &mut ctx,
        );

        assert_eq!(stats.value_count, value_count);
        if value_count > 0 {
            assert_eq!(stats.average_run_length, value_count / runs);
            assert_eq!(
                stats.distinct_count(),
                Some(u32::try_from(distinct.len()).unwrap())
            );
        }
    }
}
