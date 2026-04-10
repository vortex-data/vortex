// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Float compression statistics.

use std::hash::Hash;

use itertools::Itertools;
use num_traits::Float;
use rustc_hash::FxBuildHasher;
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
    ) -> VortexResult<Self> {
        match input.ptype() {
            PType::F16 => typed_float_stats::<f16>(input, opts.count_distinct_values),
            PType::F32 => typed_float_stats::<f32>(input, opts.count_distinct_values),
            PType::F64 => typed_float_stats::<f64>(input, opts.count_distinct_values),
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
    pub fn generate(input: &PrimitiveArray) -> Self {
        Self::generate_opts(input, GenerateStatsOptions::default())
    }

    /// Generates stats with provided options.
    pub fn generate_opts(input: &PrimitiveArray, opts: GenerateStatsOptions) -> Self {
        Self::generate_opts_fallible(input, opts)
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

    if array.all_invalid()? {
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
        .compute_null_count()
        .ok_or_else(|| vortex_err!("Failed to compute null_count"))?;
    let value_count = array.len() - null_count;

    // Keep a HashMap of T, then convert the keys into PValue afterward since value is
    // so much more efficient to hash and search for.
    let mut distinct_values = if count_distinct_values {
        HashSet::with_capacity_and_hasher(array.len() / 2, FxBuildHasher)
    } else {
        HashSet::with_hasher(FxBuildHasher)
    };

    let validity = array.validity_mask()?;

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
                    distinct_values.insert(NativeValue(value));
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
                        distinct_values.insert(NativeValue(value));
                    }

                    if value != prev {
                        prev = value;
                        runs += 1;
                    }
                }
            }
        }
    }

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
    use vortex_array::IntoArray;
    use vortex_array::ToCanonical;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;

    use super::FloatStats;

    #[test]
    fn test_float_stats() {
        let floats = buffer![0.0f32, 1.0f32, 2.0f32].into_array();
        let floats = floats.to_primitive();

        let stats = FloatStats::generate_opts(
            &floats,
            crate::stats::GenerateStatsOptions {
                count_distinct_values: true,
            },
        );

        assert_eq!(stats.value_count, 3);
        assert_eq!(stats.null_count, 0);
        assert_eq!(stats.average_run_length, 1);
        assert_eq!(stats.distinct_count().unwrap(), 3);
    }

    #[test]
    fn test_float_stats_leading_nulls() {
        let floats = PrimitiveArray::new(
            buffer![0.0f32, 1.0f32, 2.0f32],
            Validity::from_iter([false, true, true]),
        );

        let stats = FloatStats::generate_opts(
            &floats,
            crate::stats::GenerateStatsOptions {
                count_distinct_values: true,
            },
        );

        assert_eq!(stats.value_count, 2);
        assert_eq!(stats.null_count, 1);
        assert_eq!(stats.average_run_length, 1);
        assert_eq!(stats.distinct_count().unwrap(), 2);
    }
}
