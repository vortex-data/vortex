// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;

use itertools::Itertools;
use num_traits::Float;
use rustc_hash::FxBuildHasher;
use vortex_array::ToCanonical;
use vortex_array::arrays::NativeValue;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::PrimitiveVTable;
use vortex_dtype::NativePType;
use vortex_dtype::PType;
use vortex_dtype::half::f16;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_mask::AllOr;
use vortex_utils::aliases::hash_set::HashSet;

use crate::CompressorStats;
use crate::GenerateStatsOptions;
use crate::rle::RLEStats;
use crate::sample::sample;

#[derive(Debug, Clone)]
pub struct DistinctValues<T> {
    pub values: HashSet<NativeValue<T>, FxBuildHasher>,
}

#[derive(Debug, Clone)]
pub enum ErasedDistinctValues {
    F16(DistinctValues<f16>),
    F32(DistinctValues<f32>),
    F64(DistinctValues<f64>),
}

macro_rules! impl_from_typed {
    ($typ:ty, $variant:path) => {
        impl From<DistinctValues<$typ>> for ErasedDistinctValues {
            fn from(value: DistinctValues<$typ>) -> Self {
                $variant(value)
            }
        }
    };
}

impl_from_typed!(f16, ErasedDistinctValues::F16);
impl_from_typed!(f32, ErasedDistinctValues::F32);
impl_from_typed!(f64, ErasedDistinctValues::F64);

/// Array of floating-point numbers and relevant stats for compression.
#[derive(Debug, Clone)]
pub struct FloatStats {
    pub(super) src: PrimitiveArray,
    // cache for validity.false_count()
    pub(super) null_count: u32,
    // cache for validity.true_count()
    pub(super) value_count: u32,
    #[allow(dead_code)]
    pub(super) average_run_length: u32,
    pub(super) distinct_values: ErasedDistinctValues,
    pub(super) distinct_values_count: u32,
}

impl FloatStats {
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
}

impl CompressorStats for FloatStats {
    type ArrayVTable = PrimitiveVTable;

    fn generate_opts(input: &PrimitiveArray, opts: GenerateStatsOptions) -> Self {
        Self::generate_opts_fallible(input, opts)
            .vortex_expect("FloatStats::generate_opts should not fail")
    }

    fn source(&self) -> &PrimitiveArray {
        &self.src
    }

    fn sample_opts(&self, sample_size: u32, sample_count: u32, opts: GenerateStatsOptions) -> Self {
        let sampled = sample(self.src.as_ref(), sample_size, sample_count).to_primitive();

        Self::generate_opts(&sampled, opts)
    }
}

impl RLEStats for FloatStats {
    fn value_count(&self) -> u32 {
        self.value_count
    }

    fn average_run_length(&self) -> u32 {
        self.average_run_length
    }

    fn source(&self) -> &PrimitiveArray {
        &self.src
    }
}

fn typed_float_stats<T: NativePType + Float>(
    array: &PrimitiveArray,
    count_distinct_values: bool,
) -> VortexResult<FloatStats>
where
    DistinctValues<T>: Into<ErasedDistinctValues>,
    NativeValue<T>: Hash + Eq,
{
    // Special case: empty array
    if array.is_empty() {
        return Ok(FloatStats {
            src: array.clone(),
            null_count: 0,
            value_count: 0,
            average_run_length: 0,
            distinct_values_count: 0,
            distinct_values: DistinctValues {
                values: HashSet::<NativeValue<T>, FxBuildHasher>::with_hasher(FxBuildHasher),
            }
            .into(),
        });
    } else if array.all_invalid()? {
        return Ok(FloatStats {
            src: array.clone(),
            null_count: u32::try_from(array.len())?,
            value_count: 0,
            average_run_length: 0,
            distinct_values_count: 0,
            distinct_values: DistinctValues {
                values: HashSet::<NativeValue<T>, FxBuildHasher>::with_hasher(FxBuildHasher),
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
    let distinct_values_count = if count_distinct_values {
        u32::try_from(distinct_values.len())?
    } else {
        u32::MAX
    };

    Ok(FloatStats {
        null_count,
        value_count,
        distinct_values_count,
        src: array.clone(),
        average_run_length: value_count / runs,
        distinct_values: DistinctValues {
            values: distinct_values,
        }
        .into(),
    })
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::ToCanonical;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;

    use crate::CompressorStats;
    use crate::float::stats::FloatStats;

    #[test]
    fn test_float_stats() {
        let floats = buffer![0.0f32, 1.0f32, 2.0f32].into_array();
        let floats = floats.to_primitive();

        let stats = FloatStats::generate(&floats);

        assert_eq!(stats.value_count, 3);
        assert_eq!(stats.null_count, 0);
        assert_eq!(stats.average_run_length, 1);
        assert_eq!(stats.distinct_values_count, 3);
    }

    #[test]
    fn test_float_stats_leading_nulls() {
        let floats = PrimitiveArray::new(
            buffer![0.0f32, 1.0f32, 2.0f32],
            Validity::from_iter([false, true, true]),
        );

        let stats = FloatStats::generate(&floats);

        assert_eq!(stats.value_count, 2);
        assert_eq!(stats.null_count, 1);
        assert_eq!(stats.average_run_length, 1);
        assert_eq!(stats.distinct_values_count, 2);
    }
}
