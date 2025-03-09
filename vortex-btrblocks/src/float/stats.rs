use std::hash::Hash;

use itertools::Itertools;
use num_traits::Float;
use rustc_hash::FxBuildHasher;
use vortex_array::aliases::hash_map::HashMap;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{Array, ToCanonical};
use vortex_dtype::half::f16;
use vortex_dtype::{NativePType, PType};
use vortex_error::{VortexExpect, VortexUnwrap, vortex_panic};
use vortex_mask::AllOr;

use crate::sample::sample;
use crate::{CompressorStats, GenerateStatsOptions};

#[derive(Clone)]
pub struct DistinctValues<T> {
    pub values: HashMap<T, u32, FxBuildHasher>,
}

#[derive(Clone)]
pub enum ErasedDistinctValues {
    F16(DistinctValues<u16>),
    F32(DistinctValues<u32>),
    F64(DistinctValues<u64>),
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

impl_from_typed!(u16, ErasedDistinctValues::F16);
impl_from_typed!(u32, ErasedDistinctValues::F32);
impl_from_typed!(u64, ErasedDistinctValues::F64);

// We want to allow not rebuilding all of the stats every time.
#[derive(Clone)]
pub struct FloatStats {
    pub(super) src: PrimitiveArray,
    // cache for validity.false_count()
    pub(super) null_count: u32,
    // cache for validity.true_count()
    pub(super) value_count: u32,
    pub(super) average_run_length: u32,
    pub(super) distinct_values: ErasedDistinctValues,
    pub(super) distinct_values_count: u32,
}

trait ToBits {
    type Target: Eq + Hash;

    fn to_bits(self) -> Self::Target;
}

macro_rules! impl_to_bits {
    ($typ:ty, $uint:ty) => {
        impl ToBits for $typ {
            type Target = $uint;

            fn to_bits(self) -> $uint {
                <$typ>::to_bits(self)
            }
        }
    };
}

impl_to_bits!(f16, u16);
impl_to_bits!(f32, u32);
impl_to_bits!(f64, u64);

impl CompressorStats for FloatStats {
    type ArrayType = PrimitiveArray;

    fn generate_opts(input: &PrimitiveArray, opts: GenerateStatsOptions) -> Self {
        match input.ptype() {
            PType::F16 => typed_float_stats::<f16>(input, opts.count_distinct_values),
            PType::F32 => typed_float_stats::<f32>(input, opts.count_distinct_values),
            PType::F64 => typed_float_stats::<f64>(input, opts.count_distinct_values),
            _ => vortex_panic!("cannot generate FloatStats from ptype {}", input.ptype()),
        }
    }

    fn source(&self) -> &Self::ArrayType {
        &self.src
    }

    fn sample_opts(&self, sample_size: u16, sample_count: u16, opts: GenerateStatsOptions) -> Self {
        let sampled = sample(self.src.clone(), sample_size, sample_count)
            .to_primitive()
            .vortex_expect("primitive");

        Self::generate_opts(&sampled, opts)
    }
}

fn typed_float_stats<T: NativePType + Float + ToBits>(
    array: &PrimitiveArray,
    count_distinct_values: bool,
) -> FloatStats
where
    DistinctValues<T::Target>: Into<ErasedDistinctValues>,
{
    // Special case: empty array
    if array.is_empty() {
        return FloatStats {
            src: array.clone(),
            null_count: 0,
            value_count: 0,
            average_run_length: 0,
            distinct_values_count: 0,
            distinct_values: DistinctValues {
                values: HashMap::<T::Target, u32, FxBuildHasher>::with_hasher(FxBuildHasher),
            }
            .into(),
        };
    } else if array.all_invalid().vortex_expect("all_invalid") {
        return FloatStats {
            src: array.clone(),
            null_count: array.len().try_into().vortex_expect("null_count"),
            value_count: 0,
            average_run_length: 0,
            distinct_values_count: 0,
            distinct_values: DistinctValues {
                values: HashMap::<T::Target, u32, FxBuildHasher>::with_hasher(FxBuildHasher),
            }
            .into(),
        };
    }

    let validity = array.validity_mask().vortex_expect("logical_validity");
    let null_count = validity.false_count();
    let value_count = validity.true_count();
    let mut min = T::max_value();
    let mut max = T::min_value();
    // Keep a HashMap of T, then convert the keys into PValue afterward since value is
    // so much more efficient to hash and search for.
    let mut distinct_values = if count_distinct_values {
        HashMap::with_capacity_and_hasher(array.len() / 2, FxBuildHasher)
    } else {
        HashMap::with_hasher(FxBuildHasher)
    };

    let mut runs = 1;
    let head_idx = validity
        .first()
        .vortex_expect("All null masks have been handled before");
    let buff = array.buffer::<T>();
    let mut prev = buff[head_idx];

    let first_valid_buff = buff.slice(head_idx..array.len());
    match validity.boolean_buffer() {
        AllOr::All => {
            for value in first_valid_buff {
                min = min.min(value);
                max = max.max(value);

                if count_distinct_values {
                    *distinct_values.entry(value.to_bits()).or_insert(0) += 1;
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
                .zip_eq(v.slice(head_idx, array.len() - head_idx).iter())
            {
                if valid {
                    min = min.min(value);
                    max = max.max(value);

                    if count_distinct_values {
                        *distinct_values.entry(value.to_bits()).or_insert(0) += 1;
                    }

                    if value != prev {
                        prev = value;
                        runs += 1;
                    }
                }
            }
        }
    }

    let null_count = null_count
        .try_into()
        .vortex_expect("null_count must fit in u32");
    let value_count = value_count
        .try_into()
        .vortex_expect("null_count must fit in u32");
    let distinct_values_count = if count_distinct_values {
        distinct_values.len().try_into().vortex_unwrap()
    } else {
        u32::MAX
    };

    FloatStats {
        null_count,
        value_count,
        distinct_values_count,
        src: array.clone(),
        average_run_length: value_count / runs,
        distinct_values: DistinctValues {
            values: distinct_values,
        }
        .into(),
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_array::{IntoArray, ToCanonical};
    use vortex_buffer::buffer;

    use crate::CompressorStats;
    use crate::float::stats::FloatStats;

    #[test]
    fn test_float_stats() {
        let floats = buffer![0.0f32, 1.0f32, 2.0f32].into_array();
        let floats = floats.to_primitive().unwrap();

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
