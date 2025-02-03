use std::hash::Hash;

use num_traits::Float;
use rustc_hash::FxBuildHasher;
use vortex_array::aliases::hash_map::HashMap;
use vortex_array::array::PrimitiveArray;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::IntoArrayVariant;
use vortex_dtype::half::f16;
use vortex_dtype::{NativePType, PType};
use vortex_error::{vortex_panic, VortexExpect};

use crate::sample::sample;
use crate::CompressorStats;

// We want to allow not rebuilding all of the stats every time.
#[derive(Clone)]
pub struct FloatStats {
    pub(super) src: PrimitiveArray,
    // cache for validity.false_count()
    pub(super) null_count: u32,
    // cache for validity.true_count()
    pub(super) value_count: u32,
    pub(super) average_run_length: u32,

    pub(super) distinct_value_count: u32,
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

    fn generate(input: &Self::ArrayType) -> Self {
        match input.ptype() {
            PType::F16 => typed_float_stats::<f16>(input),
            PType::F32 => typed_float_stats::<f32>(input),
            PType::F64 => typed_float_stats::<f64>(input),
            _ => vortex_panic!("cannot generate FloatStats from ptype {}", input.ptype()),
        }
    }

    fn source(&self) -> &Self::ArrayType {
        &self.src
    }

    fn sample(&self, sample_size: u16, sample_count: u16) -> Self {
        let sampled = sample(self.src.clone(), sample_size, sample_count)
            .into_primitive()
            .vortex_expect("primitive");

        Self::generate(&sampled)
    }
}

fn typed_float_stats<T: NativePType + Float + ToBits>(array: &PrimitiveArray) -> FloatStats {
    let validity = array.validity_mask().vortex_expect("logical_validity");
    let null_count = validity.false_count();
    let value_count = validity.true_count();
    let mut min = T::max_value();
    let mut max = T::min_value();
    let mut distinct_values = HashMap::with_capacity_and_hasher(array.len() / 2, FxBuildHasher);

    // Keep a HashMap of T, then convert the keys into PValue afterward since value is
    // so much more efficient to hash and search for.
    let mut runs = 1;
    let mut prev = array.as_slice::<T>()[0];

    for (idx, &value) in array.buffer::<T>().iter().enumerate() {
        if validity.value(idx) {
            min = min.min(value);
            max = max.max(value);
            *distinct_values.entry(value.to_bits()).or_insert(0) += 1;

            if value != prev {
                prev = value;
                runs += 1;
            }
        }
    }

    let null_count = null_count
        .try_into()
        .vortex_expect("null_count must fit in u32");
    let value_count = value_count
        .try_into()
        .vortex_expect("null_count must fit in u32");
    let distinct_value_count: u32 = distinct_values
        .len()
        .try_into()
        .vortex_expect("distinct_value_count must fit in u32");

    FloatStats {
        src: array.clone(),
        null_count,
        value_count,
        average_run_length: value_count / runs,
        distinct_value_count,
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::{IntoArray, IntoArrayVariant};
    use vortex_buffer::buffer;

    use crate::float::stats::FloatStats;
    use crate::CompressorStats;

    #[test]
    fn test_float_stats() {
        let floats = buffer![0.0f32, 1.0f32, 2.0f32].into_array();
        let floats = floats.into_primitive().unwrap();

        let stats = FloatStats::generate(&floats);

        assert_eq!(stats.value_count, 3);
        assert_eq!(stats.null_count, 0);
        assert_eq!(stats.average_run_length, 1);
    }
}
