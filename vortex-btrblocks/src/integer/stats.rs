use std::hash::Hash;

use num_traits::PrimInt;
use rustc_hash::FxBuildHasher;
use vortex_array::aliases::hash_map::HashMap;
use vortex_array::array::PrimitiveArray;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::IntoArrayVariant;
use vortex_dtype::{match_each_integer_ptype, NativePType};
use vortex_error::VortexExpect;
use vortex_scalar::PValue;

use crate::sample::sample;
use crate::CompressorStats;

#[derive(Clone)]
pub struct TypedStats<T> {
    min: T,
    max: T,
    distinct_values: HashMap<T, u32, FxBuildHasher>,
}

/// Type-erased container for one of the [TypedStats] variants.
///
/// Building the `TypedStats` is considerably faster and cheaper than building a type-erased
/// set of stats. We then perform a variety of access methods on them.
#[derive(Clone)]
pub enum ErasedStats {
    U8(TypedStats<u8>),
    U16(TypedStats<u16>),
    U32(TypedStats<u32>),
    U64(TypedStats<u64>),
    I8(TypedStats<i8>),
    I16(TypedStats<i16>),
    I32(TypedStats<i32>),
    I64(TypedStats<i64>),
}

impl ErasedStats {
    pub fn min_is_zero(&self) -> bool {
        match &self {
            ErasedStats::U8(x) => x.min == 0,
            ErasedStats::U16(x) => x.min == 0,
            ErasedStats::U32(x) => x.min == 0,
            ErasedStats::U64(x) => x.min == 0,
            ErasedStats::I8(x) => x.min == 0,
            ErasedStats::I16(x) => x.min == 0,
            ErasedStats::I32(x) => x.min == 0,
            ErasedStats::I64(x) => x.min == 0,
        }
    }

    pub fn min_is_negative(&self) -> bool {
        match &self {
            ErasedStats::U8(_)
            | ErasedStats::U16(_)
            | ErasedStats::U32(_)
            | ErasedStats::U64(_) => false,
            ErasedStats::I8(x) => x.min < 0,
            ErasedStats::I16(x) => x.min < 0,
            ErasedStats::I32(x) => x.min < 0,
            ErasedStats::I64(x) => x.min < 0,
        }
    }

    pub fn distinct_value_count(&self) -> usize {
        match &self {
            ErasedStats::U8(x) => x.distinct_values.len(),
            ErasedStats::U16(x) => x.distinct_values.len(),
            ErasedStats::U32(x) => x.distinct_values.len(),
            ErasedStats::U64(x) => x.distinct_values.len(),
            ErasedStats::I8(x) => x.distinct_values.len(),
            ErasedStats::I16(x) => x.distinct_values.len(),
            ErasedStats::I32(x) => x.distinct_values.len(),
            ErasedStats::I64(x) => x.distinct_values.len(),
        }
    }

    // Difference between max and min.
    pub fn max_minus_min(&self) -> u64 {
        match &self {
            ErasedStats::U8(x) => (x.max - x.min) as u64,
            ErasedStats::U16(x) => (x.max - x.min) as u64,
            ErasedStats::U32(x) => (x.max - x.min) as u64,
            ErasedStats::U64(x) => x.max - x.min,
            ErasedStats::I8(x) => (x.max - x.min) as u64,
            ErasedStats::I16(x) => (x.max - x.min) as u64,
            ErasedStats::I32(x) => (x.max - x.min) as u64,
            ErasedStats::I64(x) => (x.max - x.min) as u64,
        }
    }

    /// Get the most commonly occurring value and its count
    pub fn top_value(&self) -> (PValue, u32) {
        match &self {
            ErasedStats::U8(x) => extract_top_value(&x.distinct_values),
            ErasedStats::U16(x) => extract_top_value(&x.distinct_values),
            ErasedStats::U32(x) => extract_top_value(&x.distinct_values),
            ErasedStats::U64(x) => extract_top_value(&x.distinct_values),
            ErasedStats::I8(x) => extract_top_value(&x.distinct_values),
            ErasedStats::I16(x) => extract_top_value(&x.distinct_values),
            ErasedStats::I32(x) => extract_top_value(&x.distinct_values),
            ErasedStats::I64(x) => extract_top_value(&x.distinct_values),
        }
    }
}

fn extract_top_value<T: Copy + Into<PValue>, S>(values: &HashMap<T, u32, S>) -> (PValue, u32) {
    let (&top_value, &top_count) = values
        .iter()
        .max_by_key(|(_, &count)| count)
        .vortex_expect("non-empty");

    (top_value.into(), top_count)
}

macro_rules! impl_from_typed {
    ($T:ty, $variant:path) => {
        impl From<TypedStats<$T>> for ErasedStats {
            fn from(typed: TypedStats<$T>) -> Self {
                $variant(typed)
            }
        }
    };
}

impl_from_typed!(u8, ErasedStats::U8);
impl_from_typed!(u16, ErasedStats::U16);
impl_from_typed!(u32, ErasedStats::U32);
impl_from_typed!(u64, ErasedStats::U64);
impl_from_typed!(i8, ErasedStats::I8);
impl_from_typed!(i16, ErasedStats::I16);
impl_from_typed!(i32, ErasedStats::I32);
impl_from_typed!(i64, ErasedStats::I64);

#[derive(Clone)]
pub struct IntegerStats {
    pub(super) src: PrimitiveArray,
    // cache for validity.false_count()
    pub(super) null_count: u32,
    // cache for validity.true_count()
    pub(super) value_count: u32,
    pub(super) average_run_length: u32,

    pub(super) typed: ErasedStats,
}

impl CompressorStats for IntegerStats {
    type ArrayType = PrimitiveArray;

    fn generate(input: &PrimitiveArray) -> IntegerStats {
        match_each_integer_ptype!(input.ptype(), |$T| {
            typed_int_stats::<$T>(input)
        })
    }

    fn source(&self) -> &PrimitiveArray {
        &self.src
    }

    fn sample(&self, sample_size: u16, sample_count: u16) -> IntegerStats {
        let sampled = sample(self.src.clone(), sample_size, sample_count)
            .into_primitive()
            .vortex_expect("primitive");

        Self::generate(&sampled)
    }
}

fn typed_int_stats<T: NativePType + Hash + PrimInt>(array: &PrimitiveArray) -> IntegerStats
where
    TypedStats<T>: Into<ErasedStats>,
{
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
    // Do nulls count for runs

    for (idx, &value) in array.buffer::<T>().iter().enumerate() {
        if validity.value(idx) {
            min = min.min(value);
            max = max.max(value);
            *distinct_values.entry(value).or_insert(0) += 1;

            if value != prev {
                prev = value;
                runs += 1;
            }
        }
    }

    let typed = TypedStats {
        min,
        max,
        distinct_values,
    };

    let null_count = null_count
        .try_into()
        .vortex_expect("null_count must fit in u32");
    let value_count = value_count
        .try_into()
        .vortex_expect("value_count must fit in u32");

    IntegerStats {
        src: array.clone(),
        null_count,
        value_count,
        average_run_length: value_count / runs,
        typed: typed.into(),
    }
}
