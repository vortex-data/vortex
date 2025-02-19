use std::hash::Hash;

use arrow_buffer::BooleanBuffer;
use num_traits::PrimInt;
use rustc_hash::FxBuildHasher;
use vortex_array::aliases::hash_map::HashMap;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{Array, ToCanonical};
use vortex_dtype::{match_each_integer_ptype, NativePType};
use vortex_error::{VortexExpect, VortexUnwrap};
use vortex_scalar::PValue;

use crate::sample::sample;
use crate::{CompressorStats, GenerateStatsOptions};

#[derive(Clone, Debug)]
pub struct TypedStats<T> {
    pub min: T,
    pub max: T,
    pub top_value: T,
    pub top_count: u32,
    pub distinct_values: HashMap<T, u32, FxBuildHasher>,
}

/// Type-erased container for one of the [TypedStats] variants.
///
/// Building the `TypedStats` is considerably faster and cheaper than building a type-erased
/// set of stats. We then perform a variety of access methods on them.
#[derive(Clone, Debug)]
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
    pub fn top_value_and_count(&self) -> (PValue, u32) {
        match &self {
            ErasedStats::U8(x) => (x.top_value.into(), x.top_count),
            ErasedStats::U16(x) => (x.top_value.into(), x.top_count),
            ErasedStats::U32(x) => (x.top_value.into(), x.top_count),
            ErasedStats::U64(x) => (x.top_value.into(), x.top_count),
            ErasedStats::I8(x) => (x.top_value.into(), x.top_count),
            ErasedStats::I16(x) => (x.top_value.into(), x.top_count),
            ErasedStats::I32(x) => (x.top_value.into(), x.top_count),
            ErasedStats::I64(x) => (x.top_value.into(), x.top_count),
        }
    }
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

#[derive(Clone, Debug)]
pub struct IntegerStats {
    pub(super) src: PrimitiveArray,
    // cache for validity.false_count()
    pub(super) null_count: u32,
    // cache for validity.true_count()
    pub(super) value_count: u32,
    pub(super) average_run_length: u32,
    pub(super) distinct_values_count: u32,
    pub(crate) typed: ErasedStats,
}

impl CompressorStats for IntegerStats {
    type ArrayType = PrimitiveArray;

    fn generate_opts(input: &PrimitiveArray, opts: GenerateStatsOptions) -> Self {
        match_each_integer_ptype!(input.ptype(), |$T| {
            typed_int_stats::<$T>(input, opts.count_distinct_values)
        })
    }

    fn source(&self) -> &PrimitiveArray {
        &self.src
    }

    fn sample_opts(&self, sample_size: u16, sample_count: u16, opts: GenerateStatsOptions) -> Self {
        let sampled = sample(self.src.clone(), sample_size, sample_count)
            .to_primitive()
            .vortex_expect("primitive");

        Self::generate_opts(&sampled, opts)
    }
}

fn typed_int_stats<T: NativePType + Hash + PrimInt>(
    array: &PrimitiveArray,
    count_distinct_values: bool,
) -> IntegerStats
where
    TypedStats<T>: Into<ErasedStats>,
{
    // Special case: empty array
    if array.is_empty() {
        return IntegerStats {
            src: array.clone(),
            null_count: 0,
            value_count: 0,
            average_run_length: 0,
            distinct_values_count: 0,
            typed: TypedStats {
                min: T::max_value(),
                max: T::min_value(),
                top_value: T::default(),
                top_count: 0,
                distinct_values: HashMap::with_hasher(FxBuildHasher),
            }
            .into(),
        };
    } else if array.all_invalid().vortex_expect("all_invalid") {
        return IntegerStats {
            src: array.clone(),
            null_count: array.len().try_into().vortex_expect("null_count"),
            value_count: 0,
            average_run_length: 0,
            distinct_values_count: 0,
            typed: TypedStats {
                min: T::max_value(),
                max: T::min_value(),
                top_value: T::default(),
                top_count: 0,
                distinct_values: HashMap::with_hasher(FxBuildHasher),
            }
            .into(),
        };
    }

    let validity = array.validity_mask().vortex_expect("logical_validity");
    let null_count = validity.false_count();
    let value_count = validity.true_count();

    // Initialize loop state
    let head = array.as_slice::<T>()[0];
    let mut loop_state = LoopState {
        min: head,
        max: head,
        distinct_values: if count_distinct_values {
            HashMap::with_capacity_and_hasher(array.len() / 2, FxBuildHasher)
        } else {
            HashMap::with_hasher(FxBuildHasher)
        },
        distinct_values_count: if count_distinct_values { 0 } else { u32::MAX },
        prev: head,
        runs: 1,
    };

    let values = array.buffer::<T>();
    let mask = validity.to_boolean_buffer();

    let mut offset = 0;
    for chunk in values.as_slice().chunks(64) {
        let validity = mask.slice(offset, chunk.len());
        offset += chunk.len();

        if chunk.len() < 64 {
            // Final iteration, run naive loop
            inner_loop_naive(chunk, count_distinct_values, &validity, &mut loop_state);
            break;
        }

        let set_bits = validity.count_set_bits();

        match set_bits {
            // All nulls -> no stats to update
            0 => continue,
            // Inner loop for when validity check can be elided
            64 => inner_loop_nonnull(
                chunk.try_into().vortex_unwrap(),
                count_distinct_values,
                &mut loop_state,
            ),
            // Inner loop for when we need to check validity
            _ => inner_loop_nullable(
                chunk.try_into().vortex_unwrap(),
                count_distinct_values,
                &validity,
                &mut loop_state,
            ),
        }
    }

    let (top_value, top_count) = if count_distinct_values {
        let (&top_value, &top_count) = loop_state
            .distinct_values
            .iter()
            .max_by_key(|(_, &count)| count)
            .vortex_expect("non-empty");
        (top_value, top_count)
    } else {
        (T::default(), 0)
    };

    let runs = loop_state.runs;
    let distinct_values_count = loop_state.distinct_values_count;

    let typed = TypedStats {
        min: loop_state.min,
        max: loop_state.max,
        distinct_values: loop_state.distinct_values,
        top_value,
        top_count,
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
        distinct_values_count,
        typed: typed.into(),
    }
}

struct LoopState<T> {
    min: T,
    max: T,
    prev: T,
    runs: u32,
    distinct_values_count: u32,
    distinct_values: HashMap<T, u32, FxBuildHasher>,
}

#[inline(always)]
fn inner_loop_nonnull<T: PrimInt + Hash>(
    values: &[T; 64],
    count_distinct_values: bool,
    state: &mut LoopState<T>,
) {
    for &value in values {
        state.min = state.min.min(value);
        state.max = state.max.max(value);

        if count_distinct_values {
            *state.distinct_values.entry(state.prev).or_insert(0) += 1;
            state.distinct_values_count = state.distinct_values.len().try_into().vortex_unwrap();
        }

        if value != state.prev {
            state.prev = value;
            state.runs += 1;
        }
    }
}

#[inline(always)]
fn inner_loop_nullable<T: PrimInt + Hash>(
    values: &[T; 64],
    count_distinct_values: bool,
    is_valid: &BooleanBuffer,
    state: &mut LoopState<T>,
) {
    for (idx, &value) in values.iter().enumerate() {
        if is_valid.value(idx) {
            state.min = state.min.min(value);
            state.max = state.max.max(value);

            if count_distinct_values {
                *state.distinct_values.entry(state.prev).or_insert(0) += 1;
                state.distinct_values_count =
                    state.distinct_values.len().try_into().vortex_unwrap();
            }

            if value != state.prev {
                state.prev = value;
                state.runs += 1;
            }
        }
    }
}

#[inline(always)]
fn inner_loop_naive<T: PrimInt + Hash>(
    values: &[T],
    count_distinct_values: bool,
    is_valid: &BooleanBuffer,
    state: &mut LoopState<T>,
) {
    for (idx, &value) in values.iter().enumerate() {
        if is_valid.value(idx) {
            state.min = state.min.min(value);
            state.max = state.max.max(value);

            if count_distinct_values {
                *state.distinct_values.entry(state.prev).or_insert(0) += 1;
                state.distinct_values_count =
                    state.distinct_values.len().try_into().vortex_unwrap();
            }

            if value != state.prev {
                state.prev = value;
                state.runs += 1;
            }
        }
    }
}
