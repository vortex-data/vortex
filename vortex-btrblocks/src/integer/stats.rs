use std::hash::Hash;

use arrow_buffer::BooleanBuffer;
use num_traits::PrimInt;
use rustc_hash::FxBuildHasher;
use vortex_array::aliases::hash_map::HashMap;
use vortex_array::arrays::{NativeValue, PrimitiveArray};
use vortex_array::stats::Stat;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{Array, ToCanonical};
use vortex_dtype::{NativePType, match_each_integer_ptype};
use vortex_error::{VortexError, VortexExpect, VortexUnwrap};
use vortex_mask::AllOr;
use vortex_scalar::{PValue, ScalarValue};

use crate::sample::sample;
use crate::{CompressorStats, GenerateStatsOptions};

#[derive(Clone, Debug)]
pub struct TypedStats<T> {
    pub min: T,
    pub max: T,
    pub top_value: T,
    pub top_count: u32,
    pub distinct_values: HashMap<NativeValue<T>, u32, FxBuildHasher>,
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
            ErasedStats::I8(x) => (x.max as i16 - x.min as i16) as u64,
            ErasedStats::I16(x) => (x.max as i32 - x.min as i32) as u64,
            ErasedStats::I32(x) => (x.max as i64 - x.min as i64) as u64,
            ErasedStats::I64(x) => u64::try_from(x.max as i128 - x.min as i128)
                .vortex_expect("max minus min result bigger than u64"),
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

fn typed_int_stats<T>(array: &PrimitiveArray, count_distinct_values: bool) -> IntegerStats
where
    T: NativePType + PrimInt + for<'a> TryFrom<&'a ScalarValue, Error = VortexError>,
    TypedStats<T>: Into<ErasedStats>,
    NativeValue<T>: Eq + Hash,
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
    let head_idx = validity
        .first()
        .vortex_expect("All null masks have been handled before");
    let buffer = array.buffer::<T>();
    let head = buffer[head_idx];

    let mut loop_state = LoopState {
        distinct_values: if count_distinct_values {
            HashMap::with_capacity_and_hasher(array.len() / 2, FxBuildHasher)
        } else {
            HashMap::with_hasher(FxBuildHasher)
        },
        prev: head,
        runs: 1,
    };

    let sliced = buffer.slice(head_idx..array.len());
    let mut chunks = sliced.as_slice().array_chunks::<64>();
    match validity.boolean_buffer() {
        AllOr::All => {
            for chunk in &mut chunks {
                inner_loop_nonnull(chunk, count_distinct_values, &mut loop_state)
            }
            let remainder = chunks.remainder();
            inner_loop_naive(
                remainder,
                count_distinct_values,
                &BooleanBuffer::new_set(remainder.len()),
                &mut loop_state,
            );
        }
        AllOr::None => unreachable!("All invalid arrays have been handled before"),
        AllOr::Some(v) => {
            let mask = v.slice(head_idx, array.len() - head_idx);
            let mut offset = 0;
            for chunk in &mut chunks {
                let validity = mask.slice(offset, 64);
                offset += 64;

                match validity.count_set_bits() {
                    // All nulls -> no stats to update
                    0 => continue,
                    // Inner loop for when validity check can be elided
                    64 => inner_loop_nonnull(chunk, count_distinct_values, &mut loop_state),
                    // Inner loop for when we need to check validity
                    _ => inner_loop_nullable(
                        chunk,
                        count_distinct_values,
                        &validity,
                        &mut loop_state,
                    ),
                }
            }
            // Final iteration, run naive loop
            let remainder = chunks.remainder();
            inner_loop_naive(
                remainder,
                count_distinct_values,
                &mask.slice(offset, remainder.len()),
                &mut loop_state,
            );
        }
    }

    let (top_value, top_count) = if count_distinct_values {
        let (&top_value, &top_count) = loop_state
            .distinct_values
            .iter()
            .max_by_key(|&(_, &count)| count)
            .vortex_expect("non-empty");
        (top_value.0, top_count)
    } else {
        (T::default(), 0)
    };

    let runs = loop_state.runs;
    let distinct_values_count = if count_distinct_values {
        loop_state.distinct_values.len().try_into().vortex_unwrap()
    } else {
        u32::MAX
    };

    let min = array
        .statistics()
        .compute_as::<T>(Stat::Min)
        .vortex_expect("min should be computed");

    let max = array
        .statistics()
        .compute_as::<T>(Stat::Max)
        .vortex_expect("max should be computed");

    let typed = TypedStats {
        min,
        max,
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
    prev: T,
    runs: u32,
    distinct_values: HashMap<NativeValue<T>, u32, FxBuildHasher>,
}

#[inline(always)]
fn inner_loop_nonnull<T: NativePType>(
    values: &[T; 64],
    count_distinct_values: bool,
    state: &mut LoopState<T>,
) where
    NativeValue<T>: Eq + Hash,
{
    for &value in values {
        if count_distinct_values {
            *state.distinct_values.entry(NativeValue(value)).or_insert(0) += 1;
        }

        if value != state.prev {
            state.prev = value;
            state.runs += 1;
        }
    }
}

#[inline(always)]
fn inner_loop_nullable<T: NativePType>(
    values: &[T; 64],
    count_distinct_values: bool,
    is_valid: &BooleanBuffer,
    state: &mut LoopState<T>,
) where
    NativeValue<T>: Eq + Hash,
{
    for (idx, &value) in values.iter().enumerate() {
        if is_valid.value(idx) {
            if count_distinct_values {
                *state.distinct_values.entry(NativeValue(value)).or_insert(0) += 1;
            }

            if value != state.prev {
                state.prev = value;
                state.runs += 1;
            }
        }
    }
}

#[inline(always)]
fn inner_loop_naive<T: NativePType>(
    values: &[T],
    count_distinct_values: bool,
    is_valid: &BooleanBuffer,
    state: &mut LoopState<T>,
) where
    NativeValue<T>: Eq + Hash,
{
    for (idx, &value) in values.iter().enumerate() {
        if is_valid.value(idx) {
            if count_distinct_values {
                *state.distinct_values.entry(NativeValue(value)).or_insert(0) += 1;
            }

            if value != state.prev {
                state.prev = value;
                state.runs += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::iter;

    use arrow_buffer::BooleanBuffer;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::{Buffer, buffer};

    use crate::CompressorStats;
    use crate::integer::IntegerStats;
    use crate::integer::stats::typed_int_stats;

    #[test]
    fn test_naive_count_distinct_values() {
        let array = PrimitiveArray::new(buffer![217u8, 0], Validity::NonNullable);
        let stats = typed_int_stats::<u8>(&array, true);
        assert_eq!(stats.distinct_values_count, 2);
    }

    #[test]
    fn test_naive_count_distinct_values_nullable() {
        let array = PrimitiveArray::new(
            buffer![217u8, 0],
            Validity::from(BooleanBuffer::from(vec![true, false])),
        );
        let stats = typed_int_stats::<u8>(&array, true);
        assert_eq!(stats.distinct_values_count, 1);
    }

    #[test]
    fn test_count_distinct_values() {
        let array = PrimitiveArray::new((0..128u8).collect::<Buffer<u8>>(), Validity::NonNullable);
        let stats = typed_int_stats::<u8>(&array, true);
        assert_eq!(stats.distinct_values_count, 128);
    }

    #[test]
    fn test_count_distinct_values_nullable() {
        let array = PrimitiveArray::new(
            (0..128u8).collect::<Buffer<u8>>(),
            Validity::from(BooleanBuffer::from_iter(
                iter::repeat_n(vec![true, false], 64).flatten(),
            )),
        );
        let stats = typed_int_stats::<u8>(&array, true);
        assert_eq!(stats.distinct_values_count, 64);
    }

    #[test]
    fn test_integer_stats_leading_nulls() {
        let ints = PrimitiveArray::new(buffer![0, 1, 2], Validity::from_iter([false, true, true]));

        let stats = IntegerStats::generate(&ints);

        assert_eq!(stats.value_count, 2);
        assert_eq!(stats.null_count, 1);
        assert_eq!(stats.average_run_length, 1);
        assert_eq!(stats.distinct_values_count, 2);
    }
}
