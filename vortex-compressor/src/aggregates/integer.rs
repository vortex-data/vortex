// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Exact integer frequencies for compression decisions and dictionary preparation.
//!
//! Partial states retain native keys and exclude nulls. Dense counters and run accumulation keep
//! the serial dictionary order. Scalar conversion is only needed for partial exchange.

use std::hash::Hash;
use std::sync::Arc;

use rustc_hash::FxBuildHasher;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::Columnar;
use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::AggregateArgs;
use vortex_array::aggregate_fn::AggregateDTypes;
use vortex_array::aggregate_fn::AggregateFnId;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::min_max::min_max;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::NativeValue;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::IntegerPType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::expr::list_length;
use vortex_array::expr::root;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::PValue;
use vortex_array::scalar::Scalar;
use vortex_buffer::BitBuffer;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::AllOr;
use vortex_session::registry::CachedId;
use vortex_utils::aliases::hash_map::HashMap;

use super::runs::RunSummary;
use super::runs::RunSummaryPartial;

/// Exact non-null integer frequencies, with cardinality as the final result.
#[derive(Clone, Debug)]
pub struct IntegerFrequencies;

/// Native integer frequencies and the run summary collected in the same scan.
#[derive(Clone, Debug)]
pub struct IntegerDistribution<T> {
    /// Native valid values and positive occurrence counts.
    frequencies: HashMap<NativeValue<T>, u32, FxBuildHasher>,
    /// Ordered run counts and endpoints from the same valid values.
    runs: RunSummaryPartial,
    /// Cached mode using frequency-map iteration order to break ties.
    mode: Option<(T, u32)>,
}

impl<T: NativePType> IntegerDistribution<T>
where
    NativeValue<T>: Eq + Hash,
{
    /// Cache the mode without changing map reservation or iteration order.
    fn new(
        frequencies: HashMap<NativeValue<T>, u32, FxBuildHasher>,
        runs: RunSummaryPartial,
    ) -> Self {
        let mode = frequencies
            .iter()
            .max_by_key(|&(_, &count)| count)
            .map(|(value, &count)| (value.0, count));

        Self {
            frequencies,
            runs,
            mode,
        }
    }

    /// Returns native integer values and their valid occurrence counts.
    pub fn frequencies(&self) -> &HashMap<NativeValue<T>, u32, FxBuildHasher> {
        &self.frequencies
    }

    /// Returns the ordered valid-value run summary collected with the frequencies.
    pub fn run_summary(&self) -> &RunSummaryPartial {
        &self.runs
    }
}

/// Integer distributions retained for dictionary preparation and run estimates.
#[derive(Clone, Debug)]
pub enum IntegerFrequenciesPartial {
    /// Frequencies and runs of `u8` values.
    U8(IntegerDistribution<u8>),
    /// Frequencies and runs of `u16` values.
    U16(IntegerDistribution<u16>),
    /// Frequencies and runs of `u32` values.
    U32(IntegerDistribution<u32>),
    /// Frequencies and runs of `u64` values.
    U64(IntegerDistribution<u64>),
    /// Frequencies and runs of `i8` values.
    I8(IntegerDistribution<i8>),
    /// Frequencies and runs of `i16` values.
    I16(IntegerDistribution<i16>),
    /// Frequencies and runs of `i32` values.
    I32(IntegerDistribution<i32>),
    /// Frequencies and runs of `i64` values.
    I64(IntegerDistribution<i64>),
}

/// Visit the native distribution variant.
macro_rules! with_frequencies {
    ($partial:expr, | $values:ident | $body:expr) => {
        match $partial {
            IntegerFrequenciesPartial::U8($values) => $body,
            IntegerFrequenciesPartial::U16($values) => $body,
            IntegerFrequenciesPartial::U32($values) => $body,
            IntegerFrequenciesPartial::U64($values) => $body,
            IntegerFrequenciesPartial::I8($values) => $body,
            IntegerFrequenciesPartial::I16($values) => $body,
            IntegerFrequenciesPartial::I32($values) => $body,
            IntegerFrequenciesPartial::I64($values) => $body,
        }
    };
}

impl IntegerFrequenciesPartial {
    /// Returns the ordered run summary collected during frequency accumulation.
    pub fn run_summary(&self) -> &RunSummaryPartial {
        with_frequencies!(self, |values| &values.runs)
    }

    /// Returns the number of distinct valid values.
    ///
    /// # Panics
    ///
    /// Panics if the frequency cardinality exceeds the supported `u32` range.
    pub fn distinct_count(&self) -> u32 {
        with_frequencies!(self, |values| {
            u32::try_from(values.frequencies.len())
                .vortex_expect("frequency cardinality fits in u32")
        })
    }

    /// Returns the mode and its count, using map iteration to resolve frequency ties.
    ///
    /// Empty and all-null inputs have no mode.
    pub fn most_frequent_value_and_count(&self) -> Option<(PValue, u32)> {
        with_frequencies!(self, |values| {
            values.mode.map(|(value, count)| (value.into(), count))
        })
    }
}

/// Erase each native integer distribution without rebuilding its map.
macro_rules! impl_from_frequencies {
    ($($variant:ident: $T:ty),+ $(,)?) => {
        $(impl From<IntegerDistribution<$T>> for IntegerFrequenciesPartial {
            fn from(values: IntegerDistribution<$T>) -> Self {
                Self::$variant(values)
            }
        })+
    };
}

impl_from_frequencies!(U8: u8, U16: u16, U32: u32, U64: u64, I8: i8, I16: i16, I32: i32, I64: i64);

impl AggregateFnVTable for IntegerFrequencies {
    type Options = EmptyOptions;
    type Partial = IntegerFrequenciesPartial;

    fn id(&self) -> AggregateFnId {
        static ID: CachedId = CachedId::new("vortex.compressor.integer_frequencies");
        *ID
    }

    fn return_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        input_dtype
            .is_int()
            .then(|| DType::Primitive(PType::U64, Nullability::NonNullable))
    }

    fn partial_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        if !input_dtype.is_int() {
            return None;
        }

        let entry_dtype = DType::Struct(
            StructFields::from_iter([
                ("value", input_dtype.as_nonnullable()),
                (
                    "count",
                    DType::Primitive(PType::U32, Nullability::NonNullable),
                ),
            ]),
            Nullability::NonNullable,
        );
        Some(DType::Struct(
            StructFields::from_iter([
                (
                    "frequencies",
                    DType::List(Arc::new(entry_dtype), Nullability::NonNullable),
                ),
                (
                    "runs",
                    RunSummary.partial_dtype(&EmptyOptions, input_dtype)?,
                ),
            ]),
            Nullability::NonNullable,
        ))
    }

    fn empty_partial(&self, args: AggregateArgs<'_, Self::Options>) -> VortexResult<Self::Partial> {
        match_each_integer_ptype!(args.dtype.as_ptype(), |T| {
            Ok(IntegerDistribution::new(
                HashMap::<NativeValue<T>, u32, FxBuildHasher>::with_capacity_and_hasher(
                    0,
                    FxBuildHasher,
                ),
                RunSummaryPartial::default(),
            )
            .into())
        })
    }

    fn partial_from_scalar(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        scalar: Scalar,
    ) -> VortexResult<Self::Partial> {
        if scalar.dtype() != args.partial_dtype {
            vortex_bail!(
                "Expected integer frequency partial dtype {}, got {}",
                args.partial_dtype,
                scalar.dtype()
            );
        }

        let fields = scalar.as_struct();
        let frequencies = fields
            .field_by_idx(0)
            .vortex_expect("distribution has frequencies");
        let runs = fields
            .field_by_idx(1)
            .vortex_expect("distribution has a run summary");
        let run_dtypes = AggregateDTypes::try_new(&RunSummary, &EmptyOptions, args.dtype.clone())?;
        let runs = RunSummary.partial_from_scalar(run_dtypes.args(&EmptyOptions), runs)?;

        match_each_integer_ptype!(args.dtype.as_ptype(), |T| {
            let frequencies = parse_frequencies::<T>(&frequencies)?;
            let total: u64 = frequencies.values().map(|&count| u64::from(count)).sum();
            if total != runs.valid_count() {
                vortex_bail!(
                    "Distribution frequencies must match run summary valid count {}, got {total}",
                    runs.valid_count()
                );
            }
            Ok(IntegerDistribution::new(frequencies, runs).into())
        })
    }

    fn merge_partials(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        first: Self::Partial,
        second: Self::Partial,
    ) -> VortexResult<Self::Partial> {
        /// Merge matching native distribution variants.
        macro_rules! merge {
            ($($variant:ident),+) => {
                match (first, second) {
                    $((IntegerFrequenciesPartial::$variant(first), IntegerFrequenciesPartial::$variant(second)) => {
                        Ok(merge_distribution(first, second)?.into())
                    })+
                    _ => vortex_bail!("Integer frequency partials must have matching primitive types"),
                }
            };
        }

        merge!(U8, U16, U32, U64, I8, I16, I32, I64)
    }

    fn to_scalar(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> VortexResult<Scalar> {
        let fields = args.partial_dtype.as_struct_fields();
        let frequencies_dtype = fields
            .field_by_index(0)
            .vortex_expect("distribution has frequencies");
        let element_dtype = frequencies_dtype
            .as_list_element_opt()
            .vortex_expect("frequencies are a list");
        let entries = with_frequencies!(partial, |values| {
            values
                .frequencies
                .iter()
                .map(|(value, &count)| {
                    Scalar::struct_(
                        element_dtype.as_ref().clone(),
                        vec![
                            Scalar::primitive(value.0, Nullability::NonNullable),
                            Scalar::primitive(count, Nullability::NonNullable),
                        ],
                    )
                })
                .collect()
        });
        let frequencies =
            Scalar::list(Arc::clone(element_dtype), entries, Nullability::NonNullable);
        let run_dtypes = AggregateDTypes::try_new(&RunSummary, &EmptyOptions, args.dtype.clone())?;
        let runs = RunSummary.to_scalar(run_dtypes.args(&EmptyOptions), partial.run_summary())?;

        Ok(Scalar::struct_(
            args.partial_dtype.clone(),
            [frequencies, runs],
        ))
    }

    fn is_saturated(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        _partial: &Self::Partial,
    ) -> bool {
        false
    }

    fn accumulate(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        state: &mut Self::Partial,
        batch: &Columnar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let next: Self::Partial = match batch {
            Columnar::Canonical(Canonical::Primitive(array)) => {
                match_each_integer_ptype!(array.ptype(), |T| {
                    typed_frequencies::<T>(array, ctx)?.into()
                })
            }
            Columnar::Constant(array) => {
                if array.is_empty() || array.scalar().is_null() {
                    return Ok(());
                }

                match_each_integer_ptype!(args.dtype.as_ptype(), |T| {
                    let mut values = HashMap::with_capacity_and_hasher(1, FxBuildHasher);
                    let value = T::try_from(array.scalar())?;
                    let count = u32::try_from(array.len())?;
                    values.insert(NativeValue(value), count);
                    let runs = RunSummaryPartial::from_parts(
                        u64::from(count),
                        0,
                        Some(value.into()),
                        Some(value.into()),
                    )?;
                    IntegerDistribution::new(values, runs).into()
                })
            }
            _ => vortex_bail!(
                "Integer frequencies requires primitive integer input, got {}",
                batch.dtype()
            ),
        };

        let previous = std::mem::replace(state, self.empty_partial(args)?);
        *state = self.merge_partials(args, previous, next)?;
        Ok(())
    }

    fn finalize(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        states: ArrayRef,
    ) -> VortexResult<ArrayRef> {
        states.get_item("frequencies")?.apply(&list_length(root()))
    }

    fn finalize_scalar(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> VortexResult<Scalar> {
        Ok(Scalar::primitive(
            u64::from(partial.distinct_count()),
            Nullability::NonNullable,
        ))
    }
}

/// Parse unique, positive native frequencies with a supported total count.
fn parse_frequencies<T>(
    scalar: &Scalar,
) -> VortexResult<HashMap<NativeValue<T>, u32, FxBuildHasher>>
where
    T: IntegerPType + for<'a> TryFrom<&'a Scalar, Error = VortexError>,
    NativeValue<T>: Eq + Hash,
{
    let entries = scalar
        .as_list()
        .elements()
        .vortex_expect("frequency partial is non-nullable");
    let mut values = HashMap::with_capacity_and_hasher(entries.len(), FxBuildHasher);
    let mut total = 0u32;
    for entry in entries {
        let fields = entry.as_struct();
        let value = fields
            .field_by_idx(0)
            .vortex_expect("frequency entries have a value field");
        let count = fields
            .field_by_idx(1)
            .vortex_expect("frequency entries have a count field");
        let count = u32::try_from(&count)?;
        if count == 0 {
            vortex_bail!("Integer frequency counts must be positive, got 0");
        }

        total = total
            .checked_add(count)
            .ok_or_else(|| vortex_err!("Integer frequency total exceeds u32::MAX"))?;
        if values
            .insert(NativeValue(T::try_from(&value)?), count)
            .is_some()
        {
            vortex_bail!("Integer frequency entries must have distinct values");
        }
    }

    Ok(values)
}

/// Merge frequency counts and the ordered run boundary.
fn merge_distribution<T>(
    first: IntegerDistribution<T>,
    second: IntegerDistribution<T>,
) -> VortexResult<IntegerDistribution<T>>
where
    T: NativePType,
    NativeValue<T>: Eq + Hash,
{
    if first.frequencies.is_empty() {
        return Ok(second);
    }
    if second.frequencies.is_empty() {
        return Ok(first);
    }

    let runs = first.runs.merge(second.runs)?;
    u32::try_from(runs.valid_count())?;
    let mut frequencies = first.frequencies;
    for (value, count) in second.frequencies {
        *frequencies.entry(value).or_insert(0) += count;
    }

    Ok(IntegerDistribution::new(frequencies, runs))
}

/// Collect frequencies and ordered runs in the tuned native loop.
fn typed_frequencies<T>(
    array: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<IntegerDistribution<T>>
where
    T: IntegerPType + Into<PValue> + for<'a> TryFrom<&'a Scalar, Error = VortexError>,
    NativeValue<T>: Eq + Hash,
{
    let validity = array.validity()?.execute_mask(array.len(), ctx)?;
    u32::try_from(validity.true_count())?;
    u32::try_from(validity.false_count())?;
    let Some(head_idx) = validity.first() else {
        return Ok(IntegerDistribution::new(
            HashMap::with_capacity_and_hasher(0, FxBuildHasher),
            RunSummaryPartial::default(),
        ));
    };

    let extrema = min_max(array.as_ref(), ctx, NumericalAggregateOpts::default())?
        .vortex_expect("a valid integer input has extrema");
    let min = T::try_from(&extrema.min)?;
    let max = T::try_from(&extrema.max)?;
    let buffer = array.to_buffer::<T>();
    let mut state = FrequencyLoop {
        prev: buffer[head_idx],
        pending: 0,
        transitions: 0,
        distinct: DistinctCounter::new(min, max, array.len()),
    };
    let sliced = buffer.slice(head_idx..array.len());
    let (chunks, remainder) = sliced.as_slice().as_chunks::<64>();

    match validity.bit_buffer() {
        AllOr::All => {
            for chunk in chunks {
                inner_loop_nonnull(chunk, &mut state);
            }
            inner_loop_masked(remainder, &BitBuffer::new_set(remainder.len()), &mut state);
        }
        AllOr::None => unreachable!("the input has a valid value"),
        AllOr::Some(valid) => {
            let mask = valid.slice(head_idx..array.len());
            let mut offset = 0;
            for chunk in chunks {
                let validity = mask.slice(offset..offset + 64);
                offset += 64;
                match validity.true_count() {
                    0 => continue,
                    64 => inner_loop_nonnull(chunk, &mut state),
                    _ => inner_loop_masked(chunk, &validity, &mut state),
                }
            }
            inner_loop_masked(
                remainder,
                &mask.slice(offset..offset + remainder.len()),
                &mut state,
            );
        }
    }

    state.flush();
    let runs = RunSummaryPartial::from_parts(
        validity.true_count() as u64,
        state.transitions,
        Some(buffer[head_idx].into()),
        Some(state.prev.into()),
    )?;
    Ok(IntegerDistribution::new(state.distinct.finish(), runs))
}

/// Largest value range eligible for dense counting when it is no larger than the input.
const DENSE_DISTINCT_MAX_RANGE: usize = 1 << 16;
/// Largest value range eligible for dense counting regardless of the input length.
const DENSE_DISTINCT_ALWAYS_RANGE: usize = 1 << 8;

/// Chooses dense counts for narrow ranges and a native hash map for wider ranges.
enum DistinctCounter<T> {
    /// Counts indexed by the valid value's distance from the minimum.
    Dense {
        /// Minimum valid value used to reconstruct native map keys.
        min: T,
        /// Minimum cast modulo `2^usize::BITS`, giving exact wrapping differences in this range.
        min_index: usize,
        /// Occurrences of `min + index`, including zero for absent values.
        counts: Vec<u32>,
    },
    /// Native hash counts for values spanning a wider range.
    Hashed(HashMap<NativeValue<T>, u32, FxBuildHasher>),
}

impl<T: IntegerPType> DistinctCounter<T>
where
    NativeValue<T>: Eq + Hash,
{
    /// Preserve the compressor's dense-range thresholds and hashed reservation.
    fn new(min: T, max: T, len: usize) -> Self {
        let range_len = match (min.to_i128(), max.to_i128()) {
            (Some(min), Some(max)) => usize::try_from(max - min + 1).unwrap_or(usize::MAX),
            _ => usize::MAX,
        };

        if range_len <= DENSE_DISTINCT_ALWAYS_RANGE
            || (range_len <= DENSE_DISTINCT_MAX_RANGE && range_len <= len)
        {
            Self::Dense {
                min,
                min_index: min.as_(),
                counts: vec![0; range_len],
            }
        } else {
            Self::Hashed(HashMap::with_capacity_and_hasher(len / 2, FxBuildHasher))
        }
    }

    /// Add the occurrences of one complete valid-value run.
    // Preserve the existing inlined counter inside the tuned run-based loops.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn add(&mut self, value: T, count: u32) {
        match self {
            Self::Dense {
                min_index, counts, ..
            } => counts[value.as_().wrapping_sub(*min_index)] += count,
            Self::Hashed(values) => *values.entry(NativeValue(value)).or_insert(0) += count,
        }
    }

    /// Build dense keys in ascending order or move the hashed map directly.
    fn finish(self) -> HashMap<NativeValue<T>, u32, FxBuildHasher> {
        match self {
            Self::Dense { min, counts, .. } => {
                let min = min.to_i128().vortex_expect("integers fit in i128");
                counts
                    .iter()
                    .enumerate()
                    .filter(|&(_, &count)| count > 0)
                    .map(|(index, &count)| {
                        let value = <T as num_traits::NumCast>::from(min + index as i128)
                            .vortex_expect("values between min and max fit in the type");
                        (NativeValue(value), count)
                    })
                    .collect()
            }
            Self::Hashed(values) => values,
        }
    }
}

/// Defers a map update until the current valid-value run ends.
struct FrequencyLoop<T> {
    /// Last valid value, including values whose occurrences remain pending.
    prev: T,
    /// Occurrences of the current valid-value run awaiting a counter update.
    pending: u32,
    /// Inequality transitions between adjacent valid values.
    transitions: u64,
    /// Dense or hashed counts already flushed from earlier runs.
    distinct: DistinctCounter<T>,
}

impl<T: IntegerPType> FrequencyLoop<T>
where
    NativeValue<T>: Eq + Hash,
{
    /// Record the current run without changing its endpoint.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn flush(&mut self) {
        self.distinct.add(self.prev, self.pending);
        self.pending = 0;
    }

    /// Record one valid value, retaining null-skipping run boundaries.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn push(&mut self, value: T) {
        if value != self.prev {
            self.flush();
            self.prev = value;
            self.transitions += 1;
        }
        self.pending += 1;
    }
}

/// Preserve the branch-free transition count and constant-chunk shortcut.
#[allow(clippy::inline_always)]
#[inline(always)]
fn inner_loop_nonnull<T: IntegerPType>(values: &[T; 64], state: &mut FrequencyLoop<T>)
where
    NativeValue<T>: Eq + Hash,
{
    let transitions = u8::from(values[0] != state.prev)
        + values
            .iter()
            .zip(&values[1..])
            .map(|(a, b)| u8::from(a != b))
            .sum::<u8>();

    if transitions == 0 {
        state.pending += 64;
        return;
    }

    for &value in values {
        if value != state.prev {
            state.flush();
            state.prev = value;
        }
        state.pending += 1;
    }
    state.transitions += u64::from(transitions);
}

/// Skip invalid slots without breaking runs between valid values.
#[allow(clippy::inline_always)]
#[inline(always)]
fn inner_loop_masked<T: IntegerPType>(
    values: &[T],
    is_valid: &BitBuffer,
    state: &mut FrequencyLoop<T>,
) where
    NativeValue<T>: Eq + Hash,
{
    for (index, &value) in values.iter().enumerate() {
        if is_valid.value(index) {
            state.push(value);
        }
    }
}
