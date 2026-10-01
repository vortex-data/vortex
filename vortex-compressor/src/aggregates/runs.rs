// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Ordered summaries of runs among valid numeric values.
//!
//! Nulls contribute no values and do not break a run. Numeric equality groups signed zeros together,
//! while every comparison with a NaN starts another run. The compressor's historical first-NaN
//! adjustment belongs to the average accessor, so it is applied only once after merging.

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::Columnar;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::aggregate_fn::AggregateArgs;
use vortex_array::aggregate_fn::AggregateFnId;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::IntegerPType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::dtype::Nullability::Nullable;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::match_each_integer_ptype;
use vortex_array::match_each_native_ptype;
use vortex_array::scalar::PValue;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_mask::AllOr;
use vortex_mask::Mask;
use vortex_session::registry::CachedId;

/// Count runs of numerically equal, adjacent valid primitive values.
///
/// Empty and all-null inputs return zero runs. The retained partial also carries the valid count
/// and both endpoints, allowing an ordered merge to account for the boundary between batches.
#[derive(Clone, Debug)]
pub struct RunSummary;

/// Mergeable counts and endpoints for valid numeric runs.
#[derive(Clone, Debug, Default)]
pub struct RunSummaryPartial {
    /// Number of valid values, including NaNs.
    valid_count: u64,
    /// Number of unequal adjacent valid pairs, always smaller than a nonzero valid count.
    transitions: u64,
    /// First valid native value, present exactly when the valid count is nonzero.
    first: Option<PValue>,
    /// Last valid native value, present exactly when the valid count is nonzero.
    last: Option<PValue>,
}

impl RunSummaryPartial {
    /// Build an ordered summary after a native distribution loop has scanned valid values.
    ///
    /// Endpoints must both be present exactly when the count is nonzero, use the same native type,
    /// and include NaNs. Transitions count numeric inequality between adjacent valid values, with
    /// no comparison before the first value.
    ///
    /// # Errors
    ///
    /// Returns an error if the endpoints or transition count do not describe a valid summary.
    pub(crate) fn from_parts(
        valid_count: u64,
        transitions: u64,
        first: Option<PValue>,
        last: Option<PValue>,
    ) -> VortexResult<Self> {
        vortex_ensure!(
            first.is_some() == (valid_count != 0) && last.is_some() == (valid_count != 0),
            "Run summary endpoints must be present exactly when valid_count is nonzero"
        );
        vortex_ensure!(
            first.map(|value| value.ptype()) == last.map(|value| value.ptype()),
            "Run summary endpoints must have the same native type"
        );
        vortex_ensure!(
            (valid_count == 0 && transitions == 0) || transitions < valid_count,
            "Run summary transitions must be less than valid_count for a nonempty input"
        );

        Ok(Self {
            valid_count,
            transitions,
            first,
            last,
        })
    }

    /// Number of valid values, including NaNs.
    pub fn valid_count(&self) -> u64 {
        self.valid_count
    }

    /// Number of runs under numeric equality, or zero when no valid values were seen.
    pub fn run_count(&self) -> u64 {
        if self.valid_count == 0 {
            0
        } else {
            self.transitions + 1
        }
    }

    /// Whether the complete input's first valid value is a NaN.
    pub fn first_is_nan(&self) -> bool {
        self.first.is_some_and(|value| value.is_nan())
    }

    /// The floored average used by existing compression heuristics.
    ///
    /// The previous float loop compared its first value with itself, adding one run for a leading
    /// NaN. Keep that adjustment here, after all partials have merged, to preserve encoding choices.
    ///
    /// # Errors
    ///
    /// Returns an error if the valid count exceeds the compressor's supported `u32` range.
    pub fn average_run_length_for_compression(&self) -> VortexResult<u32> {
        let valid_count = u32::try_from(self.valid_count)?;
        if valid_count == 0 {
            return Ok(0);
        }

        let runs = self.run_count() + u64::from(self.first_is_nan());
        Ok(u32::try_from(u64::from(valid_count) / runs)?)
    }

    /// Append a summary for values following this input.
    ///
    /// # Errors
    ///
    /// Returns an error if the combined valid count exceeds `u64::MAX`.
    pub(crate) fn merge(self, other: Self) -> VortexResult<Self> {
        let Some(first) = self.first else {
            return Ok(other);
        };
        let Some(other_first) = other.first else {
            return Ok(self);
        };

        let valid_count = self
            .valid_count
            .checked_add(other.valid_count)
            .ok_or_else(|| vortex_err!("Run summary valid count exceeds u64::MAX"))?;
        let boundary = self
            .last
            .is_some_and(|last| numeric_values_differ(last, other_first));
        let transitions = self
            .transitions
            .checked_add(other.transitions)
            .and_then(|count| count.checked_add(u64::from(boundary)))
            .ok_or_else(|| vortex_err!("Run summary transition count exceeds u64::MAX"))?;

        Ok(Self {
            valid_count,
            transitions,
            first: Some(first),
            last: other.last,
        })
    }
}

impl AggregateFnVTable for RunSummary {
    type Options = EmptyOptions;
    type Partial = RunSummaryPartial;

    fn id(&self) -> AggregateFnId {
        static ID: CachedId = CachedId::new("vortex.compressor.run_summary");
        *ID
    }

    fn return_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        input_dtype
            .is_primitive()
            .then_some(DType::Primitive(PType::U64, NonNullable))
    }

    fn partial_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        if !input_dtype.is_primitive() {
            return None;
        }

        Some(DType::Struct(
            StructFields::from_iter([
                ("valid_count", DType::Primitive(PType::U64, NonNullable)),
                ("transitions", DType::Primitive(PType::U64, NonNullable)),
                ("first", input_dtype.as_nullable()),
                ("last", input_dtype.as_nullable()),
            ]),
            NonNullable,
        ))
    }

    fn empty_partial(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
    ) -> VortexResult<Self::Partial> {
        Ok(RunSummaryPartial::default())
    }

    fn partial_from_scalar(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        scalar: Scalar,
    ) -> VortexResult<Self::Partial> {
        let fields = scalar.as_struct();
        let field = |index| {
            fields
                .field_by_idx(index)
                .ok_or_else(|| vortex_err!("Run summary partial is missing field {index}"))
        };
        let count = |index| {
            field(index)?
                .as_primitive()
                .typed_value::<u64>()
                .ok_or_else(|| vortex_err!("Run summary count must be non-null"))
        };
        let valid_count = count(0)?;
        let transitions = count(1)?;
        let first = field(2)?.as_primitive().pvalue();
        let last = field(3)?.as_primitive().pvalue();

        RunSummaryPartial::from_parts(valid_count, transitions, first, last)
    }

    fn merge_partials(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        first: Self::Partial,
        second: Self::Partial,
    ) -> VortexResult<Self::Partial> {
        first.merge(second)
    }

    fn to_scalar(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> VortexResult<Scalar> {
        let endpoint = |value: Option<PValue>| match value {
            Some(value) => Scalar::primitive_value(value, args.dtype.as_ptype(), Nullable),
            None => Scalar::null(args.dtype.as_nullable()),
        };

        Ok(Scalar::struct_(
            args.partial_dtype.clone(),
            [
                Scalar::primitive(partial.valid_count, NonNullable),
                Scalar::primitive(partial.transitions, NonNullable),
                endpoint(partial.first),
                endpoint(partial.last),
            ],
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
        _args: AggregateArgs<'_, Self::Options>,
        state: &mut Self::Partial,
        batch: &Columnar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let summary = match batch {
            Columnar::Constant(array) => {
                if array.is_empty() || array.scalar().is_null() {
                    return Ok(());
                }

                let value = array
                    .scalar()
                    .as_primitive()
                    .pvalue()
                    .ok_or_else(|| vortex_err!("Run summary constant must be non-null"))?;
                let valid_count = u64::try_from(array.len())?;
                RunSummaryPartial {
                    valid_count,
                    transitions: if value.is_nan() { valid_count - 1 } else { 0 },
                    first: Some(value),
                    last: Some(value),
                }
            }
            Columnar::Canonical(Canonical::Primitive(array)) => {
                let validity = array.validity()?.execute_mask(array.len(), ctx)?;
                if array.ptype().is_int() {
                    match_each_integer_ptype!(array.ptype(), |T| {
                        summarize_integer::<T>(array, &validity)?
                    })
                } else {
                    match_each_native_ptype!(array.ptype(), |T| {
                        summarize_primitive::<T>(array, &validity)?
                    })
                }
            }
            _ => vortex_bail!(
                "Run summary requires primitive input, got {}",
                batch.dtype()
            ),
        };

        *state = state.clone().merge(summary)?;
        Ok(())
    }

    fn finalize(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        partials: ArrayRef,
    ) -> VortexResult<ArrayRef> {
        let valid_count = partials.get_item("valid_count")?;
        let transitions = partials.get_item("transitions")?;
        let zero = ConstantArray::new(0u64, partials.len()).into_array();
        let one = ConstantArray::new(1u64, partials.len()).into_array();
        let has_values = valid_count.binary(zero.clone(), Operator::NotEq)?;
        let runs = transitions.binary(one, Operator::Add)?;

        has_values.zip(runs, zero)
    }

    fn finalize_scalar(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> VortexResult<Scalar> {
        Ok(Scalar::primitive(partial.run_count(), NonNullable))
    }
}

/// Summarize integers with the existing 64-value runs-only kernels.
fn summarize_integer<T: IntegerPType + Into<PValue>>(
    array: &PrimitiveArray,
    validity: &Mask,
) -> VortexResult<RunSummaryPartial> {
    let Some(head_index) = validity.first() else {
        return Ok(RunSummaryPartial::default());
    };
    let buffer = array.to_buffer::<T>();
    let first = buffer[head_index];
    let mut state = IntegerRunsLoop {
        prev: first,
        transitions: 0,
    };
    let sliced = buffer.slice(head_index..array.len());
    let (chunks, remainder) = sliced.as_slice().as_chunks::<64>();

    match validity.bit_buffer() {
        AllOr::All => {
            for chunk in chunks {
                integer_runs_nonnull(chunk, &mut state);
            }
            integer_runs_masked(remainder, &BitBuffer::new_set(remainder.len()), &mut state);
        }
        AllOr::None => unreachable!("The materialized mask contains a valid value"),
        AllOr::Some(valid) => {
            let mask = valid.slice(head_index..array.len());
            let mut offset = 0;
            for chunk in chunks {
                let chunk_validity = mask.slice(offset..offset + 64);
                offset += 64;

                match chunk_validity.true_count() {
                    0 => continue,
                    64 => integer_runs_nonnull(chunk, &mut state),
                    _ => integer_runs_masked(chunk, &chunk_validity, &mut state),
                }
            }
            integer_runs_masked(
                remainder,
                &mask.slice(offset..offset + remainder.len()),
                &mut state,
            );
        }
    }

    RunSummaryPartial::from_parts(
        u64::try_from(validity.true_count())?,
        state.transitions,
        Some(first.into()),
        Some(state.prev.into()),
    )
}

/// Native loop state for runs-only integer scanning.
struct IntegerRunsLoop<T> {
    /// Last valid value, carried across null slots and chunk boundaries.
    prev: T,
    /// Number of changes between adjacent valid values.
    transitions: u64,
}

/// Count changes in a fully valid integer chunk, including its incoming boundary.
#[allow(clippy::inline_always)]
#[inline(always)]
fn integer_runs_nonnull<T: IntegerPType>(values: &[T; 64], state: &mut IntegerRunsLoop<T>) {
    // Keep the legacy kernel's byte reduction. A 64-value chunk has at most 64 transitions.
    let transitions = u8::from(values[0] != state.prev)
        + values
            .iter()
            .zip(&values[1..])
            .map(|(first, second)| u8::from(first != second))
            .sum::<u8>();

    state.transitions += u64::from(transitions);
    state.prev = values[63];
}

/// Skip invalid slots without breaking a run in a partial chunk or the final remainder.
#[allow(clippy::inline_always)]
#[inline(always)]
fn integer_runs_masked<T: IntegerPType>(
    values: &[T],
    is_valid: &BitBuffer,
    state: &mut IntegerRunsLoop<T>,
) {
    for (index, &value) in values.iter().enumerate() {
        if is_valid.value(index) && value != state.prev {
            state.prev = value;
            state.transitions += 1;
        }
    }
}

/// Summarize native values after materializing validity once.
fn summarize_primitive<T: NativePType + Into<PValue>>(
    array: &PrimitiveArray,
    validity: &Mask,
) -> VortexResult<RunSummaryPartial> {
    let Some(first_index) = validity.first() else {
        return Ok(RunSummaryPartial::default());
    };
    let values = array.as_slice::<T>();
    let first = values[first_index];
    let mut last = first;
    let mut transitions = 0u64;

    match validity {
        Mask::AllTrue(_) => {
            for &value in &values[first_index + 1..] {
                transitions += u64::from(value != last);
                last = value;
            }
        }
        Mask::AllFalse(_) => unreachable!("The materialized mask contains a valid value"),
        Mask::Values(mask) => {
            mask.bit_buffer().for_each_set_index(|index| {
                if index == first_index {
                    return;
                }

                let value = values[index];
                transitions += u64::from(value != last);
                last = value;
            });
        }
    }

    Ok(RunSummaryPartial {
        valid_count: u64::try_from(validity.true_count())?,
        transitions,
        first: Some(first.into()),
        last: Some(last.into()),
    })
}

/// Compare retained boundaries using numeric equality rather than `PValue`'s bitwise float equality.
fn numeric_values_differ(first: PValue, second: PValue) -> bool {
    debug_assert_eq!(first.ptype(), second.ptype());

    match (first, second) {
        (PValue::F16(first), PValue::F16(second)) => first != second,
        (PValue::F32(first), PValue::F32(second)) => first != second,
        (PValue::F64(first), PValue::F64(second)) => first != second,
        _ => first != second,
    }
}
