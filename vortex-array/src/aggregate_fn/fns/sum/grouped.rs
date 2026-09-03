// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use num_traits::CheckedAdd;
use num_traits::ToPrimitive;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use super::Sum;
use super::checked_add_i64;
use super::checked_add_u64;
use super::grouped_state::DenseSums;
use super::grouped_state::SumGroupedState;
use super::grouped_state::SumGroupedValues;
use super::grouped_state::add_decimal;
use super::grouped_state::match_each_decimal_state;
use super::primitive::sum_float_all;
use super::primitive::sum_signed_all;
use super::primitive::sum_unsigned_all;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::aggregate_fn::GroupIds;
use crate::aggregate_fn::GroupRuns;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::kernels::GroupedAggregateKernel;
use crate::aggregate_fn::kernels::GroupedAggregateKernelAdapter;
use crate::aggregate_fn::kernels::MIN_GROUP_RUN_LENGTH;
use crate::aggregate_fn::kernels::for_each_group_run;
use crate::aggregate_fn::kernels::has_long_group_runs;
use crate::arrays::Bool;
use crate::arrays::BoolArray;
use crate::arrays::Decimal;
use crate::arrays::DecimalArray;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::bool::BoolArrayExt;
use crate::dtype::DecimalDType;
use crate::dtype::NativeDecimalType;
use crate::dtype::NativePType;
use crate::match_each_decimal_value_type;
use crate::match_each_native_ptype;

pub(crate) static SUM_GROUPED_KERNEL: GroupedAggregateKernelAdapter<Sum, SumGroupedKernel> =
    GroupedAggregateKernelAdapter::new(SumGroupedKernel);

pub(crate) static SUM_RUN_GROUPED_KERNEL: GroupedAggregateKernelAdapter<Sum, SumRunGroupedKernel> =
    GroupedAggregateKernelAdapter::new(SumRunGroupedKernel);

/// Grouped [`Sum`] kernel for canonical values arrays.
///
/// Sums are written straight into the aggregate's dense typed state, so a group never
/// materializes an intermediate partial scalar. Overflow saturates one group at a time, matching
/// the scalar `Sum` contract where a null partial means "overflowed".
#[derive(Debug)]
pub(crate) struct SumGroupedKernel;

impl GroupedAggregateKernel<Sum> for SumGroupedKernel {
    type State = SumGroupedState;

    fn grouped_accumulate(
        &self,
        options: &NumericalAggregateOpts,
        state: &mut Self::State,
        batch: &ArrayRef,
        group_ids: &GroupIds,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<bool> {
        let Some(batch) = SumBatch::try_new(batch) else {
            return Ok(false);
        };
        let group_ids = group_ids.validated_ids(ctx)?;
        let validity = batch.validity(ctx)?;
        batch.accumulate(
            state.sums_mut(),
            group_ids.as_ref(),
            &validity,
            options.skip_nans,
        );
        Ok(true)
    }
}

/// Grouped [`Sum`] kernel for run-encoded group ids.
///
/// Reduces one run of values per group rather than reading an id per row, which is what an
/// already-grouped input - a list array adapted by
/// [`GroupedArray::dense_input`](crate::aggregate_fn::GroupedArray::dense_input) - hands over.
#[derive(Debug)]
pub(crate) struct SumRunGroupedKernel;

impl GroupedAggregateKernel<Sum> for SumRunGroupedKernel {
    type State = SumGroupedState;

    fn grouped_accumulate(
        &self,
        options: &NumericalAggregateOpts,
        state: &mut Self::State,
        batch: &ArrayRef,
        group_ids: &GroupIds,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<bool> {
        let Some(batch) = SumBatch::try_new(batch) else {
            return Ok(false);
        };
        let validity = batch.validity(ctx)?;
        let Some(runs) = group_ids.runs(ctx)? else {
            // Registered for the encoding, but these ids are not runs after all.
            let group_ids = group_ids.validated_ids(ctx)?;
            batch.accumulate(
                state.sums_mut(),
                group_ids.as_ref(),
                &validity,
                options.skip_nans,
            );
            return Ok(true);
        };
        batch.accumulate_runs(state.sums_mut(), &runs, &validity, options.skip_nans);
        Ok(true)
    }
}

/// A canonical values batch that a dense grouped sum kernel can consume.
///
/// Shared by the [`Sum`] and [`SumV2`](crate::aggregate_fn::fns::sum_v2::SumV2) kernels, which sum
/// values identically and differ only in the extra state `SumV2` keeps alongside the sums.
pub(crate) enum SumBatch {
    Primitive(PrimitiveArray),
    Bool(BoolArray),
    Decimal(DecimalArray),
}

impl SumBatch {
    /// Recognize a batch encoding that can be summed in place, or `None` to keep executing it.
    pub(crate) fn try_new(batch: &ArrayRef) -> Option<Self> {
        if let Some(primitive) = batch.as_opt::<Primitive>() {
            return Some(Self::Primitive(primitive.into_owned()));
        }
        if let Some(bools) = batch.as_opt::<Bool>() {
            return Some(Self::Bool(bools.into_owned()));
        }
        batch
            .as_opt::<Decimal>()
            .map(|decimals| Self::Decimal(decimals.into_owned()))
    }

    /// The batch's validity mask.
    pub(crate) fn validity(&self, ctx: &mut ExecutionCtx) -> VortexResult<Mask> {
        let array = match self {
            Self::Primitive(primitive) => primitive.as_ref(),
            Self::Bool(bools) => bools.as_ref(),
            Self::Decimal(decimals) => decimals.as_ref(),
        };
        array.validity()?.execute_mask(array.len(), ctx)
    }

    /// Add every valid value of each run into the sum of that run's group.
    pub(crate) fn accumulate_runs(
        &self,
        sums: &mut DenseSums,
        runs: &GroupRuns,
        validity: &Mask,
        skip_nans: bool,
    ) {
        match self {
            Self::Primitive(primitive) => {
                accumulate_runs_primitive(sums, primitive, runs, validity, skip_nans);
            }
            Self::Bool(bools) => accumulate_runs_bool(sums, bools, runs, validity),
            Self::Decimal(decimals) => accumulate_runs_decimal(sums, decimals, runs, validity),
        }
    }

    /// Add every valid value into the sum of the group its id names.
    pub(crate) fn accumulate(
        &self,
        sums: &mut DenseSums,
        group_ids: &[u32],
        validity: &Mask,
        skip_nans: bool,
    ) {
        match self {
            Self::Primitive(primitive) => {
                accumulate_grouped_primitive(sums, primitive, group_ids, validity, skip_nans);
            }
            Self::Bool(bools) => accumulate_grouped_bool(sums, bools, group_ids, validity),
            Self::Decimal(decimals) => {
                accumulate_grouped_decimal(sums, decimals, group_ids, validity);
            }
        }
    }
}

fn accumulate_grouped_unsigned(
    sums: &mut [u64],
    overflowed: &mut [bool],
    group: usize,
    value: u64,
) {
    if checked_add_u64(&mut sums[group], value) {
        overflowed[group] = true;
    }
}

fn accumulate_grouped_unsigned_all<T>(
    sums: &mut [u64],
    overflowed: &mut [bool],
    values: &[T],
    group_ids: &[u32],
) where
    T: NativePType + AsPrimitive<u64>,
{
    if !has_long_group_runs(group_ids) {
        for (&value, &group_id) in values.iter().zip(group_ids) {
            accumulate_grouped_unsigned(sums, overflowed, group_id as usize, value.as_());
        }
        return;
    }

    for_each_group_run(group_ids, |group_id, start, end| {
        let group = group_id as usize;
        if end - start >= MIN_GROUP_RUN_LENGTH {
            if sum_unsigned_all(&mut sums[group], &values[start..end]) {
                overflowed[group] = true;
            }
        } else {
            for &value in &values[start..end] {
                accumulate_grouped_unsigned(sums, overflowed, group, value.as_());
            }
        }
    });
}

fn accumulate_grouped_signed(sums: &mut [i64], overflowed: &mut [bool], group: usize, value: i64) {
    if checked_add_i64(&mut sums[group], value) {
        overflowed[group] = true;
    }
}

fn accumulate_grouped_signed_all<T>(
    sums: &mut [i64],
    overflowed: &mut [bool],
    values: &[T],
    group_ids: &[u32],
) where
    T: NativePType + AsPrimitive<i64>,
{
    if !has_long_group_runs(group_ids) {
        for (&value, &group_id) in values.iter().zip(group_ids) {
            accumulate_grouped_signed(sums, overflowed, group_id as usize, value.as_());
        }
        return;
    }

    for_each_group_run(group_ids, |group_id, start, end| {
        let group = group_id as usize;
        if end - start >= MIN_GROUP_RUN_LENGTH {
            if sum_signed_all(&mut sums[group], &values[start..end]) {
                overflowed[group] = true;
            }
        } else {
            for &value in &values[start..end] {
                accumulate_grouped_signed(sums, overflowed, group, value.as_());
            }
        }
    });
}

fn accumulate_grouped_float<T: NativePType>(
    sums: &mut [f64],
    group: usize,
    value: T,
    skip_nans: bool,
) {
    if skip_nans && value.is_nan() {
        return;
    }
    sums[group] += ToPrimitive::to_f64(&value).vortex_expect("float to f64");
}

fn accumulate_grouped_float_all<T: NativePType>(
    sums: &mut [f64],
    values: &[T],
    group_ids: &[u32],
    skip_nans: bool,
) {
    if !has_long_group_runs(group_ids) {
        for (&value, &group_id) in values.iter().zip(group_ids) {
            accumulate_grouped_float(sums, group_id as usize, value, skip_nans);
        }
        return;
    }

    for_each_group_run(group_ids, |group_id, start, end| {
        let group = group_id as usize;
        if end - start >= MIN_GROUP_RUN_LENGTH {
            sum_float_all(&mut sums[group], &values[start..end], skip_nans);
        } else {
            for &value in &values[start..end] {
                accumulate_grouped_float(sums, group, value, skip_nans);
            }
        }
    });
}

fn accumulate_grouped_primitive(
    sums: &mut DenseSums,
    primitive: &PrimitiveArray,
    group_ids: &[u32],
    validity: &Mask,
    skip_nans: bool,
) {
    let validity = validity.bit_buffer();
    let (state, overflowed) = sums.parts_mut();

    match_each_native_ptype!(primitive.ptype(),
        unsigned: |T| {
            let SumGroupedValues::Unsigned(sums) = state else {
                vortex_panic!("unsigned input with non-unsigned grouped sum state")
            };
            let values = primitive.as_slice::<T>();
            match validity {
                AllOr::All => {
                    accumulate_grouped_unsigned_all(sums, overflowed, values, group_ids);
                }
                AllOr::None => {}
                AllOr::Some(valid) => valid.for_each_set_index(|idx| {
                    accumulate_grouped_unsigned(
                        sums,
                        overflowed,
                        group_ids[idx] as usize,
                        values[idx].as_(),
                    );
                }),
            }
        },
        signed: |T| {
            let SumGroupedValues::Signed(sums) = state else {
                vortex_panic!("signed input with non-signed grouped sum state")
            };
            let values = primitive.as_slice::<T>();
            match validity {
                AllOr::All => {
                    accumulate_grouped_signed_all(sums, overflowed, values, group_ids);
                }
                AllOr::None => {}
                AllOr::Some(valid) => valid.for_each_set_index(|idx| {
                    accumulate_grouped_signed(
                        sums,
                        overflowed,
                        group_ids[idx] as usize,
                        values[idx].as_(),
                    );
                }),
            }
        },
        floating: |T| {
            let SumGroupedValues::Float(sums) = state else {
                vortex_panic!("float input with non-float grouped sum state")
            };
            let values = primitive.as_slice::<T>();
            match validity {
                AllOr::All => {
                    accumulate_grouped_float_all(sums, values, group_ids, skip_nans);
                }
                AllOr::None => {}
                AllOr::Some(valid) => valid.for_each_set_index(|idx| {
                    accumulate_grouped_float(sums, group_ids[idx] as usize, values[idx], skip_nans);
                }),
            }
        }
    );
}

fn accumulate_grouped_bool(
    sums: &mut DenseSums,
    bools: &BoolArray,
    group_ids: &[u32],
    validity: &Mask,
) {
    let values = bools.to_bit_buffer();
    let valid_true = match validity.bit_buffer() {
        AllOr::All => values,
        AllOr::None => return,
        AllOr::Some(validity) => &values & validity,
    };
    let (state, overflowed) = sums.parts_mut();
    let SumGroupedValues::Unsigned(sums) = state else {
        vortex_panic!("boolean input with non-unsigned grouped sum state")
    };
    // Only true values contribute, so nulls and falses need no per-row work.
    valid_true.for_each_set_index(|idx| {
        accumulate_grouped_unsigned(sums, overflowed, group_ids[idx] as usize, 1);
    });
}

fn accumulate_grouped_decimal(
    sums: &mut DenseSums,
    decimals: &DecimalArray,
    group_ids: &[u32],
    validity: &Mask,
) {
    let output_dtype = sums
        .decimal_dtype()
        .vortex_expect("decimal sum state dtype");
    let (state, overflowed) = sums.parts_mut();
    match_each_decimal_value_type!(decimals.values_type(), |T| {
        let values = decimals.buffer::<T>();
        match_each_decimal_state!(
            state,
            |sums| accumulate_grouped_decimal_values(
                sums,
                overflowed,
                values,
                group_ids,
                validity,
                output_dtype,
            ),
            vortex_panic!("decimal input with non-decimal grouped sum state")
        );
    });
}

fn accumulate_grouped_decimal_values<T, I>(
    sums: &mut [I],
    overflowed: &mut [bool],
    values: Buffer<T>,
    group_ids: &[u32],
    validity: &Mask,
    dtype: DecimalDType,
) where
    T: NativeDecimalType + AsPrimitive<I>,
    I: NativeDecimalType + CheckedAdd,
{
    let add = |idx: usize| {
        add_decimal(
            sums,
            overflowed,
            group_ids[idx] as usize,
            values[idx].as_(),
            dtype,
        );
    };
    match validity.bit_buffer() {
        AllOr::All => (0..values.len()).for_each(add),
        AllOr::None => {}
        AllOr::Some(valid) => valid.for_each_set_index(add),
    }
}

/// Invoke `f(group, start, end)` for each maximal run of valid rows inside each group run.
fn for_each_valid_run(runs: &GroupRuns, validity: &Mask, mut f: impl FnMut(usize, usize, usize)) {
    match validity.bit_buffer() {
        AllOr::All => runs.for_each(&mut f),
        AllOr::None => {}
        AllOr::Some(valid) => runs.for_each(|group, start, end| {
            for (valid_start, valid_end) in valid.as_view().slice(start..end).set_slices() {
                f(group, start + valid_start, start + valid_end);
            }
        }),
    }
}

fn accumulate_runs_primitive(
    sums: &mut DenseSums,
    primitive: &PrimitiveArray,
    runs: &GroupRuns,
    validity: &Mask,
    skip_nans: bool,
) {
    let (state, overflowed) = sums.parts_mut();

    match_each_native_ptype!(primitive.ptype(),
        unsigned: |T| {
            let SumGroupedValues::Unsigned(sums) = state else {
                vortex_panic!("unsigned input with non-unsigned grouped sum state")
            };
            let values = primitive.as_slice::<T>();
            for_each_valid_run(runs, validity, |group, start, end| {
                if sum_unsigned_all(&mut sums[group], &values[start..end]) {
                    overflowed[group] = true;
                }
            });
        },
        signed: |T| {
            let SumGroupedValues::Signed(sums) = state else {
                vortex_panic!("signed input with non-signed grouped sum state")
            };
            let values = primitive.as_slice::<T>();
            for_each_valid_run(runs, validity, |group, start, end| {
                if sum_signed_all(&mut sums[group], &values[start..end]) {
                    overflowed[group] = true;
                }
            });
        },
        floating: |T| {
            let SumGroupedValues::Float(sums) = state else {
                vortex_panic!("float input with non-float grouped sum state")
            };
            let values = primitive.as_slice::<T>();
            for_each_valid_run(runs, validity, |group, start, end| {
                sum_float_all(&mut sums[group], &values[start..end], skip_nans);
            });
        }
    );
}

fn accumulate_runs_bool(
    sums: &mut DenseSums,
    bools: &BoolArray,
    runs: &GroupRuns,
    validity: &Mask,
) {
    let values = bools.to_bit_buffer();
    let valid_true = match validity.bit_buffer() {
        AllOr::All => values,
        AllOr::None => return,
        AllOr::Some(validity) => &values & validity,
    };
    let (state, overflowed) = sums.parts_mut();
    let SumGroupedValues::Unsigned(sums) = state else {
        vortex_panic!("boolean input with non-unsigned grouped sum state")
    };
    // Each run contributes its count of valid true values.
    runs.for_each(|group, start, end| {
        let true_count = valid_true.count_range(start, end) as u64;
        if true_count != 0 && checked_add_u64(&mut sums[group], true_count) {
            overflowed[group] = true;
        }
    });
}

fn accumulate_runs_decimal(
    sums: &mut DenseSums,
    decimals: &DecimalArray,
    runs: &GroupRuns,
    validity: &Mask,
) {
    let output_dtype = sums
        .decimal_dtype()
        .vortex_expect("decimal sum state dtype");
    let (state, overflowed) = sums.parts_mut();
    match_each_decimal_value_type!(decimals.values_type(), |T| {
        let values = decimals.buffer::<T>();
        match_each_decimal_state!(
            state,
            |sums| for_each_valid_run(runs, validity, |group, start, end| {
                for value in &values[start..end] {
                    add_decimal(sums, overflowed, group, value.as_(), output_dtype);
                }
            }),
            vortex_panic!("decimal input with non-decimal grouped sum state")
        );
    });
}

#[cfg(test)]
mod tests {
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::DynGroupedAccumulator;
    use crate::aggregate_fn::GroupIds;
    use crate::aggregate_fn::GroupedAccumulator;
    use crate::aggregate_fn::NumericalAggregateOpts;
    use crate::aggregate_fn::fns::sum::Sum;
    use crate::array_session;
    use crate::arrays::BoolArray;
    use crate::arrays::DecimalArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::DecimalDType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::i256;
    use crate::scalar::DecimalValue;
    use crate::validity::Validity;

    fn run_grouped_sum(
        values: &ArrayRef,
        ids: impl IntoIterator<Item = u32>,
        num_groups: usize,
        options: NumericalAggregateOpts,
    ) -> VortexResult<ArrayRef> {
        let mut acc = GroupedAccumulator::try_new(Sum, options, values.dtype().clone())?;
        let group_ids = GroupIds::from_iter(ids, num_groups)?;
        let mut ctx = array_session().create_execution_ctx();
        acc.accumulate(values, &group_ids, &mut ctx)?;
        acc.finish()
    }

    /// Run-encoded ids must agree with the same ids materialized one per row.
    fn run_and_dense_agree(
        values: &ArrayRef,
        lengths: impl IntoIterator<Item = u64>,
        options: NumericalAggregateOpts,
    ) -> VortexResult<()> {
        let lengths: Buffer<u64> = lengths.into_iter().collect();
        let num_groups = lengths.len();
        let group_ids: Buffer<u32> = (0..num_groups)
            .map(u32::try_from)
            .collect::<Result<_, _>>()?;
        let mut ctx = array_session().create_execution_ctx();

        let mut runs = GroupedAccumulator::try_new(Sum, options, values.dtype().clone())?;
        runs.accumulate(
            values,
            &GroupIds::from_runs(group_ids.clone(), lengths.clone(), num_groups)?,
            &mut ctx,
        )?;
        let runs = runs.finish()?;

        let ids =
            group_ids
                .as_ref()
                .iter()
                .zip(lengths.as_ref())
                .flat_map(|(&group_id, &length)| {
                    std::iter::repeat_n(group_id, usize::try_from(length).unwrap_or_default())
                });
        let mut dense = GroupedAccumulator::try_new(Sum, options, values.dtype().clone())?;
        dense.accumulate(values, &GroupIds::from_iter(ids, num_groups)?, &mut ctx)?;
        let dense = dense.finish()?;

        assert_arrays_eq!(&runs, &dense, &mut ctx);
        Ok(())
    }

    #[test]
    fn run_ids_match_dense_ids() -> VortexResult<()> {
        // Empty runs, an all-null group, and a group that overflows.
        let values = PrimitiveArray::from_option_iter([
            Some(1i64),
            None,
            Some(3),
            None,
            None,
            Some(i64::MAX),
            Some(1),
            Some(7),
        ])
        .into_array();
        run_and_dense_agree(&values, [3, 0, 2, 2, 1], NumericalAggregateOpts::default())?;

        let bools: BoolArray = [
            Some(true),
            Some(false),
            None,
            Some(true),
            Some(true),
            None,
            Some(true),
            Some(false),
        ]
        .into_iter()
        .collect();
        run_and_dense_agree(
            &bools.into_array(),
            [2, 0, 3, 3],
            NumericalAggregateOpts::default(),
        )?;

        let decimals = DecimalArray::new(
            buffer![100i64, 200, -50, 300, 400, 0],
            DecimalDType::new(10, 2),
            Validity::from_iter([true, true, true, false, true, true]),
        )
        .into_array();
        run_and_dense_agree(&decimals, [2, 1, 0, 3], NumericalAggregateOpts::default())?;

        let floats = PrimitiveArray::from_option_iter([
            Some(1.0f64),
            Some(f64::NAN),
            Some(2.0),
            None,
            Some(4.0),
        ])
        .into_array();
        run_and_dense_agree(&floats, [3, 2], NumericalAggregateOpts::default())?;
        run_and_dense_agree(&floats, [3, 2], NumericalAggregateOpts::include_nans())?;
        Ok(())
    }

    #[test]
    fn dense_ids_repeat_reorder_and_omit_groups() -> VortexResult<()> {
        let values = PrimitiveArray::from_option_iter([
            Some(1i32),
            None,
            Some(3),
            Some(4),
            Some(5),
            Some(6),
        ])
        .into_array();
        // Group 1 sees no values and group 3 is absent from the batch: both sum to zero.
        let actual = run_grouped_sum(
            &values,
            [2, 0, 2, 0, 2, 0],
            4,
            NumericalAggregateOpts::default(),
        )?;
        let expected =
            PrimitiveArray::from_option_iter([Some(10i64), Some(0), Some(9), Some(0)]).into_array();
        let mut ctx = array_session().create_execution_ctx();
        assert_arrays_eq!(&actual, &expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn bool_and_overflow_are_group_local() -> VortexResult<()> {
        let bools: BoolArray = [true, false, true, true].into_iter().collect();
        let actual = run_grouped_sum(
            &bools.into_array(),
            [1, 0, 1, 0],
            2,
            NumericalAggregateOpts::default(),
        )?;
        let mut ctx = array_session().create_execution_ctx();
        assert_arrays_eq!(
            &actual,
            &PrimitiveArray::from_option_iter([Some(1u64), Some(2)]).into_array(),
            &mut ctx
        );

        let values =
            PrimitiveArray::new(buffer![i64::MAX, 1, 2, 3], Validity::NonNullable).into_array();
        let actual = run_grouped_sum(&values, [0, 0, 1, 1], 2, NumericalAggregateOpts::default())?;
        assert_arrays_eq!(
            &actual,
            &PrimitiveArray::from_option_iter([None, Some(5i64)]).into_array(),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn clustered_ids_match_shuffled_ids() -> VortexResult<()> {
        // Runs long enough to take the run-reduction path must agree with the scatter path.
        let values = PrimitiveArray::from_iter((0..64i32).map(|value| value * 3)).into_array();
        let clustered = run_grouped_sum(
            &values,
            (0..64u32).map(|idx| idx / 8),
            8,
            NumericalAggregateOpts::default(),
        )?;
        let shuffled = run_grouped_sum(
            &values,
            (0..64u32).map(|idx| idx % 8),
            8,
            NumericalAggregateOpts::default(),
        )?;
        let mut ctx = array_session().create_execution_ctx();
        let expected_clustered = PrimitiveArray::from_option_iter(
            (0..8i64).map(|group| Some((0..8).map(|idx| (group * 8 + idx) * 3).sum::<i64>())),
        )
        .into_array();
        let expected_shuffled = PrimitiveArray::from_option_iter(
            (0..8i64).map(|group| Some((0..8).map(|idx| (idx * 8 + group) * 3).sum::<i64>())),
        )
        .into_array();
        assert_arrays_eq!(&clustered, &expected_clustered, &mut ctx);
        assert_arrays_eq!(&shuffled, &expected_shuffled, &mut ctx);
        Ok(())
    }

    #[test]
    fn float_nan_options_match_scalar_sum() -> VortexResult<()> {
        let values =
            PrimitiveArray::new(buffer![1.0f64, f64::NAN, 2.0, 4.0], Validity::NonNullable)
                .into_array();
        let skipped = run_grouped_sum(&values, [0, 0, 1, 1], 2, NumericalAggregateOpts::default())?;
        let included = run_grouped_sum(
            &values,
            [0, 0, 1, 1],
            2,
            NumericalAggregateOpts::include_nans(),
        )?;
        let mut ctx = array_session().create_execution_ctx();
        assert_arrays_eq!(
            &skipped,
            &PrimitiveArray::from_option_iter([Some(1.0f64), Some(6.0)]).into_array(),
            &mut ctx
        );
        let group_zero = included.execute_scalar(0, &mut ctx)?;
        assert!(
            group_zero
                .as_primitive()
                .typed_value::<f64>()
                .vortex_expect("grouped float sum should be non-null")
                .is_nan()
        );
        Ok(())
    }

    #[test]
    fn exact_decimal_sum_with_reordered_ids_and_nulls() -> VortexResult<()> {
        let input_dtype = DecimalDType::new(10, 2);
        let values = DecimalArray::new(
            buffer![100i64, 200, -50, 300, 400],
            input_dtype,
            Validity::from_iter([true, true, true, false, true]),
        )
        .into_array();
        let actual = run_grouped_sum(
            &values,
            [2, 0, 2, 0, 2],
            4,
            NumericalAggregateOpts::default(),
        )?;
        let output_dtype = DecimalDType::new(20, 2);
        let expected =
            DecimalArray::new(buffer![200i64, 0, 450, 0], output_dtype, Validity::AllValid)
                .into_array();
        let mut ctx = array_session().create_execution_ctx();
        assert_arrays_eq!(&actual, &expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn exact_decimal_overflow_is_group_local() -> VortexResult<()> {
        let one = i256::from_i128(1);
        let large = i256::from_i128(10)
            .checked_pow(76)
            .vortex_expect("10^76 must fit in i256")
            - one;
        let dtype = DecimalDType::new(76, 0);
        let values = DecimalArray::new(
            buffer![large, i256::from_i128(7), large],
            dtype,
            Validity::NonNullable,
        )
        .into_array();
        let actual = run_grouped_sum(&values, [0, 1, 0], 2, NumericalAggregateOpts::default())?;
        let expected = DecimalArray::new(
            buffer![i256::ZERO, i256::from_i128(7)],
            dtype,
            Validity::from_iter([false, true]),
        )
        .into_array();
        let mut ctx = array_session().create_execution_ctx();
        assert_arrays_eq!(&actual, &expected, &mut ctx);

        let group_one = actual.execute_scalar(1, &mut ctx)?;
        assert_eq!(
            group_one.as_decimal().decimal_value(),
            Some(DecimalValue::I256(i256::from_i128(7)))
        );
        Ok(())
    }

    #[test]
    fn accumulates_typed_primitive_partials() -> VortexResult<()> {
        let input_dtype = DType::Primitive(PType::I32, Nullability::Nullable);
        // A null partial is a saturated group, so group 0 stays null.
        let partials =
            PrimitiveArray::from_option_iter([Some(2i64), Some(3), Some(5), None]).into_array();
        let mut ctx = array_session().create_execution_ctx();
        let mut acc =
            GroupedAccumulator::try_new(Sum, NumericalAggregateOpts::default(), input_dtype)?;
        acc.accumulate_partials(
            &partials,
            &GroupIds::from_iter([0u32, 1, 1, 0], 2)?,
            &mut ctx,
        )?;
        let actual = acc.finish()?;
        let expected = PrimitiveArray::from_option_iter([None, Some(8i64)]).into_array();
        assert_arrays_eq!(&actual, &expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn accumulates_typed_decimal_partials() -> VortexResult<()> {
        let input_dtype = DecimalDType::new(10, 2);
        let partial_dtype = DecimalDType::new(20, 2);
        let partials = DecimalArray::new(
            buffer![200i64, 300, 500, 0],
            partial_dtype,
            Validity::from_iter([true, true, true, false]),
        )
        .into_array();
        let mut ctx = array_session().create_execution_ctx();
        let mut acc = GroupedAccumulator::try_new(
            Sum,
            NumericalAggregateOpts::default(),
            DType::Decimal(input_dtype, Nullability::Nullable),
        )?;
        acc.accumulate_partials(
            &partials,
            &GroupIds::from_iter([0u32, 1, 1, 0], 2)?,
            &mut ctx,
        )?;
        let actual = acc.finish()?;
        let expected = DecimalArray::new(
            buffer![0i128, 800],
            partial_dtype,
            Validity::from_iter([false, true]),
        )
        .into_array();
        assert_arrays_eq!(&actual, &expected, &mut ctx);
        Ok(())
    }
}
