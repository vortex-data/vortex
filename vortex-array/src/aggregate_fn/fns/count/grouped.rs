// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;

use num_traits::ToPrimitive;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use super::Count;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::aggregate_fn::GroupIds;
use crate::aggregate_fn::GroupRuns;
use crate::aggregate_fn::GroupedState;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::kernels::GroupedAggregateKernel;
use crate::aggregate_fn::kernels::GroupedAggregateKernelAdapter;
use crate::aggregate_fn::kernels::for_each_group_run;
use crate::aggregate_fn::kernels::has_long_group_runs;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::dtype::NativePType;
use crate::match_each_native_ptype;
use crate::scalar::Scalar;

#[derive(Default)]
pub(crate) struct CountGroupedState {
    counts: Vec<u64>,
}

impl CountGroupedState {
    fn counts_mut(&mut self) -> &mut [u64] {
        &mut self.counts
    }
}

impl GroupedState for CountGroupedState {
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn len(&self) -> usize {
        self.counts.len()
    }

    fn ensure_groups(&mut self, num_groups: usize) -> VortexResult<()> {
        self.counts.resize(num_groups.max(self.counts.len()), 0);
        Ok(())
    }

    fn is_saturated(&self, _group_id: usize) -> bool {
        false
    }

    fn combine_scalar(&mut self, group_id: usize, partial: Scalar) -> VortexResult<()> {
        self.counts[group_id] += partial
            .as_primitive()
            .typed_value::<u64>()
            .vortex_expect("count partial should not be null");
        Ok(())
    }

    fn partial_scalar(&self, group_id: usize) -> VortexResult<Scalar> {
        Ok(Scalar::primitive(
            self.counts.get(group_id).copied().unwrap_or(0),
            crate::dtype::Nullability::NonNullable,
        ))
    }

    fn accumulate_partials(
        &mut self,
        partials: &ArrayRef,
        group_ids: &[u32],
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let partials = partials.clone().execute::<Buffer<u64>>(ctx)?;
        for (&partial, &group_id) in partials.iter().zip(group_ids) {
            self.counts[group_id as usize] += partial;
        }
        Ok(())
    }

    fn flush_partials(&mut self) -> VortexResult<ArrayRef> {
        Ok(Buffer::from(std::mem::take(&mut self.counts)).into_array())
    }
}

pub(crate) static COUNT_GROUPED_KERNEL: GroupedAggregateKernelAdapter<Count, CountGroupedKernel> =
    GroupedAggregateKernelAdapter::new(CountGroupedKernel);

pub(crate) static COUNT_RUN_GROUPED_KERNEL: GroupedAggregateKernelAdapter<
    Count,
    CountRunGroupedKernel,
> = GroupedAggregateKernelAdapter::new(CountRunGroupedKernel);

#[derive(Debug)]
pub(crate) struct CountGroupedKernel;

impl GroupedAggregateKernel<Count> for CountGroupedKernel {
    type State = CountGroupedState;

    fn grouped_accumulate(
        &self,
        options: &NumericalAggregateOpts,
        state: &mut Self::State,
        batch: &ArrayRef,
        group_ids: &GroupIds,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<bool> {
        let states = state.counts_mut();
        if options.skip_nans && batch.dtype().is_float() {
            let Some(primitive) = batch.as_opt::<Primitive>() else {
                return Ok(false);
            };
            let group_ids = group_ids.validated_ids(ctx)?;
            accumulate_grouped_float_count(
                states,
                &primitive.into_owned(),
                group_ids.as_ref(),
                ctx,
            )?;
            return Ok(true);
        }

        let group_ids = group_ids.validated_ids(ctx)?;
        let group_ids = group_ids.as_ref();
        match batch
            .validity()?
            .execute_mask(batch.len(), ctx)?
            .bit_buffer()
        {
            AllOr::All if has_long_group_runs(group_ids) => {
                // Clustered ids: one add per run instead of one add per row.
                for_each_group_run(group_ids, |group_id, start, end| {
                    states[group_id as usize] +=
                        u64::try_from(end - start).vortex_expect("group run length must fit u64");
                });
            }
            AllOr::All => {
                for &group_id in group_ids {
                    states[group_id as usize] += 1;
                }
            }
            AllOr::None => {}
            AllOr::Some(valid) => valid.for_each_set_index(|idx| {
                states[group_ids[idx] as usize] += 1;
            }),
        }
        Ok(true)
    }
}

/// Grouped [`Count`] kernel for run-encoded group ids.
///
/// Counts a whole run at a time with one popcount over the validity mask, rather than reading an
/// id per row.
#[derive(Debug)]
pub(crate) struct CountRunGroupedKernel;

impl GroupedAggregateKernel<Count> for CountRunGroupedKernel {
    type State = CountGroupedState;

    fn grouped_accumulate(
        &self,
        options: &NumericalAggregateOpts,
        state: &mut Self::State,
        batch: &ArrayRef,
        group_ids: &GroupIds,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<bool> {
        let Some(runs) = group_ids.runs(ctx)? else {
            // Registered for the encoding, but these ids are not runs after all.
            return CountGroupedKernel.grouped_accumulate(options, state, batch, group_ids, ctx);
        };

        let states = state.counts_mut();
        if options.skip_nans && batch.dtype().is_float() {
            let Some(primitive) = batch.as_opt::<Primitive>() else {
                return Ok(false);
            };
            count_runs_non_nan(states, &primitive.into_owned(), &runs, ctx)?;
            return Ok(true);
        }

        match batch
            .validity()?
            .execute_mask(batch.len(), ctx)?
            .bit_buffer()
        {
            AllOr::All => runs.for_each(|group, start, end| {
                states[group] +=
                    u64::try_from(end - start).vortex_expect("group run length must fit u64");
            }),
            AllOr::None => {}
            AllOr::Some(valid) => runs.for_each(|group, start, end| {
                states[group] += u64::try_from(valid.count_range(start, end))
                    .vortex_expect("valid count must fit u64");
            }),
        }
        Ok(true)
    }
}

/// Count the valid, non-NaN values of each run into its group.
fn count_runs_non_nan(
    states: &mut [u64],
    primitive: &PrimitiveArray,
    runs: &GroupRuns,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let validity = primitive
        .as_ref()
        .validity()?
        .execute_mask(primitive.as_ref().len(), ctx)?;

    match_each_native_ptype!(primitive.ptype(),
        unsigned: |_T| { unreachable!("float count received an unsigned primitive") },
        signed: |_T| { unreachable!("float count received a signed primitive") },
        floating: |T| {
            let values = primitive.as_slice::<T>();
            let mut count_non_nan = |group: usize, start: usize, end: usize| {
                states[group] += values[start..end]
                    .iter()
                    .filter(|value| !value.is_nan())
                    .count() as u64;
            };
            match validity.bit_buffer() {
                AllOr::All => runs.for_each(count_non_nan),
                AllOr::None => {}
                AllOr::Some(valid) => runs.for_each(|group, start, end| {
                    for (valid_start, valid_end) in valid.as_view().slice(start..end).set_slices() {
                        count_non_nan(group, start + valid_start, start + valid_end);
                    }
                }),
            }
        }
    );
    Ok(())
}

fn accumulate_grouped_float_count(
    states: &mut [u64],
    primitive: &PrimitiveArray,
    group_ids: &[u32],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let validity = primitive
        .as_ref()
        .validity()?
        .execute_mask(primitive.as_ref().len(), ctx)?;

    match_each_native_ptype!(primitive.ptype(),
        unsigned: |_T| { unreachable!("float count received an unsigned primitive") },
        signed: |_T| { unreachable!("float count received a signed primitive") },
        floating: |T| {
            let values = primitive.as_slice::<T>();
            accumulate_valid_non_nan::<T>(states, values, group_ids, &validity);
        }
    );
    Ok(())
}

fn accumulate_valid_non_nan<T: NativePType + ToPrimitive>(
    states: &mut [u64],
    values: &[T],
    group_ids: &[u32],
    validity: &Mask,
) {
    let count_non_nan = |idx: usize| {
        let value = ToPrimitive::to_f64(&values[idx]).vortex_expect("float to f64");
        if !value.is_nan() {
            states[group_ids[idx] as usize] += 1;
        }
    };
    match validity.bit_buffer() {
        AllOr::All => (0..values.len()).for_each(count_non_nan),
        AllOr::None => {}
        AllOr::Some(valid) => valid.for_each_set_index(count_non_nan),
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::DynGroupedAccumulator;
    use crate::aggregate_fn::GroupIds;
    use crate::aggregate_fn::GroupedAccumulator;
    use crate::aggregate_fn::NumericalAggregateOpts;
    use crate::aggregate_fn::fns::count::Count;
    use crate::array_session;
    use crate::arrays::ConstantArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::VarBinViewArray;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::validity::Validity;

    fn run_grouped_count(
        values: &crate::ArrayRef,
        ids: impl IntoIterator<Item = u32>,
        num_groups: usize,
        options: NumericalAggregateOpts,
    ) -> VortexResult<crate::ArrayRef> {
        let mut acc = GroupedAccumulator::try_new(Count, options, values.dtype().clone())?;
        let group_ids = GroupIds::from_iter(ids, num_groups)?;
        let mut ctx = array_session().create_execution_ctx();
        acc.accumulate(values, &group_ids, &mut ctx)?;
        acc.finish()
    }

    /// Run-encoded ids must agree with the same ids materialized one per row.
    #[rstest]
    #[case::defaults(NumericalAggregateOpts::default())]
    #[case::include_nans(NumericalAggregateOpts::include_nans())]
    fn run_ids_match_dense_ids(#[case] options: NumericalAggregateOpts) -> VortexResult<()> {
        let values = PrimitiveArray::from_option_iter([
            Some(1.0f64),
            None,
            Some(f64::NAN),
            None,
            Some(4.0),
            Some(5.0),
            Some(f64::NAN),
        ])
        .into_array();
        // An empty run, an all-null run, and runs holding NaNs.
        let lengths = buffer![3u64, 0, 2, 2];
        let num_groups = lengths.len();
        let group_ids: Buffer<u32> = (0..num_groups)
            .map(u32::try_from)
            .collect::<Result<_, _>>()?;
        let mut ctx = array_session().create_execution_ctx();

        let mut runs = GroupedAccumulator::try_new(Count, options, values.dtype().clone())?;
        runs.accumulate(
            &values,
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
        let mut dense = GroupedAccumulator::try_new(Count, options, values.dtype().clone())?;
        dense.accumulate(&values, &GroupIds::from_iter(ids, num_groups)?, &mut ctx)?;
        let dense = dense.finish()?;

        assert_arrays_eq!(&runs, &dense, &mut ctx);
        Ok(())
    }

    #[test]
    fn dense_ids_repeat_reorder_and_omit_groups() -> VortexResult<()> {
        let values =
            PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), Some(4), None, Some(6)])
                .into_array();
        let actual = run_grouped_count(
            &values,
            [2, 0, 2, 0, 2, 0],
            4,
            NumericalAggregateOpts::default(),
        )?;
        let expected = PrimitiveArray::from_iter([2u64, 0, 2, 0]).into_array();
        let mut ctx = array_session().create_execution_ctx();
        assert_arrays_eq!(&actual, &expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn varbinview_counts_nulls() -> VortexResult<()> {
        let values = VarBinViewArray::from_iter_nullable_str([
            Some("a"),
            None,
            Some("bbb"),
            None,
            Some("cc"),
        ])
        .into_array();
        let actual = run_grouped_count(
            &values,
            [0, 0, 1, 1, 2],
            3,
            NumericalAggregateOpts::default(),
        )?;
        let expected = PrimitiveArray::from_iter([1u64, 1, 1]).into_array();
        let mut ctx = array_session().create_execution_ctx();
        assert_arrays_eq!(&actual, &expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn float_nan_options_match_scalar_count() -> VortexResult<()> {
        let values =
            PrimitiveArray::from_option_iter([Some(1.0f64), Some(f64::NAN), None, Some(3.0)])
                .into_array();
        let skipped =
            run_grouped_count(&values, [0, 0, 1, 1], 2, NumericalAggregateOpts::default())?;
        let included = run_grouped_count(
            &values,
            [0, 0, 1, 1],
            2,
            NumericalAggregateOpts::include_nans(),
        )?;
        let mut ctx = array_session().create_execution_ctx();
        assert_arrays_eq!(
            &skipped,
            &PrimitiveArray::from_iter([1u64, 1]).into_array(),
            &mut ctx
        );
        assert_arrays_eq!(
            &included,
            &PrimitiveArray::from_iter([2u64, 1]).into_array(),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn encoded_constant_group_ids() -> VortexResult<()> {
        let values =
            PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), Some(4)]).into_array();
        let group_ids = GroupIds::new(ConstantArray::new(1u32, values.len()).into_array(), 3)?;
        let mut ctx = array_session().create_execution_ctx();
        let mut acc = GroupedAccumulator::try_new(
            Count,
            NumericalAggregateOpts::default(),
            values.dtype().clone(),
        )?;
        acc.accumulate(&values, &group_ids, &mut ctx)?;
        let actual = acc.finish()?;
        let expected = PrimitiveArray::from_iter([0u64, 3, 0]).into_array();
        assert_arrays_eq!(&actual, &expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn rejects_out_of_range_group_id() -> VortexResult<()> {
        assert!(GroupIds::from_iter([0u32, 2], 2).is_err());

        let values = PrimitiveArray::new(buffer![1i32, 2], Validity::NonNullable).into_array();
        let group_ids = GroupIds::new(
            PrimitiveArray::new(buffer![0u32, 2], Validity::NonNullable).into_array(),
            2,
        )?;
        let mut ctx = array_session().create_execution_ctx();
        let mut acc = GroupedAccumulator::try_new(
            Count,
            NumericalAggregateOpts::default(),
            values.dtype().clone(),
        )?;
        assert!(acc.accumulate(&values, &group_ids, &mut ctx).is_err());
        Ok(())
    }

    #[test]
    fn accumulates_partials_and_merges_groups() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::Nullable);
        let partials = PrimitiveArray::from_iter([2u64, 3, 5]).into_array();
        let mut ctx = array_session().create_execution_ctx();
        let mut left =
            GroupedAccumulator::try_new(Count, NumericalAggregateOpts::default(), dtype.clone())?;
        left.accumulate_partials(&partials, &GroupIds::from_iter([0u32, 1, 1], 2)?, &mut ctx)?;
        let mut right =
            GroupedAccumulator::try_new(Count, NumericalAggregateOpts::default(), dtype)?;
        right.merge_group(0, &left, 1)?;
        let actual = right.finish()?;
        let expected = PrimitiveArray::from_iter([8u64]).into_array();
        assert_arrays_eq!(&actual, &expected, &mut ctx);
        Ok(())
    }
}
