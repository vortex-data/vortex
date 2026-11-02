// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::DynGroupedAccumulator;
use vortex_array::aggregate_fn::GroupedAccumulator;
use vortex_array::aggregate_fn::GroupedArray;
use vortex_array::aggregate_fn::fns::sum::Sum;
use vortex_array::aggregate_fn::fns::sum_v2::SumV2;
use vortex_array::aggregate_fn::fns::sum_v2::SumV2GroupPartials;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::aggregate_fn::kernels::DynGroupedAggregateKernel;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;

use crate::Sparse;
use crate::SparseExt as _;

/// Sparse-specific `sum` and `sum_v2` kernel.
///
/// `sum(Sparse{ F, patches }) = sum(patches.values) + F * (N - patches.num_patches())`.
///
/// The constant contribution is computed via the aggregate accumulator's constant short-circuit
/// (`multiply_constant`), so overflow and empty-input semantics match the baseline. The work is
/// `O(P)` instead of `O(N)`.
#[derive(Debug)]
pub(crate) struct SparseSumKernel;

impl DynAggregateKernel for SparseSumKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        if !aggregate_fn.is::<Sum>() && !aggregate_fn.is::<SumV2>() {
            return Ok(None);
        }

        let Some(sparse) = batch.as_opt::<Sparse>() else {
            return Ok(None);
        };

        let patches = sparse.patches();
        let n_fill = sparse.len() - patches.num_patches();

        // Build a fresh accumulator over the array dtype and fold in the fill and patch
        // contributions. This preserves each aggregate's partial state and semantics.
        let mut acc = aggregate_fn.accumulator(batch.dtype())?;

        if n_fill > 0 {
            let fill_array = ConstantArray::new(sparse.fill_scalar().clone(), n_fill).into_array();
            acc.accumulate(&fill_array, ctx)?;
        }

        if !patches.values().is_empty() {
            acc.accumulate(patches.values(), ctx)?;
        }

        Ok(Some(acc.partial_scalar()?))
    }
}

/// Grouped [`SumV2`] kernel for sparse elements.
///
/// The patches in each group are summed as one group of the patch values, and the fill adds once
/// for every element of the group no patch covers, so the sparse elements are never decoded.
#[derive(Debug)]
pub(crate) struct SparseGroupedSumV2Kernel;

impl DynGroupedAggregateKernel for SparseGroupedSumV2Kernel {
    fn grouped_aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        groups: &GroupedArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(options) = aggregate_fn.as_opt::<SumV2>() else {
            return Ok(None);
        };
        let Some(sparse) = groups.elements().as_opt::<Sparse>() else {
            return Ok(None);
        };
        let Some(mut partials) =
            SumV2GroupPartials::new(options, groups.elements().dtype(), groups.len())
        else {
            return Ok(None);
        };
        let group_ranges = groups.group_ranges(ctx)?;
        let group_validity = groups.group_validity(ctx)?;

        let patches = sparse.patches();
        let patch_offset = patches.offset() as u64;
        let indices = patches
            .indices()
            .cast(DType::Primitive(PType::U64, Nullability::NonNullable))?
            .execute::<PrimitiveArray>(ctx)?;
        let indices = indices.as_slice::<u64>();

        let mut patch_starts = BufferMut::<u64>::with_capacity(group_ranges.len());
        let mut patch_counts = BufferMut::<u64>::with_capacity(group_ranges.len());
        let mut fill_counts = Vec::with_capacity(group_ranges.len());
        for (offset, size) in group_ranges.iter() {
            let start = offset as u64 + patch_offset;
            let first = indices.partition_point(|&index| index < start);
            let count = indices[first..].partition_point(|&index| index < start + size as u64);
            patch_starts.push(first as u64);
            patch_counts.push(count as u64);
            fill_counts.push((size - count) as u64);
        }

        // SAFETY: each group's patches are a range of the patch values, one per group.
        let patch_groups = unsafe {
            ListViewArray::new_unchecked(
                patches.values().clone(),
                patch_starts.into_array(),
                patch_counts.into_array(),
                Validity::from_mask(group_validity.clone(), Nullability::Nullable),
            )
        };
        let mut accumulator =
            GroupedAccumulator::try_new(SumV2, *options, patches.values().dtype().clone())?;
        accumulator.accumulate_list(&patch_groups.into_array(), ctx)?;
        let groups_in_order: Vec<usize> = (0..group_ranges.len()).collect();
        partials.merge(&groups_in_order, &accumulator.flush()?, ctx)?;
        partials.add_repeated(sparse.fill_scalar(), &fill_counts)?;
        Ok(Some(partials.finish(&group_validity)))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::Accumulator;
    use vortex_array::aggregate_fn::AggregateFnVTableExt;
    use vortex_array::aggregate_fn::DynAccumulator;
    use vortex_array::aggregate_fn::DynGroupedAccumulator;
    use vortex_array::aggregate_fn::GroupedAccumulator;
    use vortex_array::aggregate_fn::NumericalAggregateOpts;
    use vortex_array::aggregate_fn::fns::sum::sum;
    use vortex_array::aggregate_fn::fns::sum_v2::SumV2;
    use vortex_array::aggregate_fn::fns::sum_v2::sum_v2;
    use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ListViewArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::scalar::Scalar;
    use vortex_array::session::ArraySessionExt;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_bail;
    use vortex_session::VortexSession;

    use crate::Sparse;
    use crate::SparseArray;
    use crate::initialize;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        initialize(&session);
        session
    });

    static CANONICAL_SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        session.arrays().register(Sparse);
        session
    });

    fn check(array: SparseArray) -> VortexResult<Scalar> {
        let arr = array.into_array();
        let kernel_result = sum(&arr, &mut SESSION.create_execution_ctx())?;
        let canonical_result = sum(&arr, &mut CANONICAL_SESSION.create_execution_ctx())?;
        assert_eq!(
            kernel_result, canonical_result,
            "kernel and canonical sum paths disagree"
        );
        Ok(kernel_result)
    }

    fn run_v2_kernel(arr: &ArrayRef, options: NumericalAggregateOpts) -> VortexResult<Scalar> {
        let aggregate = SumV2.bind(options);
        let Some(kernel) = SESSION
            .aggregate_fns()
            .find_aggregate_kernel(arr.encoding_id(), aggregate.id())
        else {
            vortex_bail!("Sparse SumV2 kernel is not registered");
        };
        let Some(partial) =
            kernel.aggregate(&aggregate, arr, &mut SESSION.create_execution_ctx())?
        else {
            vortex_bail!("Sparse SumV2 kernel declined the aggregate");
        };

        let mut accumulator = Accumulator::try_new(SumV2, options, arr.dtype().clone())?;
        accumulator.combine_partials(&partial)?;
        accumulator.finish()
    }

    fn check_v2(array: SparseArray) -> VortexResult<Scalar> {
        let arr = array.into_array();
        let kernel_result = run_v2_kernel(&arr, NumericalAggregateOpts::default())?;
        let canonical_result = sum_v2(&arr, &mut CANONICAL_SESSION.create_execution_ctx())?;
        assert_eq!(
            kernel_result, canonical_result,
            "kernel and canonical sum_v2 paths disagree"
        );
        Ok(kernel_result)
    }

    #[rstest]
    #[case::positive_fill(
        Sparse::try_new(
            buffer![0u64, 2].into_array(),
            buffer![10i32, 20].into_array(),
            5,
            Scalar::from(1i32),
            &mut SESSION.create_execution_ctx(),
        ).unwrap(),
        // 10 + 1 + 20 + 1 + 1 = 33
        33i64,
    )]
    #[case::zero_fill(
        Sparse::try_new(
            buffer![1u64, 4].into_array(),
            buffer![7i32, 8].into_array(),
            10,
            Scalar::from(0i32),
            &mut SESSION.create_execution_ctx(),
        ).unwrap(),
        15i64,
    )]
    fn sum_kernel_i32(#[case] array: SparseArray, #[case] expected: i64) {
        let result = check(array).unwrap();
        assert_eq!(result.as_primitive().typed_value::<i64>(), Some(expected));
    }

    #[rstest]
    #[case::null_fill_no_overflow(
        Sparse::try_new(
            buffer![0u64, 3].into_array(),
            vortex_array::arrays::PrimitiveArray::from_option_iter([Some(5i64), Some(11)])
                .into_array(),
            6,
            Scalar::null(vortex_array::dtype::DType::Primitive(
                vortex_array::dtype::PType::I64,
                vortex_array::dtype::Nullability::Nullable,
            )),
            &mut SESSION.create_execution_ctx(),
        ).unwrap(),
        16i64,
    )]
    fn sum_kernel_nullable(#[case] array: SparseArray, #[case] expected: i64) {
        let result = check(array).unwrap();
        assert_eq!(result.as_primitive().typed_value::<i64>(), Some(expected));
    }

    #[test]
    fn sum_v2_kernel_nonempty() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = Sparse::try_new(
            buffer![0u64, 2].into_array(),
            buffer![10i32, 20].into_array(),
            5,
            Scalar::from(1i32),
            &mut ctx,
        )?;
        let result = check_v2(array)?;
        assert_eq!(result.as_primitive().typed_value::<i64>(), Some(33));
        Ok(())
    }

    #[test]
    fn sum_v2_kernel_all_null() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = Sparse::try_new(
            buffer![0u64, 3].into_array(),
            vortex_array::arrays::PrimitiveArray::from_option_iter([None::<i32>, None])
                .into_array(),
            6,
            Scalar::null_native::<i32>(),
            &mut ctx,
        )?;
        assert!(check_v2(array)?.is_null());
        Ok(())
    }

    #[test]
    fn sum_v2_kernel_preserves_options() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = Sparse::try_new(
            buffer![0u64].into_array(),
            buffer![1.0f64].into_array(),
            3,
            Scalar::from(f64::NAN),
            &mut ctx,
        )?
        .into_array();
        let result = run_v2_kernel(&array, NumericalAggregateOpts::include_nans())?;
        assert!(
            result
                .as_primitive()
                .typed_value::<f64>()
                .is_some_and(f64::is_nan)
        );
        Ok(())
    }

    fn sparse(
        indices: ArrayRef,
        values: ArrayRef,
        len: usize,
        fill: Scalar,
    ) -> VortexResult<ArrayRef> {
        Ok(Sparse::try_new(
            indices,
            values,
            len,
            fill,
            &mut SESSION.create_execution_ctx(),
        )?
        .into_array())
    }

    fn grouped_sum_v2(elements: &ArrayRef, session: &VortexSession) -> VortexResult<ArrayRef> {
        let mut ctx = session.create_execution_ctx();
        let len = u32::try_from(elements.len())?;
        let groups = ListViewArray::try_new(
            elements.clone(),
            vortex_array::arrays::PrimitiveArray::from_iter([0u32, 0, 2, len - 1, 4, 1])
                .into_array(),
            vortex_array::arrays::PrimitiveArray::from_iter([len, 3, 5, 1, 0, len - 2])
                .into_array(),
            Validity::from_iter([true, true, true, true, true, false]),
            &mut ctx,
        )?
        .into_array();
        let mut accumulator = GroupedAccumulator::try_new(
            SumV2,
            NumericalAggregateOpts::default(),
            elements.dtype().clone(),
        )?;
        accumulator.accumulate_list(&groups, &mut ctx)?;
        accumulator.finish()
    }

    /// Grouped `sum_v2` over sparse elements, summed from the fill and the patches of each group,
    /// matches summing the decoded elements.
    #[rstest]
    #[case::zero_fill_null_patches(sparse(
        buffer![1u32, 3, 4, 8].into_array(),
        vortex_array::arrays::PrimitiveArray::from_option_iter([Some(2u64), None, Some(1), None]).into_array(),
        10,
        Scalar::from(Some(0u64)),
    )?)]
    #[case::true_fill(sparse(
        buffer![0u16, 5, 6].into_array(),
        BoolArray::from_iter([Some(false), None, Some(false)]).into_array(),
        9,
        Scalar::from(Some(true)),
    )?)]
    #[case::null_fill(sparse(
        buffer![2u64, 7].into_array(),
        vortex_array::arrays::PrimitiveArray::from_option_iter([Some(-3i32), Some(9)]).into_array(),
        8,
        Scalar::null_native::<i32>(),
    )?)]
    #[case::fill_overflows(sparse(
        buffer![3u32].into_array(),
        vortex_array::arrays::PrimitiveArray::from_option_iter([Some(1i64)]).into_array(),
        8,
        Scalar::from(Some(i64::MAX / 2)),
    )?)]
    #[case::sliced(sparse(
        buffer![1u32, 3, 4, 8, 11].into_array(),
        vortex_array::arrays::PrimitiveArray::from_option_iter([Some(2u64), None, Some(1), Some(7), Some(5)]).into_array(),
        14,
        Scalar::from(Some(4u64)),
    )?
    .slice(2..13)?)]
    fn grouped_sum_matches_decoded(#[case] elements: ArrayRef) -> VortexResult<()> {
        let result = grouped_sum_v2(&elements, &SESSION)?;
        let expected = grouped_sum_v2(&elements, &CANONICAL_SESSION)?;
        assert_arrays_eq!(&result, &expected, &mut SESSION.create_execution_ctx());
        Ok(())
    }
}
