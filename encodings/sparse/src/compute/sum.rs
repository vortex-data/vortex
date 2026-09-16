// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::fns::sum::Sum;
use vortex_array::aggregate_fn::fns::sum_v2::SumV2;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::arrays::ConstantArray;
use vortex_array::scalar::Scalar;
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
    use vortex_array::aggregate_fn::NumericalAggregateOpts;
    use vortex_array::aggregate_fn::fns::sum::sum;
    use vortex_array::aggregate_fn::fns::sum_v2::SumV2;
    use vortex_array::aggregate_fn::fns::sum_v2::sum_v2;
    use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
    use vortex_array::scalar::Scalar;
    use vortex_array::session::ArraySessionExt;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_bail;
    use vortex_session::VortexSession;

    use crate::Sparse;
    use crate::SparseArray;
    use crate::initialize;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        initialize(&session);
        session
    });

    static CANONICAL_SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
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
        accumulator.combine_partials(partial)?;
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
        ).unwrap(),
        16i64,
    )]
    fn sum_kernel_nullable(#[case] array: SparseArray, #[case] expected: i64) {
        let result = check(array).unwrap();
        assert_eq!(result.as_primitive().typed_value::<i64>(), Some(expected));
    }

    #[test]
    fn sum_v2_kernel_nonempty() -> VortexResult<()> {
        let array = Sparse::try_new(
            buffer![0u64, 2].into_array(),
            buffer![10i32, 20].into_array(),
            5,
            Scalar::from(1i32),
        )?;
        let result = check_v2(array)?;
        assert_eq!(result.as_primitive().typed_value::<i64>(), Some(33));
        Ok(())
    }

    #[test]
    fn sum_v2_kernel_all_null() -> VortexResult<()> {
        let array = Sparse::try_new(
            buffer![0u64, 3].into_array(),
            vortex_array::arrays::PrimitiveArray::from_option_iter([None::<i32>, None])
                .into_array(),
            6,
            Scalar::null_native::<i32>(),
        )?;
        assert!(check_v2(array)?.is_null());
        Ok(())
    }

    #[test]
    fn sum_v2_kernel_preserves_options() -> VortexResult<()> {
        let array = Sparse::try_new(
            buffer![0u64].into_array(),
            buffer![1.0f64].into_array(),
            3,
            Scalar::from(f64::NAN),
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
}
