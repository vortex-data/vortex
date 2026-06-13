// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::aggregate_fn::Accumulator;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::DynAccumulator;
use vortex_array::aggregate_fn::fns::sum::Sum;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::arrays::ConstantArray;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;

use crate::Sparse;
use crate::SparseExt as _;

/// Sparse-specific `sum` kernel.
///
/// `sum(Sparse{ F, patches }) = sum(patches.values) + F * (N - patches.num_patches())`.
///
/// The constant contribution is computed via the existing `Sum` accumulator's constant
/// short-circuit (`multiply_constant`), so overflow saturates to null exactly as in the
/// baseline. The work is `O(P)` instead of `O(N)`.
#[derive(Debug)]
pub(crate) struct SparseSumKernel;

impl DynAggregateKernel for SparseSumKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        let Some(options) = aggregate_fn.as_opt::<Sum>() else {
            return Ok(None);
        };

        let Some(sparse) = batch.as_opt::<Sparse>() else {
            return Ok(None);
        };

        let patches = sparse.patches();
        let n_fill = sparse.len() - patches.num_patches();

        // Build a fresh Sum accumulator over the array dtype and fold in the fill and patch
        // contributions. The accumulator's existing semantics (checked overflow → null
        // partial, NaN handling per the options) are preserved.
        let mut acc = Accumulator::try_new(Sum, *options, batch.dtype().clone())?;

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
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::fns::sum::sum;
    use vortex_array::scalar::Scalar;
    use vortex_array::session::ArraySessionExt;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
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
}
