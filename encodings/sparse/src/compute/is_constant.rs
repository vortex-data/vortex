// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::fns::is_constant::IsConstant;
use vortex_array::aggregate_fn::fns::is_constant::is_constant;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;

use crate::Sparse;
use crate::SparseExt as _;

/// Sparse-specific `is_constant` kernel.
///
/// A `SparseArray` of length `N` with `P` patches and a fill value `F` is constant iff:
/// - `P == 0`: all positions hold `F`.
/// - `0 < P < N`: every patch equals `F`, i.e. `is_constant(patch_values)` AND the common
///   patch value equals `F`.
/// - `P == N`: every position is patched, so the answer is `is_constant(patch_values)`.
///
/// In all cases the work is `O(P)` instead of `O(N)`.
#[derive(Debug)]
pub(crate) struct SparseIsConstantKernel;

impl DynAggregateKernel for SparseIsConstantKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        if !aggregate_fn.is::<IsConstant>() {
            return Ok(None);
        }

        let Some(sparse) = batch.as_opt::<Sparse>() else {
            return Ok(None);
        };

        let patches = sparse.patches();
        let num_patches = patches.num_patches();
        let len = sparse.len();

        let result = if num_patches == 0 {
            // Whole array is the fill value.
            true
        } else if num_patches < len {
            // Mixed: needs all patches equal AND equal to fill.
            if !is_constant(patches.values(), ctx)? {
                false
            } else {
                let first_patch = patches.values().execute_scalar(0, ctx)?;
                &first_patch == sparse.fill_scalar()
            }
        } else {
            // Every position is patched; answer depends purely on patch_values.
            is_constant(patches.values(), ctx)?
        };

        Ok(Some(IsConstant::make_partial(batch, result, ctx)?))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::fns::is_constant::is_constant;
    use vortex_array::scalar::Scalar;
    use vortex_array::session::ArraySession;
    use vortex_array::session::ArraySessionExt;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::Sparse;
    use crate::SparseArray;
    use crate::initialize;

    /// Session with Sparse + its pushdown kernels.
    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty().with::<ArraySession>();
        initialize(&session);
        session
    });

    /// Baseline session: Sparse registered but no pushdown kernels.
    static CANONICAL_SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty().with::<ArraySession>();
        session.arrays().register(Sparse);
        session
    });

    fn check(array: SparseArray) -> VortexResult<bool> {
        let arr = array.into_array();
        let kernel_result = is_constant(&arr, &mut SESSION.create_execution_ctx())?;
        let canonical_result = is_constant(&arr, &mut CANONICAL_SESSION.create_execution_ctx())?;
        assert_eq!(
            kernel_result, canonical_result,
            "kernel and canonical paths disagree"
        );
        Ok(kernel_result)
    }

    #[rstest]
    #[case::all_patches_equal_fill(
        Sparse::try_new(
            buffer![1u64, 3, 5].into_array(),
            buffer![7i32, 7, 7].into_array(),
            10,
            Scalar::from(7i32),
        ).unwrap(),
        true,
    )]
    #[case::mixed_patches_but_unequal_fill(
        Sparse::try_new(
            buffer![1u64, 3].into_array(),
            buffer![9i32, 9].into_array(),
            5,
            Scalar::from(7i32),
        ).unwrap(),
        false,
    )]
    #[case::single_patch_differs(
        Sparse::try_new(
            buffer![1u64].into_array(),
            buffer![3i32].into_array(),
            5,
            Scalar::from(7i32),
        ).unwrap(),
        false,
    )]
    #[case::all_patched_constant(
        Sparse::try_new(
            buffer![0u64, 1, 2, 3].into_array(),
            buffer![5i32, 5, 5, 5].into_array(),
            4,
            Scalar::from(99i32), // fill is unreachable
        ).unwrap(),
        true,
    )]
    #[case::all_patched_not_constant(
        Sparse::try_new(
            buffer![0u64, 1, 2].into_array(),
            buffer![1i32, 2, 3].into_array(),
            3,
            Scalar::from(99i32),
        ).unwrap(),
        false,
    )]
    fn is_constant_kernel(#[case] array: SparseArray, #[case] expected: bool) {
        assert_eq!(check(array).unwrap(), expected);
    }
}
