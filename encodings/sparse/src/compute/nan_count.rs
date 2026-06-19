// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::aggregate_fn::Accumulator;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::AggregateFnVTable as _;
use vortex_array::aggregate_fn::DynAccumulator;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::aggregate_fn::fns::nan_count::NanCount;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::arrays::ConstantArray;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;

use crate::Sparse;
use crate::SparseExt as _;

/// Sparse-specific NaN-count kernel.
///
/// `nan_count(Sparse{ F, patches }) = nan_count(patch_values) + (F is NaN ? N - P : 0)`.
///
/// Declines for non-float dtypes. The work is `O(P)` instead of `O(N)`.
#[derive(Debug)]
pub(crate) struct SparseNanCountKernel;

impl DynAggregateKernel for SparseNanCountKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        if !aggregate_fn.is::<NanCount>() {
            return Ok(None);
        }

        let Some(sparse) = batch.as_opt::<Sparse>() else {
            return Ok(None);
        };

        // NaN count is only defined for floating-point dtypes.
        if NanCount
            .return_dtype(&EmptyOptions, batch.dtype())
            .is_none()
        {
            return Ok(None);
        }

        let patches = sparse.patches();

        let mut acc = Accumulator::try_new(NanCount, EmptyOptions, batch.dtype().clone())?;

        let n_fill = sparse.len() - patches.num_patches();
        if n_fill > 0 {
            // The Constant accumulate path checks `is_nan` once and multiplies by length.
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
    use vortex_array::aggregate_fn::fns::nan_count::nan_count;
    use vortex_array::scalar::Scalar;
    use vortex_array::session::ArraySessionExt;
    use vortex_buffer::buffer;
    use vortex_session::VortexSession;

    use crate::Sparse;
    use crate::SparseArray;
    use crate::initialize;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let mut builder = vortex_array::default_session_builder();
        initialize(&mut builder);
        builder.build()
    });

    static CANONICAL_SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::default_session_builder().build();
        session.arrays().register(Sparse);
        session
    });

    #[rstest]
    // NaN fill value → all unpatched positions are NaN
    #[case(Sparse::try_new(buffer![1u64, 3].into_array(), buffer![1.0f32, 2.0].into_array(), 6, Scalar::from(f32::NAN)).unwrap())]
    // NaN patch values, finite fill
    #[case(Sparse::try_new(buffer![1u64, 3].into_array(), buffer![f32::NAN, 2.0].into_array(), 6, Scalar::from(0.0f32)).unwrap())]
    // no NaNs anywhere
    #[case(Sparse::try_new(buffer![1u64, 3].into_array(), buffer![1.0f32, 2.0].into_array(), 6, Scalar::from(0.0f32)).unwrap())]
    fn nan_count_matches_canonical(#[case] array: SparseArray) {
        let arr = array.into_array();
        let kernel = nan_count(&arr, &mut SESSION.create_execution_ctx()).unwrap();
        let canonical = nan_count(&arr, &mut CANONICAL_SESSION.create_execution_ctx()).unwrap();
        assert_eq!(kernel, canonical);
    }
}
