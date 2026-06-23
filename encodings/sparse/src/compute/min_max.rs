// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::aggregate_fn::Accumulator;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::DynAccumulator;
use vortex_array::aggregate_fn::fns::min_max::MinMax;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::arrays::ConstantArray;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;

use crate::Sparse;
use crate::SparseExt as _;

/// Sparse-specific min/max kernel.
///
/// `min/max(Sparse{ F, patches })` folds the min/max of `patch_values` together with the
/// fill scalar `F` — but only when `F` is reachable (`P < N`) and valid. The work is
/// `O(P)` instead of `O(N)`.
#[derive(Debug)]
pub(crate) struct SparseMinMaxKernel;

impl DynAggregateKernel for SparseMinMaxKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        let Some(options) = aggregate_fn.as_opt::<MinMax>() else {
            return Ok(None);
        };

        let Some(sparse) = batch.as_opt::<Sparse>() else {
            return Ok(None);
        };

        let patches = sparse.patches();

        let mut acc = Accumulator::try_new(MinMax, *options, batch.dtype().clone())?;

        if !patches.values().is_empty() {
            acc.accumulate(patches.values(), ctx)?;
        }

        // Fold the fill value in only when at least one position is unpatched and the fill
        // is non-null (null fill never participates in min/max).
        if patches.num_patches() < sparse.len() && sparse.fill_scalar().is_valid() {
            let fill_array = ConstantArray::new(sparse.fill_scalar().clone(), 1).into_array();
            acc.accumulate(&fill_array, ctx)?;
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
    use vortex_array::aggregate_fn::NumericalAggregateOpts;
    use vortex_array::aggregate_fn::fns::min_max::MinMaxResult;
    use vortex_array::aggregate_fn::fns::min_max::min_max;
    use vortex_array::scalar::Scalar;
    use vortex_array::session::ArraySessionExt;
    use vortex_buffer::buffer;
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

    #[rstest]
    // fill below all patches
    #[case(Sparse::try_new(buffer![1u64, 3, 5].into_array(), buffer![10i32, 20, 30].into_array(), 8, Scalar::from(1i32)).unwrap())]
    // fill above all patches
    #[case(Sparse::try_new(buffer![1u64, 3, 5].into_array(), buffer![10i32, 20, 30].into_array(), 8, Scalar::from(99i32)).unwrap())]
    // fill in the middle
    #[case(Sparse::try_new(buffer![1u64, 3, 5].into_array(), buffer![10i32, 20, 30].into_array(), 8, Scalar::from(15i32)).unwrap())]
    // every position patched (fill unreachable)
    #[case(Sparse::try_new(buffer![0u64, 1, 2].into_array(), buffer![7i32, 3, 9].into_array(), 3, Scalar::from(99i32)).unwrap())]
    fn min_max_matches_canonical(#[case] array: SparseArray) {
        let arr = array.into_array();
        let kernel: Option<MinMaxResult> = min_max(
            &arr,
            &mut SESSION.create_execution_ctx(),
            NumericalAggregateOpts::default(),
        )
        .unwrap();
        let canonical: Option<MinMaxResult> = min_max(
            &arr,
            &mut CANONICAL_SESSION.create_execution_ctx(),
            NumericalAggregateOpts::default(),
        )
        .unwrap();
        assert_eq!(kernel, canonical);
    }
}
