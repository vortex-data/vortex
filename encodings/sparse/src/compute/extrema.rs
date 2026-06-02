// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::aggregate_fn::Accumulator;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::DynAccumulator;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::aggregate_fn::fns::max::Max;
use vortex_array::aggregate_fn::fns::min::Min;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::arrays::ConstantArray;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;

use crate::Sparse;
use crate::SparseExt as _;

/// Sparse-specific min/max kernel.
///
/// `min/max(Sparse { fill, patches })` folds the extrema of `patch_values` together with the
/// fill scalar, but only when `fill` is reachable (`patches < len`) and non-null. The work is
/// `O(patches)` instead of `O(len)`.
#[derive(Debug)]
pub(crate) struct SparseExtremaKernel;

impl DynAggregateKernel for SparseExtremaKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        let is_min = aggregate_fn.is::<Min>();
        let is_max = aggregate_fn.is::<Max>();
        if !is_min && !is_max {
            return Ok(None);
        }

        let Some(sparse) = batch.as_opt::<Sparse>() else {
            return Ok(None);
        };

        if is_min {
            Ok(Some(sparse_extremum(Min, sparse, batch, ctx)?))
        } else {
            Ok(Some(sparse_extremum(Max, sparse, batch, ctx)?))
        }
    }
}

fn sparse_extremum<V>(
    vtable: V,
    sparse: ArrayView<'_, Sparse>,
    batch: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Scalar>
where
    V: AggregateFnVTable<Options = EmptyOptions>,
{
    let patches = sparse.patches();
    let mut acc = Accumulator::try_new(vtable, EmptyOptions, batch.dtype().clone())?;

    if !patches.values().is_empty() {
        acc.accumulate(patches.values(), ctx)?;
    }

    // Fold the fill value in only when at least one position is unpatched and the fill is
    // non-null. Null fill never participates in min/max.
    if patches.num_patches() < sparse.len() && sparse.fill_scalar().is_valid() {
        let fill_array = ConstantArray::new(sparse.fill_scalar().clone(), 1).into_array();
        acc.accumulate(&fill_array, ctx)?;
    }

    acc.partial_scalar()
}

#[cfg(test)]
mod tests {
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::AggregateFn;
    use vortex_array::aggregate_fn::EmptyOptions;
    use vortex_array::aggregate_fn::fns::max::Max;
    use vortex_array::aggregate_fn::fns::min::Min;
    use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::scalar::Scalar;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::Sparse;
    use crate::SparseArray;
    use crate::compute::extrema::SparseExtremaKernel;

    fn kernel_extrema(array: &ArrayRef, is_min: bool) -> VortexResult<Option<i32>> {
        let aggregate_fn = if is_min {
            AggregateFn::new(Min, EmptyOptions).erased()
        } else {
            AggregateFn::new(Max, EmptyOptions).erased()
        };
        let scalar = SparseExtremaKernel
            .aggregate(
                &aggregate_fn,
                array,
                &mut LEGACY_SESSION.create_execution_ctx(),
            )?
            .expect("sparse extrema kernel should handle sparse arrays");

        Option::<i32>::try_from(&scalar)
    }

    fn assert_sparse_extrema(
        array: SparseArray,
        expected_min: Option<i32>,
        expected_max: Option<i32>,
    ) -> VortexResult<()> {
        let array = array.into_array();

        assert_eq!(kernel_extrema(&array, true)?, expected_min);
        assert_eq!(kernel_extrema(&array, false)?, expected_max);
        Ok(())
    }

    #[test]
    fn sparse_extrema_kernel_fill_below() -> VortexResult<()> {
        assert_sparse_extrema(
            Sparse::try_new(
                buffer![1u64, 3, 5].into_array(),
                buffer![10i32, 20, 30].into_array(),
                8,
                Scalar::from(1i32),
            )?,
            Some(1),
            Some(30),
        )
    }

    #[test]
    fn sparse_extrema_kernel_fill_above() -> VortexResult<()> {
        assert_sparse_extrema(
            Sparse::try_new(
                buffer![1u64, 3, 5].into_array(),
                buffer![10i32, 20, 30].into_array(),
                8,
                Scalar::from(99i32),
            )?,
            Some(10),
            Some(99),
        )
    }

    #[test]
    fn sparse_extrema_kernel_fill_unreachable() -> VortexResult<()> {
        assert_sparse_extrema(
            Sparse::try_new(
                buffer![0u64, 1, 2].into_array(),
                buffer![7i32, 3, 9].into_array(),
                3,
                Scalar::from(99i32),
            )?,
            Some(3),
            Some(9),
        )
    }

    #[test]
    fn sparse_extrema_kernel_null_fill() -> VortexResult<()> {
        let patch_values = buffer![10i32, 20, 30]
            .into_array()
            .cast(Scalar::null_native::<i32>().dtype().clone())?;

        assert_sparse_extrema(
            Sparse::try_new(
                buffer![1u64, 3, 5].into_array(),
                patch_values,
                8,
                Scalar::null_native::<i32>(),
            )?,
            Some(10),
            Some(30),
        )
    }
}
