// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::fns::null_count::NullCount;
use vortex_array::aggregate_fn::fns::null_count::null_count;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;

use crate::Sparse;
use crate::SparseExt as _;

/// Sparse-specific null-count kernel.
///
/// `null_count(Sparse{ F, patches }) = null_count(patch_values) + (F is null ? N - P : 0)`.
///
/// When the fill is non-null this is just the patches' null count (often a cached `O(1)`
/// statistic); either way the work is `O(P)` instead of `O(N)`.
#[derive(Debug)]
pub(crate) struct SparseNullCountKernel;

impl DynAggregateKernel for SparseNullCountKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        if !aggregate_fn.is::<NullCount>() {
            return Ok(None);
        }

        let Some(sparse) = batch.as_opt::<Sparse>() else {
            return Ok(None);
        };

        let patches = sparse.patches();
        let fill_nulls = if sparse.fill_scalar().is_null() {
            (sparse.len() - patches.num_patches()) as u64
        } else {
            0
        };
        let patch_nulls = null_count(patches.values(), ctx)? as u64;

        Ok(Some(Scalar::primitive(
            fill_nulls + patch_nulls,
            NonNullable,
        )))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::fns::null_count::null_count;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
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

    fn nullable_i32() -> DType {
        DType::Primitive(PType::I32, Nullability::Nullable)
    }

    #[rstest]
    // non-null fill, no null patches → 0
    #[case(Sparse::try_new(buffer![1u64, 3].into_array(), buffer![10i32, 20].into_array(), 5, Scalar::from(1i32)).unwrap())]
    // null fill (8 - 2 = 6 fill nulls), patches non-null
    #[case(Sparse::try_new(
        buffer![1u64, 3].into_array(),
        PrimitiveArray::from_option_iter([Some(10i32), Some(20)]).into_array().cast(nullable_i32()).unwrap(),
        8,
        Scalar::null(nullable_i32()),
    ).unwrap())]
    // null fill + some null patches
    #[case(Sparse::try_new(
        buffer![0u64, 2, 4].into_array(),
        PrimitiveArray::from_option_iter([Some(10i32), None, Some(30)]).into_array().cast(nullable_i32()).unwrap(),
        6,
        Scalar::null(nullable_i32()),
    ).unwrap())]
    fn null_count_matches_canonical(#[case] array: SparseArray) {
        let arr = array.into_array();
        let kernel = null_count(&arr, &mut SESSION.create_execution_ctx()).unwrap();
        let canonical = null_count(&arr, &mut CANONICAL_SESSION.create_execution_ctx()).unwrap();
        assert_eq!(kernel, canonical);
    }
}
