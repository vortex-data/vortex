// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::fill_null::FillNullKernel;
use vortex_error::VortexResult;

use crate::Sparse;
use crate::SparseExt as _;

/// Sparse-specific fill_null kernel.
///
/// `fill_null(Sparse{ F, patches }, v)` replaces nulls in the fill and in each patch value
/// with the (non-null) `v`, staying sparse: the new fill is `v` if `F` was null, else `F`
/// cast to the non-nullable result dtype. The work is `O(P)`.
impl FillNullKernel for Sparse {
    fn fill_null(
        array: ArrayView<'_, Self>,
        fill_value: &Scalar,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let new_fill = if array.fill_scalar().is_null() {
            fill_value.clone()
        } else {
            array.fill_scalar().cast(fill_value.dtype())?
        };

        let new_patches = array
            .patches()
            .map_values(|values| values.fill_null(fill_value.clone()))?;

        Ok(Some(
            Sparse::try_new_from_patches(new_patches, new_fill)?.into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::Canonical;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::scalar::Scalar;
    use vortex_array::session::ArraySession;
    use vortex_buffer::buffer;
    use vortex_session::VortexSession;

    use crate::Sparse;
    use crate::initialize;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty().with::<ArraySession>();
        initialize(&session);
        session
    });

    fn nullable_i32() -> DType {
        DType::Primitive(PType::I32, Nullability::Nullable)
    }

    #[rstest]
    // null fill, some null patches
    #[case(Sparse::try_new(
        buffer![1u64, 3, 5].into_array(),
        PrimitiveArray::from_option_iter([Some(10i32), None, Some(30)]).into_array().cast(nullable_i32()).unwrap(),
        8,
        Scalar::null(nullable_i32()),
    ).unwrap().into_array())]
    // non-null fill, nullable patches with a null
    #[case(Sparse::try_new(
        buffer![0u64, 2].into_array(),
        PrimitiveArray::from_option_iter([Some(7i32), None]).into_array().cast(nullable_i32()).unwrap(),
        4,
        Scalar::from(1i32).cast(&nullable_i32()).unwrap(),
    ).unwrap().into_array())]
    fn fill_null_matches_canonical(#[case] array: vortex_array::ArrayRef) {
        let mut ctx = SESSION.create_execution_ctx();
        let fill = Scalar::from(0i32);

        // Kernel path: fill_null pushes through the Sparse encoding.
        let kernel = array
            .fill_null(fill.clone())
            .unwrap()
            .execute::<Canonical>(&mut ctx)
            .unwrap();

        // Baseline: canonicalize first, then fill_null on the PrimitiveArray.
        let canonical_input = array.execute::<Canonical>(&mut ctx).unwrap().into_array();
        let baseline = canonical_input
            .fill_null(fill)
            .unwrap()
            .execute::<Canonical>(&mut ctx)
            .unwrap();

        assert_arrays_eq!(kernel, baseline);
    }
}
