// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Conversion of [`Sparse`] arrays into [`Patched`] arrays.

use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Patched;
use vortex_array::arrays::PatchedArray;
use vortex_error::VortexResult;

use crate::Sparse;
use crate::SparseExt;

/// Converts a sparse array into the equivalent patched array over its constant fill value.
///
/// The patches are laid out in a single lane per chunk, which keeps them in their sparse order,
/// so the patch values, compressed or not, are used as they are.
pub fn sparse_to_patched(
    sparse: ArrayView<'_, Sparse>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PatchedArray> {
    let fill = ConstantArray::new(sparse.fill_scalar().clone(), sparse.len()).into_array();
    Patched::from_array_and_patches_with_lanes(fill, &sparse.patches(), 1, ctx)
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::compute::conformance::consistency::test_array_consistency;
    use vortex_array::dtype::Nullability;
    use vortex_array::scalar::Scalar;
    use vortex_buffer::buffer;
    use vortex_session::VortexSession;

    use super::sparse_to_patched;
    use crate::Sparse;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    fn sparse(indices: ArrayRef, values: ArrayRef, len: usize, fill: Scalar) -> ArrayRef {
        Sparse::try_new(
            indices,
            values,
            len,
            fill,
            &mut SESSION.create_execution_ctx(),
        )
        .unwrap()
        .into_array()
    }

    #[rstest]
    #[case::null_fill(sparse(
        buffer![2u64, 5, 8].into_array(),
        PrimitiveArray::from_option_iter([Some(100i32), None, Some(300)]).into_array(),
        10,
        Scalar::null_native::<i32>(),
    ))]
    #[case::value_fill(sparse(
        buffer![1u8, 3, 7].into_array(),
        buffer![42i32, 84, 126].into_array(),
        10,
        Scalar::from(0i32),
    ))]
    #[case::large(sparse(
        buffer![100u16, 1500, 3000, 4095].into_array(),
        buffer![1.5f64, 2.5, 3.5, 4.5].into_array(),
        4096,
        Scalar::from(0.0f64),
    ))]
    #[case::strings(sparse(
        buffer![0u32, 9].into_array(),
        VarBinViewArray::from_iter_nullable_str([Some("first value, not inlined"), None])
            .into_array(),
        10,
        Scalar::utf8("fill", Nullability::Nullable),
    ))]
    fn converts_to_equivalent_patched(#[case] array: ArrayRef) {
        let mut ctx = SESSION.create_execution_ctx();
        let Some(sparse) = array.as_opt::<Sparse>() else {
            // Slicing can reduce a sparse array to another encoding.
            return;
        };
        let patched = sparse_to_patched(sparse, &mut ctx).unwrap().into_array();
        assert_arrays_eq!(patched, array, &mut ctx);
        test_array_consistency(&patched, &mut ctx);
    }
}
