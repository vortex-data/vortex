// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::scalar_fn::fns::binary::CompareKernel;
use vortex_array::scalar_fn::fns::binary::scalar_cmp;
use vortex_array::scalar_fn::fns::operators::CompareOperator;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_error::VortexResult;

use crate::Sparse;
use crate::SparseExt as _;

/// Sparse-specific compare kernel.
///
/// When the RHS is a constant scalar, the result of any comparison is itself sparse:
/// every unpatched position resolves to `compare(fill, rhs)`, and every patched position
/// to `compare(patch, rhs)`. We push the comparison into the patches and rebuild a
/// `Sparse<Bool>` with the new fill, preserving downstream sparsity (filter masks, etc.).
///
/// For non-constant RHS we decline and let the canonical fallback handle it.
impl CompareKernel for Sparse {
    fn compare(
        lhs: ArrayView<'_, Self>,
        rhs: &ArrayRef,
        operator: CompareOperator,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(rhs_scalar) = rhs.as_constant() else {
            return Ok(None);
        };

        let fill_bool = scalar_cmp(lhs.fill_scalar(), &rhs_scalar, operator)?;
        let patches = lhs.patches();

        let new_patches = patches.map_values(|values| {
            let len = values.len();
            values.binary(
                ConstantArray::new(rhs_scalar.clone(), len).into_array(),
                Operator::from(operator),
            )
        })?;

        Ok(Some(
            Sparse::try_new_from_patches(new_patches, fill_bool)?.into_array(),
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
    use vortex_array::arrays::ConstantArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar_fn::fns::operators::Operator;
    use vortex_array::session::ArraySession;
    use vortex_buffer::buffer;
    use vortex_session::VortexSession;

    use crate::Sparse;
    use crate::SparseArray;
    use crate::initialize;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty().with::<ArraySession>();
        initialize(&session);
        session
    });

    #[rstest]
    #[case::eq_fill(Scalar::from(1i32), Operator::Eq)]
    #[case::eq_patch(Scalar::from(10i32), Operator::Eq)]
    #[case::gt(Scalar::from(5i32), Operator::Gt)]
    #[case::lte(Scalar::from(10i32), Operator::Lte)]
    #[case::neq(Scalar::from(1i32), Operator::NotEq)]
    fn compare_matches_canonical(#[case] rhs: Scalar, #[case] op: Operator) {
        let array: SparseArray = Sparse::try_new(
            buffer![1u64, 3, 5].into_array(),
            buffer![10i32, 20, 30].into_array(),
            8,
            Scalar::from(1i32),
        )
        .unwrap();
        let arr = array.into_array();
        let len = arr.len();
        let mut ctx = SESSION.create_execution_ctx();

        // Kernel path: compare pushes through the Sparse encoding.
        let kernel_bool = arr
            .binary(ConstantArray::new(rhs.clone(), len).into_array(), op)
            .unwrap()
            .execute::<Canonical>(&mut ctx)
            .unwrap();

        // Baseline: canonicalize first, then compare on the PrimitiveArray.
        let canonical_input = arr.execute::<Canonical>(&mut ctx).unwrap().into_array();
        let canonical_bool = canonical_input
            .binary(ConstantArray::new(rhs, len).into_array(), op)
            .unwrap()
            .execute::<Canonical>(&mut ctx)
            .unwrap();

        assert_arrays_eq!(kernel_bool, canonical_bool);
    }
}
