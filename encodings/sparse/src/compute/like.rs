// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::scalar_fn::fns::like::Like;
use vortex_array::scalar_fn::fns::like::LikeKernel;
use vortex_array::scalar_fn::fns::like::LikeOptions;
use vortex_error::VortexResult;

use crate::Sparse;
use crate::SparseExt as _;

/// Sparse-specific like kernel.
///
/// With a constant pattern, every unpatched position resolves to `like(fill, pattern)` and every
/// patched position to `like(patch, pattern)`. We push the match into the patches and rebuild a
/// `Sparse<Bool>` with the new fill, preserving downstream sparsity.
///
/// Declines (falls back to canonical) unless the pattern is a constant.
impl LikeKernel for Sparse {
    fn like(
        array: ArrayView<'_, Self>,
        pattern: &ArrayRef,
        options: LikeOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(pattern) = pattern.as_constant() else {
            return Ok(None);
        };

        let fill_bool = Like::try_new(
            ConstantArray::new(array.fill_scalar().clone(), 1).into_array(),
            ConstantArray::new(pattern.clone(), 1).into_array(),
            options,
        )?
        .into_array()
        .execute_scalar(0, ctx)?;

        let new_patches = array.patches().map_values(|values| {
            let len = values.len();
            Ok(Like::try_new(
                values,
                ConstantArray::new(pattern.clone(), len).into_array(),
                options,
            )?
            .into_array())
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
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar_fn::fns::like::Like;
    use vortex_array::scalar_fn::fns::like::LikeOptions;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::Sparse;
    use crate::initialize;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        initialize(&session);
        session
    });

    #[rstest]
    #[case::fill_matches("he%", Scalar::from("hello"), LikeOptions::default())]
    #[case::patch_matches("wor%", Scalar::from("hello"), LikeOptions::default())]
    #[case::negated("he%", Scalar::from("hello"), LikeOptions { negated: true, case_insensitive: false })]
    #[case::case_insensitive("HE%", Scalar::from("hello"), LikeOptions { negated: false, case_insensitive: true })]
    #[case::null_fill(
        "%o%",
        Scalar::null(DType::Utf8(Nullability::Nullable)),
        LikeOptions::default()
    )]
    fn like_matches_canonical(
        #[case] pattern: &str,
        #[case] fill: Scalar,
        #[case] options: LikeOptions,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = VarBinViewArray::from_iter_str(["world", "help", "foo"]).into_array();
        let values = values.cast(fill.dtype().clone())?;
        let array = Sparse::try_new(buffer![1u64, 3, 5].into_array(), values, 8, fill, &mut ctx)?
            .into_array();
        let pattern = ConstantArray::new(pattern, array.len()).into_array();

        // Kernel path: like pushes through the Sparse encoding.
        let kernel = Like::try_new(array.clone(), pattern.clone(), options)?
            .into_array()
            .execute::<Canonical>(&mut ctx)?;

        // Baseline: canonicalize the input first so like runs on a VarBinViewArray.
        let canonical_input = array.execute::<Canonical>(&mut ctx)?.into_array();
        let baseline = Like::try_new(canonical_input, pattern, options)?
            .into_array()
            .execute::<Canonical>(&mut ctx)?;

        assert_arrays_eq!(kernel, baseline, &mut ctx);
        Ok(())
    }
}
